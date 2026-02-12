import os, random, math
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio

from transformers import Wav2Vec2Model

# -----------------------
# 1) Dataset
# -----------------------
class FTCDataset(torch.utils.data.Dataset):
    def __init__(self, ftc_root, metadata_csv="metadata.csv", max_items=None, only_en=True):
        self.ftc_root = ftc_root
        df = pd.read_csv(os.path.join(ftc_root, metadata_csv))
        if only_en and "language" in df.columns:
            df = df[df["language"] == "en"].reset_index(drop=True)
        if max_items is not None:
            df = df.sample(n=min(max_items, len(df)), random_state=0).reset_index(drop=True)
        self.df = df

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        wav_path = os.path.join(self.ftc_root, row["file_name"])
        wav, sr = torchaudio.load(wav_path)  # [C, T]
        wav = wav.mean(dim=0)  # mono [T]
        return wav, sr, wav_path


def pad_collate(batch, target_sr=16000, max_sec=10.0):
    # resample to target_sr, random crop/pad to fixed length
    max_len = int(target_sr * max_sec)
    wavs = []
    for wav, sr, path in batch:
        if sr != target_sr:
            wav = torchaudio.functional.resample(wav, sr, target_sr)
        if wav.numel() > max_len:
            start = random.randint(0, wav.numel() - max_len)
            wav = wav[start:start+max_len]
        else:
            pad = max_len - wav.numel()
            wav = F.pad(wav, (0, pad))
        wavs.append(wav)
    wavs = torch.stack(wavs, dim=0)  # [B, T]
    return wavs


# -----------------------
# 2) Telephone augmentations
# -----------------------
def biquad_bandpass(wav, sr, low_hz=300.0, high_hz=3400.0):
    # approximate by lowpass(high) then highpass(low)
    wav = torchaudio.functional.lowpass_biquad(wav, sr, cutoff_freq=high_hz)
    wav = torchaudio.functional.highpass_biquad(wav, sr, cutoff_freq=low_hz)
    return wav

def add_noise_snr(wav, snr_db):
    # wav: [B, T]
    # synthetic white noise (fast MVP). Later you can add babble by mixing other utts.
    noise = torch.randn_like(wav)
    wav_power = wav.pow(2).mean(dim=1, keepdim=True) + 1e-12
    noise_power = noise.pow(2).mean(dim=1, keepdim=True) + 1e-12
    target_noise_power = wav_power / (10 ** (snr_db / 10))
    noise = noise * torch.sqrt(target_noise_power / noise_power)
    return wav + noise

def random_resample_cycle(wav, sr=16000):
    # emulate SR mismatch: 16k -> 8k -> 16k randomly
    if random.random() < 0.5:
        wav = torchaudio.functional.resample(wav, sr, 8000)
        wav = torchaudio.functional.resample(wav, 8000, sr)
    return wav

def codec_proxy(wav):
    # lightweight "lossy" proxy:
    # 1) mu-law companding + decompanding (quantization-like)
    # 2) mild clipping
    # wav: [B, T] in [-1,1] roughly
    mu = 255.0
    x = torch.clamp(wav, -1.0, 1.0)
    y = torch.sign(x) * torch.log1p(mu * torch.abs(x)) / math.log1p(mu)
    # quantize
    q = torch.round((y + 1) * (mu / 2)) / (mu / 2) - 1
    x_hat = torch.sign(q) * (1.0 / mu) * ((1 + mu) ** torch.abs(q) - 1)
    # mild clip
    x_hat = torch.clamp(x_hat, -0.8, 0.8)
    return x_hat

def make_view(wav, sr=16000, snr_choices=(0,5,10,15,20), view_id=1):
    # wav: [B,T]
    # telephone bandlimit always
    wav = biquad_bandpass(wav, sr, 300.0, 3400.0)
    # SR mismatch
    wav = random_resample_cycle(wav, sr)

    # different view compositions
    snr = random.choice(snr_choices)
    wav = add_noise_snr(wav, snr)

    if view_id == 2 and random.random() < 0.8:
        wav = codec_proxy(wav)
        wav = biquad_bandpass(wav, sr, 300.0, 3400.0)

    return wav


# -----------------------
# 3) Model: wav2vec2 + head
# -----------------------
class W2V2OneClass(nn.Module):
    def __init__(self, base_name="facebook/wav2vec2-base", proj_dim=256):
        super().__init__()
        self.enc = Wav2Vec2Model.from_pretrained(base_name)
        hid = self.enc.config.hidden_size
        self.proj = nn.Sequential(
            nn.Linear(hid, 512),
            nn.ReLU(),
            nn.Linear(512, proj_dim),
        )

    def forward(self, wav_16k):
        # wav_16k: [B,T], float
        out = self.enc(wav_16k, output_hidden_states=False).last_hidden_state  # [B, L, H]
        z = out.mean(dim=1)  # mean pooling over frames
        z = self.proj(z)     # [B, D]
        return z


# -----------------------
# 4) EMA teacher + losses (VICReg-style)
# -----------------------
@torch.no_grad()
def ema_update(teacher, student, m=0.99):
    for pt, ps in zip(teacher.parameters(), student.parameters()):
        pt.data.mul_(m).add_(ps.data, alpha=1-m)

def cosine_distill(zs, zt):
    zs = F.normalize(zs, dim=-1)
    zt = F.normalize(zt, dim=-1)
    return 1.0 - (zs * zt).sum(dim=-1).mean()

def var_loss(z, gamma=1.0, eps=1e-4):
    std = torch.sqrt(z.var(dim=0) + eps)
    return torch.mean(F.relu(gamma - std))

def cov_loss(z):
    B, D = z.shape
    z = z - z.mean(dim=0, keepdim=True)
    cov = (z.T @ z) / (B - 1)
    off = cov - torch.diag(torch.diag(cov))
    return (off ** 2).sum() / D

@torch.no_grad()
def update_center(center, z, m=0.99):
    # center: [D], z: [B,D]
    batch_center = z.mean(dim=0)
    center.mul_(m).add_(batch_center, alpha=1-m)

def oc_center_loss(z, center):
    return ((z - center.unsqueeze(0)) ** 2).mean()


# -----------------------
# 5) Training loop skeleton
# -----------------------
def train(ftc_root, epochs=5, batch_size=8, device="cuda" if torch.cuda.is_available() else "cpu"):
    ds = FTCDataset(ftc_root, metadata_csv="metadata.csv", only_en=True)
    dl = torch.utils.data.DataLoader(
        ds, batch_size=batch_size, shuffle=True, num_workers=0,
        collate_fn=lambda b: pad_collate(b, target_sr=16000, max_sec=10.0),
        drop_last=True
    )

    student = W2V2OneClass().to(device)
    teacher = W2V2OneClass().to(device)
    teacher.load_state_dict(student.state_dict())
    for p in teacher.parameters():
        p.requires_grad = False

    # (옵션) 처음엔 encoder freeze -> 안정화 후 unfreeze
    for p in student.enc.parameters():
        p.requires_grad = False

    opt = torch.optim.AdamW(filter(lambda p: p.requires_grad, student.parameters()), lr=1e-4, weight_decay=1e-4)

    proj_dim = student.proj[-1].out_features
    center = torch.zeros(proj_dim, device=device)

    lam_var, lam_cov, lam_oc = 10.0, 1.0, 0.1
    ema_m = 0.99

    student.train()
    for ep in range(1, epochs+1):
        for it, wav in enumerate(dl, start=1):
            wav = wav.to(device)

            # two noisy telephone views
            v1 = make_view(wav, sr=16000, view_id=1)
            v2 = make_view(wav, sr=16000, view_id=2)

            zs = student(v1)
            with torch.no_grad():
                zt = teacher(v2)

            loss_inv = cosine_distill(zs, zt)
            loss_var = var_loss(zs)            # student만으로도 충분
            loss_cov = cov_loss(zs)
            loss_oc  = oc_center_loss(zs, center)

            loss = loss_inv + lam_var*loss_var + lam_cov*loss_cov + lam_oc*loss_oc

            opt.zero_grad()
            loss.backward()
            opt.step()

            # EMA + center update
            with torch.no_grad():
                ema_update(teacher, student, m=ema_m)
                update_center(center, zs, m=0.99)

            if it % 50 == 0:
                print(f"[ep {ep} it {it}] loss={loss.item():.4f} inv={loss_inv.item():.4f} var={loss_var.item():.4f} cov={loss_cov.item():.4f} oc={loss_oc.item():.4f}")

        # after a couple epochs, unfreeze encoder for finetune
        if ep == 2:
            for p in student.enc.parameters():
                p.requires_grad = True
            opt = torch.optim.AdamW(student.parameters(), lr=1e-5, weight_decay=1e-4)

    return student, center


# -----------------------
# 6) Scoring (distance-to-center) + optional multi-view TTA
# -----------------------
@torch.no_grad()
def score_batch(model, center, wav, tta_k=1, device="cpu"):
    model.eval()
    wav = wav.to(device)
    scores = []
    for _ in range(tta_k):
        v = make_view(wav, sr=16000, view_id=1) if tta_k > 1 else wav
        z = model(v)
        s = torch.norm(z - center.unsqueeze(0), dim=-1)  # [B]
        scores.append(s)
    return torch.stack(scores, dim=0).mean(dim=0)  # [B]

if __name__ == "__main__":
    import argparse
    import torch
    import multiprocessing as mp

    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        pass

    parser = argparse.ArgumentParser()
    parser.add_argument("--ftc_root", type=str, required=True, help="folder containing metadata.csv and audio-wav-16khz/")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--save_path", type=str, default="ftc_ocsd_student.pt")
    args = parser.parse_args()

    student, center = train(
        ftc_root=args.ftc_root,
        epochs=args.epochs,
        batch_size=args.batch_size,
        device=args.device,
    )

    torch.save({"model": student.state_dict(), "center": center.detach().cpu()}, args.save_path)
    print("Saved:", args.save_path)

