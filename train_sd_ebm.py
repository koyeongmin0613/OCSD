import os
import math
import copy
import random
from dataclasses import dataclass
from typing import Optional, Dict, Any, List

import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio

from transformers import Wav2Vec2Model


# -------------------------
# Utils
# -------------------------
def set_seed(seed: int = 0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def l2_normalize(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    return x / (x.norm(dim=-1, keepdim=True) + eps)


# -------------------------
# Augmentations (bonafide-only)
# -------------------------
class AudioAugment(nn.Module):
    """
    간단/안정 위주 augmentation:
    - random gain
    - additive noise (white)
    - time shift (roll)
    - optional speed perturbation (torchaudio.sox_effects, 환경에 따라 비활성)
    """
    def __init__(
        self,
        p_gain: float = 0.8,
        p_noise: float = 0.8,
        p_shift: float = 0.5,
        noise_snr_db: tuple = (10, 30),
        gain_db: tuple = (-6, 6),
        shift_max_ms: int = 100,
        sample_rate: int = 16000,
    ):
        super().__init__()
        self.p_gain = p_gain
        self.p_noise = p_noise
        self.p_shift = p_shift
        self.noise_snr_db = noise_snr_db
        self.gain_db = gain_db
        self.shift_max_ms = shift_max_ms
        self.sr = sample_rate

    @torch.no_grad()
    def forward(self, wav: torch.Tensor) -> torch.Tensor:
        # wav: (T,)
        x = wav

        # random gain
        if random.random() < self.p_gain:
            g = random.uniform(self.gain_db[0], self.gain_db[1])
            x = x * (10 ** (g / 20))

        # random shift
        if random.random() < self.p_shift:
            shift = int((self.shift_max_ms / 1000.0) * self.sr)
            if shift > 0:
                s = random.randint(-shift, shift)
                x = torch.roll(x, shifts=s, dims=0)

        # additive noise by SNR
        if random.random() < self.p_noise:
            snr_db = random.uniform(self.noise_snr_db[0], self.noise_snr_db[1])
            sig_pow = x.pow(2).mean().clamp_min(1e-12)
            noise = torch.randn_like(x)
            noise_pow = noise.pow(2).mean().clamp_min(1e-12)
            # scale noise to match target snr
            k = torch.sqrt(sig_pow / (noise_pow * (10 ** (snr_db / 10))))
            x = x + k * noise

        # clamp to avoid explosion
        x = x.clamp(-1.0, 1.0)
        return x


# -------------------------
# Dataset
# -------------------------
class FTCRobocallDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        root: str,
        metadata_csv: str = "metadata.csv",
        sample_rate: int = 16000,
        max_seconds: float = 4.0,
        col_candidates: List[str] = ("file_name", "path", "audio", "wav", "wav_path"),
        only_en: bool = True,
        max_items: Optional[int] = None,
        seed: int = 0,
    ):
        self.root = root
        self.sr = sample_rate
        self.max_len = int(sample_rate * max_seconds)

        df = pd.read_csv(os.path.join(root, metadata_csv))

        # optional language filter
        if only_en and "language" in df.columns:
            df = df[df["language"] == "en"].reset_index(drop=True)

        # find audio path column
        path_col = None
        for c in col_candidates:
            if c in df.columns:
                path_col = c
                break
        if path_col is None:
            raise ValueError(f"metadata.csv에 오디오 경로 컬럼이 없어요. 후보={col_candidates}")

        self.path_col = path_col
        self.df = df

        if max_items is not None:
            df = df.sample(n=min(max_items, len(df)), random_state=seed).reset_index(drop=True)
            self.df = df

        # cache for speed
        self._resampler_cache: Dict[int, torchaudio.transforms.Resample] = {}

    def __len__(self):
        return len(self.df)

    def _load_wav(self, rel_or_abs: str) -> torch.Tensor:
        p = rel_or_abs
        if not os.path.isabs(p):
            p = os.path.join(self.root, p)
        if not os.path.exists(p):
            raise FileNotFoundError(f"오디오 파일을 못 찾음: {p}")

        wav, sr = torchaudio.load(p)  # (C,T)
        wav = wav.mean(dim=0)  # mono (T,)

        if sr != self.sr:
            if sr not in self._resampler_cache:
                self._resampler_cache[sr] = torchaudio.transforms.Resample(sr, self.sr)
            wav = self._resampler_cache[sr](wav)

        # fix length: truncate or repeat
        if wav.numel() >= self.max_len:
            wav = wav[: self.max_len]
        else:
            # repeat padding
            rep = math.ceil(self.max_len / wav.numel())
            wav = wav.repeat(rep)[: self.max_len]

        return wav

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        row = self.df.iloc[idx]
        wav_path = str(row[self.path_col])
        wav = self._load_wav(wav_path)
        return {"wav": wav}


def collate_fn(batch: List[Dict[str, Any]]) -> torch.Tensor:
    wavs = torch.stack([b["wav"] for b in batch], dim=0)  # (B,T)
    return wavs


# -------------------------
# Model (Wav2Vec2 frozen + simple bridge/backbone)
# -------------------------
class BridgeBackend(nn.Module):
    """
    논문 bridge/back-end를 "간소화" 구현:
    - Wav2Vec2 hidden states (L layers) -> layer-wise projection
    - layer attention weight alpha -> weighted sum feature map
    - ASP-like pooling: mean+std -> embedding
    """
    def __init__(self, w2v_hidden: int = 1024, d1: int = 128, d2: int = 128, emb_dim: int = 160, num_layers: int = 12):
        super().__init__()
        self.num_layers = num_layers
        self.proj1 = nn.ModuleList([nn.Linear(w2v_hidden, d1) for _ in range(num_layers)])
        self.proj2 = nn.ModuleList([nn.Linear(d1, d2) for _ in range(num_layers)])
        self.ln = nn.ModuleList([nn.LayerNorm(d2) for _ in range(num_layers)])

        # attention over layer representations (mean+std concat -> 2*d2)
        self.att_w = nn.Linear(2 * d2, 1, bias=False)

        self.out = nn.Sequential(
            nn.Linear(2 * d2, emb_dim),
            nn.LayerNorm(emb_dim),
        )

    def asp(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, D)
        mu = x.mean(dim=1)
        std = x.std(dim=1).clamp_min(1e-6)
        return torch.cat([mu, std], dim=-1)  # (B, 2D)

    def forward(self, hidden_states: List[torch.Tensor]) -> torch.Tensor:
        # hidden_states: list length L, each (B, T, H)
        assert len(hidden_states) >= self.num_layers, f"hidden_states len={len(hidden_states)} < num_layers={self.num_layers}"

        vs = []
        gs = []
        for i in range(self.num_layers):
            h = hidden_states[i]  # (B,T,H)
            g = self.proj2[i](F.relu(self.proj1[i](h)))
            g = self.ln[i](g)  # (B,T,d2)
            gs.append(g)
            v = self.asp(g)  # (B,2d2)
            vs.append(v)

        V = torch.stack(vs, dim=1)  # (B,L,2d2)
        alpha = F.softmax(self.att_w(V).squeeze(-1), dim=1)  # (B,L)

        # weighted sum feature map
        Fmap = 0.0
        for i in range(self.num_layers):
            Fmap = Fmap + alpha[:, i].unsqueeze(-1).unsqueeze(-1) * gs[i]  # (B,T,d2)

        emb = self.out(self.asp(Fmap))  # (B,emb_dim)
        emb = l2_normalize(emb)
        return emb


class StudentModel(nn.Module):
    def __init__(self, w2v_name: str = "facebook/wav2vec2-xls-r-300m", num_layers: int = 24):
        super().__init__()
        self.w2v = Wav2Vec2Model.from_pretrained(w2v_name, output_hidden_states=True)
        # freeze w2v
        for p in self.w2v.parameters():
            p.requires_grad = False

        # xls-r-300m hidden size is 1024, layers around 24 (확인용)
        self.head = BridgeBackend(w2v_hidden=self.w2v.config.hidden_size, num_layers=num_layers)

    def forward(self, wav: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            out = self.w2v(wav)
            hidden = list(out.hidden_states)
        return self.head(hidden[: self.head.num_layers])


# -------------------------
# Losses
# -------------------------
def bpl_loss(e1: torch.Tensor, e2: torch.Tensor, beta1: float = 20.0) -> torch.Tensor:
    # e1,e2: (B, D), normalized
    cos = (e1 * e2).sum(dim=-1).clamp(-1, 1)
    # log(1 + exp(beta*(1-cos)))
    return torch.log1p(torch.exp(beta1 * (1.0 - cos))).mean()


def oc_bonafide_margin_pull(e: torch.Tensor, w: torch.Tensor, beta2: float = 5.0, m1: float = 0.85) -> torch.Tensor:
    # e: (B,D) normalized, w: (D,) normalized
    cos = (e * w.unsqueeze(0)).sum(dim=-1).clamp(-1, 1)
    # log(1 + exp(beta*(cos - m1)))  -> cos를 m1 이상으로 밀어올림
    return torch.log1p(torch.exp(beta2 * (cos - m1))).mean()


def sd_emb_loss(e_s: torch.Tensor, e_t: torch.Tensor) -> torch.Tensor:
    # 1 - cosine
    cos = (e_s * e_t).sum(dim=-1).clamp(-1, 1)
    return (1.0 - cos).mean()


def variance_reg(e: torch.Tensor, gamma: float = 0.1) -> torch.Tensor:
    # e: (B,D)
    std = e.std(dim=0)
    return F.relu(gamma - std).mean()


@torch.no_grad()
def ema_update(teacher: nn.Module, student: nn.Module, mu: float):
    for pt, ps in zip(teacher.parameters(), student.parameters()):
        pt.data.mul_(mu).add_(ps.data, alpha=(1.0 - mu))


@torch.no_grad()
def ema_centroid_update(w: torch.Tensor, e_batch: torch.Tensor, eta: float = 0.99) -> torch.Tensor:
    # w: (D,), e_batch: (B,D) normalized
    m = e_batch.mean(dim=0)
    m = l2_normalize(m.unsqueeze(0)).squeeze(0)
    w = eta * w + (1.0 - eta) * m
    w = l2_normalize(w.unsqueeze(0)).squeeze(0)
    return w


# -------------------------
# Train
# -------------------------
@dataclass
class TrainConfig:
    root: str
    metadata_csv: str = "metadata.csv"
    w2v_name: str = "facebook/wav2vec2-xls-r-300m"
    num_layers: int = 24

    sample_rate: int = 16000
    max_seconds: float = 4.0

    batch_size: int = 16
    epochs: int = 5
    lr: float = 2e-4
    weight_decay: float = 1e-4

    beta1: float = 20.0
    beta2: float = 5.0
    m1: float = 0.85

    # self-distillation
    ema_mu: float = 0.999
    lambda_sd: float = 0.5

    # one-class pull
    lambda_oc: float = 1.0

    # variance reg (collapse 방지)
    use_var: bool = True
    lambda_var: float = 0.1
    gamma: float = 0.1

    # centroid EMA
    centroid_eta: float = 0.99

    # bpl
    lambda_bpl: float = 1.0

    # augmentation
    crop2_seconds: float = 3.0  # 논문처럼 2번째는 더 짧게 crop 후 augment
    seed: int = 0
    num_workers: int = 0
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


def train(cfg: TrainConfig):
    set_seed(cfg.seed)
    device = torch.device(cfg.device)

    ds = FTCRobocallDataset(
        root=cfg.root,
        metadata_csv=cfg.metadata_csv,
        sample_rate=cfg.sample_rate,
        max_seconds=cfg.max_seconds,
        only_en=True,
        max_items=20000,
        seed=cfg.seed,
    )
    dl = torch.utils.data.DataLoader(
        ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        collate_fn=collate_fn,
        drop_last=True,
        pin_memory=(cfg.device == "cuda"),
    )

    student = StudentModel(w2v_name=cfg.w2v_name, num_layers=cfg.num_layers).to(device)
    teacher = copy.deepcopy(student).to(device)
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad = False

    # 학습 파라미터는 student.head 쪽만 (w2v frozen)
    optim = torch.optim.AdamW(
        [p for p in student.parameters() if p.requires_grad],
        lr=cfg.lr,
        weight_decay=cfg.weight_decay,
    )

    aug = AudioAugment(sample_rate=cfg.sample_rate).to(device)

    # centroid init (처음엔 랜덤보다 "첫 배치 평균"이 안정적이라서 1 epoch 시작 전에 한번 잡음)
    w = torch.zeros(student.head.out[0].out_features, device=device)  # emb_dim
    w = l2_normalize(torch.randn_like(w).unsqueeze(0)).squeeze(0)

    # warm init centroid with first few batches
    with torch.no_grad():
        it = iter(dl)
        for _ in range(min(5, len(dl))):
            wav = next(it).to(device)
            e = student(wav)
            w = ema_centroid_update(w, e, eta=0.0)  # 그냥 평균으로 세팅

    print(f"[Init] centroid norm={w.norm().item():.4f}, device={device}")

    max_len = int(cfg.sample_rate * cfg.max_seconds)
    crop2_len = int(cfg.sample_rate * cfg.crop2_seconds)

    for ep in range(1, cfg.epochs + 1):
        student.train()
        pbar = tqdm(dl, desc=f"Epoch {ep}/{cfg.epochs}")
        losses = []

        for wav in pbar:
            wav = wav.to(device)  # (B,T)

            # BPL pair 생성:
            # x^(1): full-length augment
            x1 = torch.stack([aug(w) for w in wav], dim=0)

            # x^(2): shorter crop then augment then pad back
            # crop 먼저
            if crop2_len < max_len:
                start = torch.randint(0, max_len - crop2_len + 1, (wav.size(0),), device=device)
                x2_list = []
                for i in range(wav.size(0)):
                    c = wav[i, start[i] : start[i] + crop2_len]
                    c = aug(c)
                    # pad/repeat to max_len
                    if c.numel() < max_len:
                        rep = math.ceil(max_len / c.numel())
                        c = c.repeat(rep)[:max_len]
                    x2_list.append(c)
                x2 = torch.stack(x2_list, dim=0)
            else:
                x2 = torch.stack([aug(w) for w in wav], dim=0)

            # student embeddings
            e1 = student(x1)
            e2 = student(x2)
            e = e1  # 대표로 e1을 전체 embedding으로 사용

            # teacher embeddings (no grad)
            with torch.no_grad():
                et = teacher(x1)

            # losses
            Lbpl = bpl_loss(e1, e2, beta1=cfg.beta1)
            Loc = oc_bonafide_margin_pull(e, w, beta2=cfg.beta2, m1=cfg.m1)
            Lsd = sd_emb_loss(e, et)

            Lvar = torch.tensor(0.0, device=device)
            if cfg.use_var:
                Lvar = variance_reg(e, gamma=cfg.gamma)

            L = (
                cfg.lambda_bpl * Lbpl
                + cfg.lambda_oc * Loc
                + cfg.lambda_sd * Lsd
                + (cfg.lambda_var * Lvar if cfg.use_var else 0.0)
            )

            optim.zero_grad(set_to_none=True)
            L.backward()
            torch.nn.utils.clip_grad_norm_([p for p in student.parameters() if p.requires_grad], 5.0)
            optim.step()

            # centroid EMA update (bonafide-only)
            with torch.no_grad():
                w = ema_centroid_update(w, e.detach(), eta=cfg.centroid_eta)

            # teacher EMA update
            with torch.no_grad():
                ema_update(teacher, student, mu=cfg.ema_mu)

            losses.append(L.item())
            pbar.set_postfix({
                "L": f"{np.mean(losses[-50:]):.4f}",
                "bpl": f"{Lbpl.item():.3f}",
                "oc": f"{Loc.item():.3f}",
                "sd": f"{Lsd.item():.3f}",
                "var": f"{Lvar.item():.3f}" if cfg.use_var else "off",
                "cos_w": f"{(e * w.unsqueeze(0)).sum(-1).mean().item():.3f}",
            })

        # epoch save
        ckpt = {
            "student": student.state_dict(),
            "teacher": teacher.state_dict(),
            "centroid": w.detach().cpu(),
            "cfg": cfg.__dict__,
        }
        save_dir = os.path.expanduser("~/checkpoints_ftc")
        os.makedirs(save_dir, exist_ok=True)
        
        save_path = os.path.join(save_dir, f"ftc_bonafide_sd_ebm_ep{ep}.pt")
        torch.save(ckpt, save_path)
        print(f"[Saved] {save_path}")

    print("Done.")


if __name__ == "__main__":
    # 예시: python train_ftc_bonafide_sd_ebm.py --root /path/to/ftc_robocall
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=str, required=True)
    ap.add_argument("--metadata_csv", type=str, default="metadata.csv")
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--m1", type=float, default=0.85)
    ap.add_argument("--lambda_sd", type=float, default=0.5)
    ap.add_argument("--lambda_oc", type=float, default=1.0)
    ap.add_argument("--use_var", action="store_true")
    args = ap.parse_args()

    cfg = TrainConfig(
        root=args.root,
        metadata_csv=args.metadata_csv,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        m1=args.m1,
        lambda_sd=args.lambda_sd,
        lambda_oc=args.lambda_oc,
        use_var=args.use_var,
    )
    train(cfg)

# python train_sd_ebm.py \
#   --root /mnt/c/Users/User/Desktop/Dataset/FTC/dataset/wav \
#   --epochs 30 \
#   --batch_size 16 \
#   --lr 2e-4 \
#   --m1 0.8 \
#   --lambda_sd 0.7 \
#   --lambda_oc 1.0 \
#   --use_var