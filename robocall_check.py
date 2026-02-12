import os, random, glob
import soundfile as sf
import numpy as np

from scipy.signal import welch

root = "/Users/koyoungmin/Desktop/SCAI/Dataset/robocall-audio-dataset-main/audio-wav-16khz"
paths = glob.glob(os.path.join(root, "**", '*.wav'), recursive=True)
sample = random.sample(paths, min(200, len(paths)))

srs, chs, durs = [], [], []
for p in sample:
    info = sf.info(p)
    srs.append(info.samplerate)
    chs.append(info.channels)
    durs.append(info.duration)

print("SR unique:", sorted(set(srs)))
print("CH unique:", sorted(set(chs)))
print("Duration sec: mean", np.mean(durs), "p50", np.median(durs), "p95", np.percentile(durs, 95))

def band_energy_ratio(x, sr, f1, f2):
    f, Pxx = welch(x, fs=sr, nperseg=min(4096, len(x)))
    mask = (f >= f1) & (f <= f2)
    return np.trapz(Pxx[mask], f[mask])

def analyze_one(p):
    x, sr = sf.read(p)
    if x.ndim > 1: x = x.mean(axis=1)
    x = x.astype(np.float32)
    e_total = band_energy_ratio(x, sr, 0, sr/2)
    e_hi = band_energy_ratio(x, sr, 4000, sr/2) if sr/2 > 4000 else 0.0
    e_mid = band_energy_ratio(x, sr, 300, 3400) if sr/2 > 3400 else band_energy_ratio(x, sr, 300, sr/2)
    return sr, e_mid / (e_total + 1e-12), e_hi / (e_total + 1e-12)

rows = [analyze_one(p) for p in sample[:100]]
srs = [r[0] for r in rows]
mid_ratio = [r[1] for r in rows]
hi_ratio = [r[2] for r in rows]
print("SR unique:", sorted(set(srs)))
print("Mid(300~3400)/Total mean:", float(np.mean(mid_ratio)))
print("High(>=4k)/Total mean:", float(np.mean(hi_ratio)))
