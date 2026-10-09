"""
Explore the 7 channels BEFORE training the CNN.
  Plot 1: average spectrum (Welch) per channel x surface, one line per car
          -> do surfaces look different? does each surface look the same in every car?
  Plot 2: quick test per channel: band energies -> random forest, leave-one-car-out
          -> which channels separate the surfaces on an UNSEEN car?
Input: windows_v2.npz from preprocess_v2.py
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import welch
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score

BASE = "data"                     # PVS folders and all outputs live here; run scripts from the repo root
DATA_FILE = os.path.join(BASE, "windows_v2.npz")
OUT_DIR = os.path.join(BASE, "exploration")
FS = 100
CARS = [1, 2, 3]
NAMES = ["dirt", "cobblestone", "asphalt"]
CAR_COLORS = {1: "#BA7517", 2: "#534AB7", 3: "#1D9E75"}
BANDS = [0.5, 2, 4, 8, 12, 16, 24, 32, 50]          # Hz edges -> 8 bands

os.makedirs(OUT_DIR, exist_ok=True)
d = np.load(DATA_FILE)
X, y, car = d["X"], d["y"], d["car"]
channels = [str(c) for c in d["channels"]]
short = [c.replace("_below_suspension", "") for c in channels]
print(f"X: {X.shape}  channels: {short}")

# spectrum of every window, every channel: (windows, channels, freqs)
freqs, psd = welch(X, fs=FS, nperseg=64, axis=-1)
psd_db = 10 * np.log10(psd + 1e-12)

# ---------------- plot 1: spectra ----------------
fig, axes = plt.subplots(len(channels), 3, figsize=(13, 2.3 * len(channels)),
                         sharex=True, sharey="row")
for i, ch in enumerate(short):
    for k, name in enumerate(NAMES):
        ax = axes[i, k]
        for c in CARS:
            m = (car == c) & (y == k)
            ax.plot(freqs, psd_db[m, i].mean(axis=0), color=CAR_COLORS[c], lw=1.4, label=f"Car {c}")
        if i == 0:
            ax.set_title(name, fontsize=12)
        if k == 0:
            ax.set_ylabel(f"{ch}\npower (dB)", fontsize=9)
        ax.grid(alpha=0.3)
for ax in axes[-1]:
    ax.set_xlabel("frequency (Hz)")
axes[0, 0].legend(fontsize=8)
fig.suptitle("Average spectrum per channel and surface (each line = one car)", y=1.0)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "spectra_channel_surface_car.png"), dpi=130)
plt.show()

# ---------------- plot 2: which channels work on an unseen car ----------------
def band_features(ch_idx):
    """log band energy per channel: (windows, len(ch_idx) * 8)"""
    feats = []
    for i in ch_idx:
        for lo, hi in zip(BANDS[:-1], BANDS[1:]):
            sel = (freqs >= lo) & (freqs < hi)
            feats.append(np.log10(psd[:, i, sel].sum(axis=1) + 1e-12))
    return np.stack(feats, axis=1)


def leave_one_car_out(F):
    scores = []
    for test_car in CARS:
        tr, te = car != test_car, car == test_car
        rf = RandomForestClassifier(n_estimators=200, min_samples_leaf=5, n_jobs=-1, random_state=0)
        rf.fit(F[tr], y[tr])
        scores.append(f1_score(y[te], rf.predict(F[te]), average="macro"))
    return scores


sets = {ch: [i] for i, ch in enumerate(short)}
sets["mag + acc xyz"] = [0, 1, 2, 3]
sets["gyro xyz"] = [4, 5, 6]
sets["all 7"] = list(range(len(channels)))

rows = []
for name, idx in sets.items():
    s = leave_one_car_out(band_features(idx))
    rows.append({"channels": name, "test car 1": s[0], "test car 2": s[1],
                 "test car 3": s[2], "mean": np.mean(s)})
    print(f"{name:<15} F1 per test car: {np.round(s, 3)}  mean {np.mean(s):.3f}")

table = pd.DataFrame(rows).set_index("channels").round(3)
table.to_csv(os.path.join(OUT_DIR, "channel_rf_scores.csv"))

plt.figure(figsize=(9, 4.5))
order = table["mean"].sort_values().index
plt.barh(order, table.loc[order, "mean"], color="#7F77DD")
plt.axvline(1 / 3, ls="--", c="gray", label="chance")
plt.xlabel("macro F1, leave-one-car-out (mean of 3 cars)")
plt.title("Which channels separate surfaces on an unseen car? (band energies + random forest)")
plt.legend(); plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "channel_rf_scores.png"), dpi=130)
plt.show()

print("\n", table.to_string())
print(f"\nfigures saved in {OUT_DIR}")