"""
Road surface classification (dirt / cobblestone / asphalt) with a small 1D CNN, v2.
Leave-one-car-out evaluation, 3 rounds x N seeds.

Input : windows_v2.npz from preprocess_v2.py, X (n, 7 channels, 200), per-recording normalized
Train : optional augmentations (training only, never validation/test):
          - STRETCH: random time-stretch -> shifts all frequencies, like a car whose suspension
                     resonance is a few Hz higher or lower (Car 3 peaks ~17 Hz vs ~12 Hz)
          - AMPLITUDE: random gain per window (hurt in run B, off by default)
          - AXIS_FLIP: random sign per axis channel (sensor mounted the other way)
Output: one folder per run, named from the settings, so runs never overwrite each other.
"""
import os
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ======================= settings =======================
BASE = "data"                     # PVS folders and all outputs live here; run scripts from the repo root
DATA_FILE = os.path.join(BASE, "windows_v2.npz")

ACC3 = ["acc_x_below_suspension", "acc_y_below_suspension", "acc_z_below_suspension"]
GYRO_Z = ["gyro_z_below_suspension"]

# defaults; the RUNS list at the bottom overrides them per run
USE_CHANNELS = ACC3 + GYRO_Z     # "all" or a list of channel names
AMPLITUDE = False                # random gain per window
SCALE_RANGE = (0.6, 1.6)
STRETCH = False                  # random time-stretch (frequency shift)
STRETCH_RANGE = (0.8, 1.25)      # >1 stretches (lower frequencies), <1 compresses (higher)
AXIS_FLIP = False

CARS = [1, 2, 3]
SEEDS = [0, 1, 2]                # final run: 3 seeds
N_CLASSES = 3
NAMES = ["dirt", "cobblestone", "asphalt"]

BATCH = 64
EVAL_BATCH = 1024
LR = 1e-3
WEIGHT_DECAY = 1e-4
MAX_EPOCHS = 100
PATIENCE = 10
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def run_name():
    """Folder name built from the current settings."""
    if USE_CHANNELS == "all":
        ch = "all7"
    else:
        ch = "+".join(c.replace("_below_suspension", "") for c in USE_CHANNELS)
    aug = [n for n, on in [("amp", AMPLITUDE), ("stretch", STRETCH), ("flip", AXIS_FLIP)] if on]
    return f"{ch}__{'-'.join(aug) if aug else 'noaug'}"


# ======================= model =======================
class SurfaceCNN(nn.Module):
    """3 conv blocks + global average pooling. in_channels = number of sensor channels."""

    def __init__(self, in_channels, n_classes=N_CLASSES):
        super().__init__()

        def block(c_in, c_out, k):
            return nn.Sequential(
                nn.Conv1d(c_in, c_out, kernel_size=k, padding=k // 2, bias=False),
                nn.BatchNorm1d(c_out),
                nn.ReLU(),
                nn.MaxPool1d(2),
                nn.Dropout(0.2),
            )

        self.features = nn.Sequential(
            block(in_channels, 32, 5),   # (C, 200)  -> (32, 100)
            block(32, 64, 3),            # (32, 100) -> (64, 50)
            block(64, 128, 3),           # (64, 50)  -> (128, 25)
        )
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Linear(128, n_classes)

    def forward(self, x):                # x: (batch, C, 200)
        return self.head(self.pool(self.features(x)).squeeze(-1))


# ======================= data =======================
def load_data(path):
    d = np.load(path)
    X, y, car = d["X"].astype(np.float32), d["y"].astype(np.int64), d["car"].astype(np.int64)
    names = [str(c) for c in d["channels"]]
    if USE_CHANNELS != "all":
        missing = [c for c in USE_CHANNELS if c not in names]
        assert not missing, f"unknown channels {missing}; available: {names}"
        idx = [names.index(c) for c in USE_CHANNELS]
        X, names = X[:, idx, :], [names[i] for i in idx]

    assert X.ndim == 3 and len(X) == len(y) == len(car), "shape mismatch"
    assert not np.isnan(X).any(), "NaN in X: check preprocessing"
    assert set(np.unique(y)) <= set(range(N_CLASSES)), "unexpected label"
    for c in CARS:
        counts = np.bincount(y[car == c], minlength=N_CLASSES)
        assert counts.min() > 0, f"car {c} is missing a class: {counts}"
    return X, y, car, names


def to_tensors(X, y, mean, std):
    """Normalize with training statistics (per channel) and move to the device once."""
    Xn = (X - mean[None, :, None]) / std[None, :, None]
    return torch.from_numpy(Xn.astype(np.float32)).to(DEVICE), torch.from_numpy(y).to(DEVICE)


def batches(X, y, size, shuffle):
    """Fast batching on tensors. Drops the last incomplete batch when shuffling,
    so BatchNorm never sees a batch of one."""
    n = len(X)
    idx = torch.randperm(n, device=X.device) if shuffle else torch.arange(n, device=X.device)
    stop = n - n % size if shuffle else n
    for i in range(0, stop, size):
        j = idx[i:i + size]
        yield X[j], y[j]


# ======================= augmentation (training only) =======================
def time_stretch(xb):
    """Shift all frequencies by one random factor per batch.
    The window is extended with its mirror image (continuous at the joint), resampled,
    and a random 200-sample piece is cut out. Stretch s: a 12 Hz peak moves to 12 / s Hz.
    Note: compressing moves content near 45-50 Hz above Nyquist; linear interpolation
    only partly suppresses that, so keep the range moderate."""
    L = xb.shape[-1]
    s = float(np.exp(np.random.uniform(np.log(STRETCH_RANGE[0]), np.log(STRETCH_RANGE[1]))))
    ext = torch.cat([xb, xb.flip(-1)], dim=-1)                       # (B, C, 2L)
    new_len = max(L, int(round(2 * L * s)))
    res = F.interpolate(ext, size=new_len, mode="linear", align_corners=False)
    start = np.random.randint(0, new_len - L + 1)
    return res[..., start:start + L]


def augment(xb, axis_idx):
    if STRETCH:
        xb = time_stretch(xb)
    if AMPLITUDE:
        lo, hi = SCALE_RANGE
        xb = xb * torch.empty(len(xb), 1, 1, device=xb.device).uniform_(lo, hi)
    if AXIS_FLIP and axis_idx:
        sign = torch.ones(len(xb), xb.shape[1], 1, device=xb.device)
        sign[:, axis_idx] = torch.randint(0, 2, (len(xb), len(axis_idx), 1), device=xb.device) * 2.0 - 1
        xb = xb * sign
    return xb


# ======================= training =======================
def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def train_epoch(model, X, y, loss_fn, opt, axis_idx):
    model.train()
    total = 0.0
    use_aug = STRETCH or AMPLITUDE or AXIS_FLIP
    for xb, yb in batches(X, y, BATCH, shuffle=True):
        if use_aug:
            xb = augment(xb, axis_idx)
        loss = loss_fn(model(xb), yb)
        opt.zero_grad()
        loss.backward()
        opt.step()
        total += loss.item() * len(yb)
    return total / (len(X) - len(X) % BATCH)


@torch.no_grad()
def evaluate(model, X, y, loss_fn):
    """Returns mean loss and predicted classes. Never augmented."""
    model.eval()
    total, preds = 0.0, []
    for xb, yb in batches(X, y, EVAL_BATCH, shuffle=False):
        out = model(xb)
        total += loss_fn(out, yb).item() * len(yb)
        preds.append(out.argmax(1))
    return total / len(X), torch.cat(preds).cpu().numpy()


def fit(X_tr, y_tr, seed, axis_idx, X_va=None, y_va=None, n_epochs=MAX_EPOCHS):
    """With validation data: early stopping on validation loss. Without: exactly n_epochs."""
    set_seed(seed)
    mean, std = X_tr.mean(axis=(0, 2)), X_tr.std(axis=(0, 2)) + 1e-8   # per channel, training only
    Xt, yt = to_tensors(X_tr, y_tr, mean, std)
    Xv, yv = to_tensors(X_va, y_va, mean, std) if X_va is not None else (None, None)

    model = SurfaceCNN(in_channels=X_tr.shape[1]).to(DEVICE)
    loss_fn = nn.CrossEntropyLoss()
    opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    hist = {"train": [], "val": []}
    best_loss, best_epoch, wait = np.inf, n_epochs, 0
    for epoch in range(1, n_epochs + 1):
        hist["train"].append(train_epoch(model, Xt, yt, loss_fn, opt, axis_idx))
        if Xv is None:
            continue
        v, _ = evaluate(model, Xv, yv, loss_fn)
        hist["val"].append(v)
        if v < best_loss:
            best_loss, best_epoch, wait = v, epoch, 0
        else:
            wait += 1
            if wait >= PATIENCE:
                break
    return model, mean, std, best_epoch, hist


def save_loss_plot(hist, best_epoch, title, path):
    plt.figure(figsize=(6, 4))
    plt.plot(range(1, len(hist["train"]) + 1), hist["train"], label="train")
    plt.plot(range(1, len(hist["val"]) + 1), hist["val"], label="validation")
    plt.axvline(best_epoch, ls="--", c="gray", label=f"best epoch {best_epoch}")
    plt.xlabel("epoch"); plt.ylabel("cross-entropy loss"); plt.title(title); plt.legend()
    plt.tight_layout(); plt.savefig(path); plt.close()


# ======================= one run =======================
def main():
    out_dir = os.path.join(BASE, "cnn_runs", run_name())
    os.makedirs(out_dir, exist_ok=True)
    X, y, car, names = load_data(DATA_FILE)
    axis_idx = list(range(len(names))) if "mag" not in names else [i for i, c in enumerate(names) if c != "mag"]
    n_params = sum(p.numel() for p in SurfaceCNN(X.shape[1]).parameters())
    print(f"windows: {len(X)} | channels ({len(names)}): {[c.replace('_below_suspension', '') for c in names]}\n"
          f"parameters: {n_params} | stretch: {STRETCH} | amplitude: {AMPLITUDE} | "
          f"axis flip: {AXIS_FLIP} | seeds: {SEEDS} | device: {DEVICE}\n"
          f"results -> {out_dir}\n")

    rows, cms = [], {}
    loss_fn = nn.CrossEntropyLoss()
    for seed in SEEDS:
        for test_car in CARS:
            a, b = [c for c in CARS if c != test_car]

            # 1. validation inside the training cars, both directions -> number of epochs
            best = []
            for tr_c, va_c in [(a, b), (b, a)]:
                _, _, _, ep, hist = fit(X[car == tr_c], y[car == tr_c], seed, axis_idx,
                                        X[car == va_c], y[car == va_c])
                best.append(ep)
                tag = f"seed{seed}_test{test_car}_train{tr_c}_val{va_c}"
                save_loss_plot(hist, ep, tag, os.path.join(out_dir, f"loss_{tag}.png"))
            n_epochs = max(1, round(np.mean(best)))

            # 2. final model on both training cars, fixed epochs
            tr = car != test_car
            model, mean, std, _, _ = fit(X[tr], y[tr], seed, axis_idx, n_epochs=n_epochs)

            # 3. test once on the left-out car
            te = car == test_car
            Xte, yte = to_tensors(X[te], y[te], mean, std)
            _, y_pred = evaluate(model, Xte, yte, loss_fn)
            f1 = f1_score(y[te], y_pred, average="macro")
            acc = accuracy_score(y[te], y_pred)
            cm = confusion_matrix(y[te], y_pred, labels=list(range(N_CLASSES)))

            rows.append({"run": run_name(), "seed": seed, "test_car": test_car, "val_epochs": best,
                         "final_epochs": n_epochs, "macro_f1": f1, "accuracy": acc})
            cms[f"seed{seed}_test{test_car}"] = cm.tolist()
            print(f"seed {seed} | test car {test_car} | epochs {n_epochs} {best} "
                  f"| macro F1 {f1:.3f} | acc {acc:.3f}")

            torch.save({"state_dict": model.state_dict(), "mean": mean.tolist(), "std": std.tolist(),
                        "channels": names, "n_epochs": n_epochs},
                       os.path.join(out_dir, f"cnn_seed{seed}_test{test_car}.pt"))

    res = pd.DataFrame(rows)
    res.to_csv(os.path.join(out_dir, "results.csv"), index=False)
    with open(os.path.join(out_dir, "confusion_matrices.json"), "w") as f:
        json.dump({"classes": NAMES, "channels": names, "matrices": cms}, f, indent=2)

    per_seed = res.groupby("seed")["macro_f1"].mean()
    print(f"\noverall macro F1: {per_seed.mean():.3f} ± {per_seed.std(ddof=0):.3f}\n")
    return res


# ======================= runs =======================
# each entry: (channels, stretch, amplitude). Edit this list to choose what runs.
RUNS = [
    (ACC3 + GYRO_Z, False, False),     # G: final setup (acc xyz + gyro_z, no augmentation), 3 seeds
]

if __name__ == "__main__":
    all_results = []
    for USE_CHANNELS, STRETCH, AMPLITUDE in RUNS:
        print("=" * 70)
        print(f"RUN: {run_name()}")
        print("=" * 70)
        all_results.append(main())

    summary = (pd.concat(all_results).groupby(["run", "test_car"])["macro_f1"].mean()
               .unstack().round(3))
    summary["mean"] = summary.mean(axis=1).round(3)
    print("SUMMARY (macro F1 per test car)\n", summary.to_string())