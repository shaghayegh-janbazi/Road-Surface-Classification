"""
Preprocessing v2: multi-channel windows + per-recording normalization.

Per recording (PVS 1..9):
  1. read only the needed columns (sensor + speed, surface labels), check row counts
  2. 7 channels: magnitude, acc x/y/z, gyro x/y/z (below suspension)
  3. each channel: Butterworth high-pass 0.5 Hz, order 2, sosfilt (works online)
  4. cut windows (200 samples, step 100)
  5. per-recording scale: median RMS of moving windows, per channel (no labels used)
  6. keep windows that are moving (mean speed >= 1 m/s) and contain one surface only
Output: windows_v2.npz (v1 file is not overwritten)
"""
import os
import re
import warnings
import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view
from scipy.signal import butter, sosfilt, sosfilt_zi

# ======================= settings =======================
BASE = "data"                     # PVS folders and all outputs live here; run scripts from the repo root
OUT_FILE = os.path.join(BASE, "windows_v2.npz")
RECORDINGS = [1, 2, 3, 4, 5, 6, 7, 8, 9]
FS = 100
WIN = 200
STEP = 100
MIN_SPEED = 1.0                      # m/s
NORMALIZE_PER_RECORDING = True       # fix 1: set False to compare without it
POS = "below_suspension"
SURFACES = ["dirt_road", "cobblestone_road", "asphalt_road"]   # -> 0, 1, 2
ACC = [f"acc_{a}_{POS}" for a in "xyz"]
GYRO = [f"gyro_{a}_{POS}" for a in "xyz"]
CHANNELS = ["mag"] + ACC + GYRO      # 7 channels, order matters
SOS = butter(2, 0.5, btype="highpass", fs=FS, output="sos")


# ======================= functions =======================
def find_folders(base, wanted):
    """Map PVS number -> folder path, for folders named 'PVS 1' or 'PVS1'."""
    folders = {}
    for name in os.listdir(base):
        m = re.fullmatch(r"PVS\s*(\d+)", name)
        if m and int(m.group(1)) in wanted:
            folders[int(m.group(1))] = os.path.join(base, name)
    missing = sorted(set(wanted) - set(folders))
    if missing:
        raise FileNotFoundError(f"missing PVS folders: {missing}")
    return folders


def load_recording(path, n):
    """Read only the needed columns; return sensor (samples, 6), speed, surface label per sample."""
    data = pd.read_csv(os.path.join(path, "dataset_gps_mpu_left.csv"), usecols=ACC + GYRO + ["speed"])
    labels = pd.read_csv(os.path.join(path, "dataset_labels.csv"), usecols=SURFACES)
    if len(data) != len(labels):
        raise ValueError(f"PVS {n}: {len(data)} data rows vs {len(labels)} label rows")

    raw = data[ACC + GYRO].to_numpy(dtype=np.float64)
    if np.isnan(raw).any():
        raise ValueError(f"PVS {n}: {np.isnan(raw).sum()} NaN values in sensor columns")

    onehot = labels[SURFACES].to_numpy()
    surface = np.where(onehot.sum(axis=1) == 1, onehot.argmax(axis=1), -1)   # -1 = none or several
    speed = data["speed"].to_numpy(dtype=np.float64)
    return raw, speed, surface


def highpass(x):
    """IIR high-pass with sosfilt (online), started in steady state to avoid a startup spike."""
    out, _ = sosfilt(SOS, x, zi=sosfilt_zi(SOS) * x[0])
    return out


def make_windows(sig, speed, surface):
    """sig (channels, samples) -> windows (n, channels, WIN), mean speed (n,), labels (n, WIN)."""
    starts = np.arange(0, sig.shape[1] - WIN + 1, STEP)
    win = sliding_window_view(sig, WIN, axis=1)[:, starts, :].transpose(1, 0, 2)
    with warnings.catch_warnings():                       # windows with no GPS value give NaN speed
        warnings.simplefilter("ignore", category=RuntimeWarning)
        spd = np.nanmean(sliding_window_view(speed, WIN)[starts], axis=1)
    sur = sliding_window_view(surface, WIN)[starts]
    return win, spd, sur


# ======================= main =======================
def main():
    folders = find_folders(BASE, RECORDINGS)
    X, y, cars, recs, speeds, scales = [], [], [], [], [], []

    for n in sorted(folders):
        raw, speed, surface = load_recording(folders[n], n)
        mag = np.sqrt((raw[:, :3] ** 2).sum(axis=1))
        sig = np.stack([highpass(mag)] + [highpass(raw[:, i]) for i in range(raw.shape[1])])

        win, spd, sur = make_windows(sig, speed, surface)
        moving = spd >= MIN_SPEED                          # NaN speed -> False
        single = (sur[:, 0] != -1) & np.all(sur == sur[:, :1], axis=1)

        # fix 1: per-recording, per-channel scale from moving windows (labels not used)
        rms = np.sqrt((win[moving] ** 2).mean(axis=2))     # (n_moving, channels)
        scale = np.maximum(np.median(rms, axis=0), 1e-6)   # avoid division by zero
        scales.append(scale)
        if NORMALIZE_PER_RECORDING:
            win = win / scale[None, :, None]

        keep = moving & single
        X.append(win[keep].astype(np.float32))
        y.append(sur[keep, 0])
        cars.append(np.full(keep.sum(), (n - 1) // 3 + 1))
        recs.append(np.full(keep.sum(), n))
        speeds.append(spd[keep])
        print(f"PVS {n}: {raw.shape[0]} rows, {keep.sum()} windows kept")

    X = np.concatenate(X)
    y = np.concatenate(y).astype(np.int64)
    cars = np.concatenate(cars)
    recs = np.concatenate(recs)
    speeds = np.concatenate(speeds).astype(np.float32)
    scales = np.array(scales)

    print("\nX:", X.shape, "(windows, channels, samples)   y:", y.shape)
    print("\nPer-recording scale (median window RMS before normalization):")
    print(pd.DataFrame(scales, index=[f"PVS {n}" for n in sorted(folders)],
                       columns=CHANNELS).round(3).to_string())

    names = ["dirt", "cobblestone", "asphalt"]
    print("\nWindows per car and surface:")
    print(pd.DataFrame({k: [np.sum((cars == c) & (y == i)) for c in [1, 2, 3]]
                        for i, k in enumerate(names)}, index=["Car 1", "Car 2", "Car 3"]).to_string())
    print("\nMean speed (m/s) per car and surface:")
    print(pd.DataFrame({k: [speeds[(cars == c) & (y == i)].mean() for c in [1, 2, 3]]
                        for i, k in enumerate(names)}, index=["Car 1", "Car 2", "Car 3"]).round(2).to_string())

    np.savez(OUT_FILE, X=X, y=y, car=cars, recording=recs, speed=speeds,
             channels=np.array(CHANNELS), scales=scales,
             scale_recordings=np.array(sorted(folders)))
    print(f"\nsaved {OUT_FILE}")


if __name__ == "__main__":
    main()