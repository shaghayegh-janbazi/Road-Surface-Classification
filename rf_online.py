"""
Online replay for the random forest (same simulation as deploy.py Part B, for the forest).

For each round, the final forest is rebuilt with the settings rf_model.py chose
(rf_runs/chosen_settings.json, seed 0; same data, same steps, so it is the same model).
Then each test-car recording is streamed:
  raw samples in chunks of 10 (0.1 s at 100 Hz)
  -> IIR high-pass per channel with sosfilt, filter state zi carried between chunks
  -> 200-sample buffer -> every 1 s: scale -> 48/80 features -> forest -> class
Scale modes:
  "offline": per-recording scale from preprocessing (whole drive known)
  "running": median window RMS of the drive so far, after a 30-window warm-up (what a chip can do)
Smoothing: mean class probability of the current and previous 2 predictions (only past predictions, works online).
Run AFTER rf_model.py. Separate cell is fine: everything it needs is defined here.
"""
import os
import json
import time
import joblib
import numpy as np
import pandas as pd
from scipy.signal import butter, sosfilt, sosfilt_zi, welch
from scipy.stats import kurtosis
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.metrics import f1_score, accuracy_score

# ======================= settings (must match rf_model.py and preprocess_v2.py) =======================
BASE = "data"                     # PVS folders and all outputs live here; run scripts from the repo root
DATA_FILE = os.path.join(BASE, "windows_v2.npz")
RF_DIR = os.path.join(BASE, "rf_runs")
OUT_DIR = os.path.join(RF_DIR, "online")
CHANNELS = ["acc_x_below_suspension", "acc_y_below_suspension",
            "acc_z_below_suspension", "gyro_z_below_suspension"]
FS, WIN, STEP, CHUNK = 100, 200, 100, 10
BANDS = [0.5, 2, 4, 8, 12, 16, 24, 32, 50]
MIN_SPEED = 1.0
MIN_LEAF = 5
WARMUP_WINDOWS = 30
SMOOTH = 3
CARS = [1, 2, 3]
SURFACES = ["dirt_road", "cobblestone_road", "asphalt_road"]
SOS = butter(2, 0.5, btype="highpass", fs=FS, output="sos")
MODELS = {"rf": RandomForestClassifier, "extra_trees": ExtraTreesClassifier}
os.makedirs(OUT_DIR, exist_ok=True)


# ======================= same feature extraction as rf_model.py =======================
def extract_features(X):
    short = [c.replace("_below_suspension", "") for c in CHANNELS]
    rms = np.sqrt((X ** 2).mean(axis=2))
    p2p = X.max(axis=2) - X.min(axis=2)
    kurt = kurtosis(X, axis=2)
    crest = np.abs(X).max(axis=2) / (rms + 1e-12)
    freqs, psd = welch(X, fs=FS, nperseg=64, axis=-1)
    band = np.stack([psd[:, :, (freqs >= lo) & (freqs < hi)].sum(axis=2)
                     for lo, hi in zip(BANDS[:-1], BANDS[1:])], axis=2)
    total = band.sum(axis=2, keepdims=True)
    log_band = np.log10(band + 1e-12)
    rel_band = np.log10(band / (total + 1e-12) + 1e-12)
    cols, names, base_idx, rel_idx = [], [], [], []
    for c, ch in enumerate(short):
        for nm, arr in [("rms", rms), ("p2p", p2p), ("kurtosis", kurt), ("crest", crest)]:
            base_idx.append(len(names)); cols.append(arr[:, c]); names.append(f"{ch}_{nm}")
        for b, (lo, hi) in enumerate(zip(BANDS[:-1], BANDS[1:])):
            base_idx.append(len(names)); cols.append(log_band[:, c, b]); names.append(f"{ch}_band_{lo}-{hi}Hz")
        for b, (lo, hi) in enumerate(zip(BANDS[:-1], BANDS[1:])):
            rel_idx.append(len(names)); cols.append(rel_band[:, c, b]); names.append(f"{ch}_relband_{lo}-{hi}Hz")
    return np.stack(cols, axis=1).astype(np.float32), names, {"base": base_idx, "base+relative": base_idx + rel_idx}


def make_model(kind, n_trees, depth, seed):
    return MODELS[kind](n_estimators=n_trees, max_depth=depth, min_samples_leaf=MIN_LEAF,
                        n_jobs=-1, random_state=seed)


def rebuild_models():
    """Final forest per round, exactly as rf_model.py builds it for seed 0."""
    d = np.load(DATA_FILE)
    names_all = [str(c) for c in d["channels"]]
    X = d["X"][:, [names_all.index(c) for c in CHANNELS], :].astype(np.float64)
    y, car = d["y"].astype(int), d["car"].astype(int)
    F, feat_names, sets = extract_features(X)
    with open(os.path.join(RF_DIR, "chosen_settings.json")) as f:
        chosen = json.load(f)

    models = {}
    for test_car in CARS:
        cfg = chosen[f"test_car_{test_car}"]
        tr = car != test_car
        cols_all = sets[cfg["features"]]
        ranker = make_model(cfg["model"], cfg["trees"], cfg["depth"], 0).fit(F[tr][:, cols_all], y[tr])
        cols = [cols_all[i] for i in np.argsort(ranker.feature_importances_)[::-1][:cfg["top_k"]]]
        model = make_model(cfg["model"], cfg["trees"], cfg["depth"], 0).fit(F[tr][:, cols], y[tr])
        model.set_params(n_jobs=1)                       # one window at a time: one thread is faster
        joblib.dump({"model": model, "cols": cols, "features": [feat_names[i] for i in cols]},
                    os.path.join(OUT_DIR, f"rf_test{test_car}.joblib"))
        models[test_car] = (model, cols)
        print(f"test car {test_car}: rebuilt forest, {len(cols)} features, {cfg}")
    return models, d


# ======================= streaming =======================
def folder_for(n):
    for name in os.listdir(BASE):
        if name.replace(" ", "") == f"PVS{n}":
            return os.path.join(BASE, name)
    raise FileNotFoundError(f"PVS {n} folder not found")


def load_raw(n):
    path = folder_for(n)
    data = pd.read_csv(os.path.join(path, "dataset_gps_mpu_left.csv"), usecols=CHANNELS + ["speed"])
    labels = pd.read_csv(os.path.join(path, "dataset_labels.csv"), usecols=SURFACES)
    onehot = labels[SURFACES].to_numpy()
    surface = np.where(onehot.sum(axis=1) == 1, onehot.argmax(axis=1), -1)
    return data[CHANNELS].to_numpy(np.float64).T, data["speed"].to_numpy(np.float64), surface


def replay(n, model, cols, scale_offline, mode):
    raw, speed, surface = load_raw(n)
    C, N = raw.shape
    zi = [sosfilt_zi(SOS) * raw[c, 0] for c in range(C)]
    buf = np.zeros((C, WIN))
    seen, rms_history, out, recent = 0, [], [], []

    for start in range(0, N - CHUNK + 1, CHUNK):          # every 0.1 s
        chunk = raw[:, start:start + CHUNK]
        filt = np.empty_like(chunk)
        for c in range(C):
            filt[c], zi[c] = sosfilt(SOS, chunk[c], zi=zi[c])
        buf = np.concatenate([buf[:, CHUNK:], filt], axis=1)
        seen += CHUNK
        if seen < WIN or (seen - WIN) % STEP != 0:         # predict once per second
            continue

        w0 = seen - WIN
        with np.errstate(all="ignore"):
            moving = np.nanmean(speed[w0:seen]) >= MIN_SPEED
        lab = surface[w0:seen]
        label = int(lab[0]) if (lab[0] != -1 and np.all(lab == lab[0])) else -1

        if mode == "offline":
            scale = scale_offline
        else:
            if moving:
                rms_history.append(np.sqrt((buf ** 2).mean(axis=1)))
            if len(rms_history) < WARMUP_WINDOWS:
                continue
            scale = np.median(rms_history, axis=0)

        t0 = time.perf_counter()
        F1, _, _ = extract_features((buf / scale[:, None])[None])
        proba = model.predict_proba(F1[:, cols])[0]
        ms = (time.perf_counter() - t0) * 1000
        recent = (recent + [proba])[-SMOOTH:]               # online: current + previous 2
        out.append({"recording": n, "t_s": seen / FS, "moving": moving, "label": label,
                    "pred": int(proba.argmax()), "pred_smooth3": int(np.mean(recent, axis=0).argmax()),
                    "latency_ms": ms})
    return pd.DataFrame(out)


def main():
    models, d = rebuild_models()
    scales, scale_recs = d["scales"], list(d["scale_recordings"])
    names_all = [str(c) for c in d["channels"]]
    idx = [names_all.index(c) for c in CHANNELS]

    rows, timelines = [], []
    for test_car in CARS:
        model, cols = models[test_car]
        for mode in ["offline", "running"]:
            dfs = []
            for n in [3 * (test_car - 1) + k for k in (1, 2, 3)]:
                df = replay(n, model, cols, scales[scale_recs.index(n)][idx], mode)
                df["scale"] = mode
                dfs.append(df)
                print(f"PVS {n} | {mode:<7} scale | {len(df)} predictions")
            df = pd.concat(dfs)
            timelines.append(df)
            scored = df[df["moving"] & (df["label"] >= 0)]          # same windows as offline evaluation
            for col, smooth in [("pred", "none"), ("pred_smooth3", "smooth 3")]:
                rows.append({"test_car": test_car, "scale": mode, "smoothing": smooth,
                             "scored_windows": len(scored),
                             "macro_f1": f1_score(scored["label"], scored[col], average="macro"),
                             "accuracy": accuracy_score(scored["label"], scored[col]),
                             "latency_ms_mean": df["latency_ms"].mean(),
                             "latency_ms_max": df["latency_ms"].max()})

    res = pd.DataFrame(rows)
    res.to_csv(os.path.join(OUT_DIR, "rf_online_results.csv"), index=False)
    pd.concat(timelines).to_csv(os.path.join(OUT_DIR, "rf_online_timeline.csv"), index=False)

    print("\nRANDOM FOREST ONLINE REPLAY (seed 0, all 3 recordings of each test car pooled)")
    table = res.pivot_table(index="test_car", columns=["scale", "smoothing"], values="macro_f1").round(3)
    print(table.to_string())
    mean = res.groupby(["scale", "smoothing"])["macro_f1"].mean().round(3)
    print("\nmean over the 3 test cars:\n", mean.to_string())
    print(f"\nlatency per prediction (scale + features + forest, this PC, Python): "
          f"mean {res['latency_ms_mean'].mean():.2f} ms, worst {res['latency_ms_max'].max():.2f} ms "
          f"(budget 1000 ms)")
    print("CNN for comparison (int8, no smoothing): offline scale ~0.74, running scale ~0.63")
    print(f"files saved in {OUT_DIR}")


if __name__ == "__main__":
    main()
