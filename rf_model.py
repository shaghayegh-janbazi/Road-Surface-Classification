"""
Random forest baseline for road surface classification, sized for a chip.
Same windows, same 4 channels (acc x/y/z + gyro_z) and same leave-one-car-out rounds as the CNN.

1. Feature extraction (per window, per channel)
     time:      RMS, peak-to-peak, kurtosis, crest factor
     frequency: 8 log band energies (Welch)                       -> "base" set, 12 per channel
     extra:     8 relative band energies (band / total, log)      -> "base+relative" set, 20 per channel
2. Tuning INSIDE the training cars only (train on one, validate on the other, both ways):
     a. model type (random forest / extra trees), number of trees, depth, feature set
        -> keep the SMALLEST model within 0.01 macro F1 of the best (chip-friendly)
     b. feature selection: keep the top-K most important features, K chosen the same way
3. Final model per round: train on both training cars, test once on the left-out car, 3 seeds
4. Post-learning smoothing (online): average class probabilities of the current and
     previous 2 windows of the same recording -> reported separately, never tuned on test
5. Chip numbers: tree nodes, estimated memory, comparisons per prediction, latency on this PC

Output: rf_runs/rf_raw and rf_runs/rf_smooth3 (same file format as cnn_runs, so
evaluate_runs.py works with RUNS_DIR = rf_runs), plus chosen settings and importance plots.
"""
import os
import json
import time
import numpy as np
import pandas as pd
from scipy.signal import welch
from scipy.stats import kurtosis
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ======================= settings =======================
BASE = "data"                     # PVS folders and all outputs live here; run scripts from the repo root
DATA_FILE = os.path.join(BASE, "windows_v2.npz")          # same data as the final CNN
OUT_ROOT = os.path.join(BASE, "rf_runs")
CHANNELS = ["acc_x_below_suspension", "acc_y_below_suspension",
            "acc_z_below_suspension", "gyro_z_below_suspension"]
FS = 100
BANDS = [0.5, 2, 4, 8, 12, 16, 24, 32, 50]                 # Hz edges -> 8 bands
CARS = [1, 2, 3]
SEEDS = [0, 1, 2]
NAMES = ["dirt", "cobblestone", "asphalt"]
SMOOTH = 3                                                 # windows averaged for smoothing (fixed, not tuned)
TOLERANCE = 0.01                                           # "as good as the best" margin for picking small models
BYTES_PER_NODE = 10        # feature id 1 B + threshold 4 B + two child indices 2x2 B + class 1 B

# search space (kept small on purpose: every value is a model size you could put on a chip)
MODELS = {"rf": RandomForestClassifier, "extra_trees": ExtraTreesClassifier}
N_TREES = [25, 50, 100]
DEPTHS = [6, 10, 14]
MIN_LEAF = 5
TOP_K = [8, 16, 32, None]                                  # None = keep all features of the chosen set


# ======================= 1. feature extraction =======================
def extract_features(X):
    """X (n, channels, 200) -> F (n, features), feature names, and the column groups."""
    short = [c.replace("_below_suspension", "") for c in CHANNELS]
    rms = np.sqrt((X ** 2).mean(axis=2))
    p2p = X.max(axis=2) - X.min(axis=2)
    kurt = kurtosis(X, axis=2)                               # 0 for Gaussian noise, high for sharp hits
    crest = np.abs(X).max(axis=2) / (rms + 1e-12)
    freqs, psd = welch(X, fs=FS, nperseg=64, axis=-1)        # (n, channels, 33)

    band = np.stack([psd[:, :, (freqs >= lo) & (freqs < hi)].sum(axis=2)
                     for lo, hi in zip(BANDS[:-1], BANDS[1:])], axis=2)   # (n, channels, 8)
    total = band.sum(axis=2, keepdims=True)
    log_band = np.log10(band + 1e-12)
    rel_band = np.log10(band / (total + 1e-12) + 1e-12)      # share of energy: less car-dependent

    cols, names, base_idx, rel_idx = [], [], [], []
    for c, ch in enumerate(short):
        for nm, arr in [("rms", rms), ("p2p", p2p), ("kurtosis", kurt), ("crest", crest)]:
            base_idx.append(len(names)); cols.append(arr[:, c]); names.append(f"{ch}_{nm}")
        for b, (lo, hi) in enumerate(zip(BANDS[:-1], BANDS[1:])):
            base_idx.append(len(names)); cols.append(log_band[:, c, b]); names.append(f"{ch}_band_{lo}-{hi}Hz")
        for b, (lo, hi) in enumerate(zip(BANDS[:-1], BANDS[1:])):
            rel_idx.append(len(names)); cols.append(rel_band[:, c, b]); names.append(f"{ch}_relband_{lo}-{hi}Hz")
    F = np.stack(cols, axis=1).astype(np.float32)
    sets = {"base": base_idx, "base+relative": base_idx + rel_idx}
    return F, names, sets


# ======================= helpers =======================
def make_model(kind, n_trees, depth, seed):
    return MODELS[kind](n_estimators=n_trees, max_depth=depth, min_samples_leaf=MIN_LEAF,
                        n_jobs=-1, random_state=seed)


def n_nodes(model):
    return int(sum(t.tree_.node_count for t in model.estimators_))


def comparisons(model):
    """Worst case: one comparison per level, per tree."""
    return int(sum(t.tree_.max_depth for t in model.estimators_))


def validate(F, y, car, a, b, cols, kind, n_trees, depth):
    """Train on car a -> validate on car b, and the reverse. Returns mean macro F1 and mean node count."""
    scores, nodes = [], []
    for tr, va in [(a, b), (b, a)]:
        m = make_model(kind, n_trees, depth, seed=0).fit(F[car == tr][:, cols], y[car == tr])
        scores.append(f1_score(y[car == va], m.predict(F[car == va][:, cols]), average="macro"))
        nodes.append(n_nodes(m))
    return float(np.mean(scores)), float(np.mean(nodes))


def pick_small(table, score_col, size_col):
    """Smallest option whose score is within TOLERANCE of the best."""
    ok = table[table[score_col] >= table[score_col].max() - TOLERANCE]
    return ok.sort_values([size_col, score_col], ascending=[True, False]).iloc[0]


def top_k_columns(F, y, mask, cols, cfg, k, seed=0):
    """Rank the given columns by importance, using only the rows in mask (training data)."""
    m = make_model(cfg["model"], cfg["trees"], cfg["depth"], seed).fit(F[mask][:, cols], y[mask])
    order = np.argsort(m.feature_importances_)[::-1]
    keep = order if k is None else order[:k]
    return [cols[i] for i in keep]


def smooth_proba(proba, rec):
    """Online smoothing: mean of the current and previous SMOOTH-1 windows of the same recording."""
    out = np.empty_like(proba)
    for r in np.unique(rec):
        idx = np.where(rec == r)[0]                          # windows are stored in time order
        p = proba[idx]
        csum = np.cumsum(np.vstack([np.zeros((1, p.shape[1])), p]), axis=0)
        n = np.arange(1, len(p) + 1)
        lo = np.maximum(0, n - SMOOTH)
        out[idx] = (csum[n] - csum[lo]) / (n - lo)[:, None]
    return out


def latency_ms(model, cols, X_one_window_batch):
    """Feature extraction + prediction for ONE window at a time, single thread, on this PC."""
    model.set_params(n_jobs=1)
    for i in range(10):
        F1, _, _ = extract_features(X_one_window_batch[i:i + 1])
        model.predict(F1[:, cols])
    t = []
    for i in range(min(300, len(X_one_window_batch))):
        t0 = time.perf_counter()
        F1, _, _ = extract_features(X_one_window_batch[i:i + 1])
        model.predict(F1[:, cols])
        t.append((time.perf_counter() - t0) * 1000)
    return float(np.mean(t))


# ======================= main =======================
def main():
    d = np.load(DATA_FILE)
    names_all = [str(c) for c in d["channels"]]
    X = d["X"][:, [names_all.index(c) for c in CHANNELS], :].astype(np.float64)
    y, car, rec = d["y"].astype(int), d["car"].astype(int), d["recording"].astype(int)

    F, feat_names, sets = extract_features(X)
    assert np.isfinite(F).all(), "non-finite feature values"
    print(f"windows: {len(F)} | features: base {len(sets['base'])}, "
          f"base+relative {len(sets['base+relative'])} | channels: {len(CHANNELS)}\n")

    dirs = {k: os.path.join(OUT_ROOT, k) for k in ["rf_raw", "rf_smooth3"]}
    for p in dirs.values():
        os.makedirs(p, exist_ok=True)

    rows = {k: [] for k in dirs}
    cms = {k: {} for k in dirs}
    chosen, size_rows, importances = {}, [], []

    for test_car in CARS:
        a, b = [c for c in CARS if c != test_car]
        tr_mask = car != test_car

        # ---- 2a. model type, size and feature set: validation inside the training cars ----
        grid = []
        for set_name, cols in sets.items():
            for kind in MODELS:
                for n_trees in N_TREES:
                    for depth in DEPTHS:
                        f1, nodes = validate(F, y, car, a, b, cols, kind, n_trees, depth)
                        grid.append({"features": set_name, "model": kind, "trees": n_trees,
                                     "depth": depth, "val_f1": f1, "nodes": nodes})
        grid = pd.DataFrame(grid)
        best = pick_small(grid, "val_f1", "nodes")
        cfg = {"features": best["features"], "model": best["model"],
               "trees": int(best["trees"]), "depth": int(best["depth"])}
        cols_all = sets[cfg["features"]]

        # ---- 2b. feature selection (top-K by importance), also inside the training cars ----
        ksel = []
        for k in TOP_K:
            scores = []
            for tr, va in [(a, b), (b, a)]:
                cols_k = top_k_columns(F, y, car == tr, cols_all, cfg, k)
                m = make_model(cfg["model"], cfg["trees"], cfg["depth"], 0).fit(F[car == tr][:, cols_k], y[car == tr])
                scores.append(f1_score(y[car == va], m.predict(F[car == va][:, cols_k]), average="macro"))
            ksel.append({"k": len(cols_all) if k is None else k, "val_f1": float(np.mean(scores))})
        ksel = pd.DataFrame(ksel)
        cfg["top_k"] = int(pick_small(ksel, "val_f1", "k")["k"])
        cfg["val_f1"] = float(best["val_f1"])
        chosen[f"test_car_{test_car}"] = cfg
        print(f"test car {test_car} | chosen on training cars {a},{b}: {cfg}")

        # ---- 3. final model on both training cars, 3 seeds, tested once ----
        for seed in SEEDS:
            cols = top_k_columns(F, y, tr_mask, cols_all, cfg, cfg["top_k"], seed)
            model = make_model(cfg["model"], cfg["trees"], cfg["depth"], seed).fit(F[tr_mask][:, cols], y[tr_mask])
            te = car == test_car
            proba = model.predict_proba(F[te][:, cols])
            preds = {"rf_raw": proba.argmax(1),
                     "rf_smooth3": smooth_proba(proba, rec[te]).argmax(1)}       # 4. smoothing
            for k, p in preds.items():
                rows[k].append({"run": k, "seed": seed, "test_car": test_car,
                                "macro_f1": f1_score(y[te], p, average="macro"),
                                "accuracy": accuracy_score(y[te], p)})
                cms[k][f"seed{seed}_test{test_car}"] = confusion_matrix(y[te], p, labels=[0, 1, 2]).tolist()

            # ---- 5. chip numbers ----
            nodes = n_nodes(model)
            size_rows.append({"test_car": test_car, "seed": seed, "features": len(cols),
                              "trees": cfg["trees"], "max_depth": cfg["depth"], "nodes": nodes,
                              "est_chip_kb": nodes * BYTES_PER_NODE / 1024,
                              "comparisons_worst_case": comparisons(model),
                              "latency_ms_this_pc": latency_ms(model, cols, X[te]) if seed == 0 else np.nan})
            if seed == 0:
                importances.append(pd.Series(model.feature_importances_, index=[feat_names[i] for i in cols]))

    # ---- save in the same format as cnn_runs ----
    for k, p in dirs.items():
        res = pd.DataFrame(rows[k])
        res.to_csv(os.path.join(p, "results.csv"), index=False)
        with open(os.path.join(p, "confusion_matrices.json"), "w") as f:
            json.dump({"classes": NAMES, "channels": CHANNELS, "matrices": cms[k]}, f, indent=2)
    with open(os.path.join(OUT_ROOT, "chosen_settings.json"), "w") as f:
        json.dump(chosen, f, indent=2)
    sizes = pd.DataFrame(size_rows)
    sizes.to_csv(os.path.join(OUT_ROOT, "model_size.csv"), index=False)

    # ---- report ----
    print("\nRESULTS (macro F1, mean over seeds)")
    for k in dirs:
        res = pd.DataFrame(rows[k])
        per_car = res.groupby("test_car")["macro_f1"].mean().round(3).to_dict()
        per_seed = res.groupby("seed")["macro_f1"].mean()
        print(f"  {k:<11} per car {per_car} | overall {per_seed.mean():.3f} ± {per_seed.std(ddof=0):.3f}")

    print("\nMODEL SIZE (per round, seed 0)")
    print(sizes[sizes["seed"] == 0].round(3).to_string(index=False))
    print(f"\naverage: {sizes['nodes'].mean():.0f} nodes ~ {sizes['est_chip_kb'].mean():.1f} KB on a chip "
          f"(estimate, {BYTES_PER_NODE} bytes/node), "
          f"{sizes['comparisons_worst_case'].mean():.0f} comparisons per prediction (worst case), "
          f"{sizes['latency_ms_this_pc'].mean():.2f} ms per window on this PC incl. feature extraction")
    print("For comparison, CNN int8: ~31 KB weights, ~2 million multiply-adds per window.")

    imp = pd.concat(importances, axis=1).fillna(0).mean(axis=1).sort_values(ascending=False)
    imp.to_csv(os.path.join(OUT_ROOT, "feature_importance.csv"))
    print("\nTOP 10 FEATURES (mean importance over the 3 rounds)")
    print(imp.head(10).round(3).to_string())

    plt.figure(figsize=(8, 6))
    top = imp.head(15)[::-1]
    plt.barh(top.index, top.values, color="#7F77DD")
    plt.xlabel("mean importance"); plt.title("Random forest: most important features")
    plt.tight_layout(); plt.savefig(os.path.join(OUT_ROOT, "feature_importance.png"), dpi=130); plt.close()
    print(f"\nfiles saved in {OUT_ROOT}")


if __name__ == "__main__":
    main()
