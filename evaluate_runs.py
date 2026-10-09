"""
Full evaluation report for the CNN runs in cnn_runs/.
  1. Comparison table of all runs (macro F1 per test car)
  2. For one chosen run: per car and overall
       accuracy, balanced accuracy, Cohen's kappa,
       precision / recall / F1 / support per class, macro and weighted averages
  3. Plots: confusion matrices (counts and % of true class), per-class metrics per car
  4. Publication checklist (automatic checks + the manual ones still open)
All metrics are computed from the saved confusion matrices (summed over seeds),
so nothing is retrained.
"""
import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

BASE = "data"                     # PVS folders and all outputs live here; run scripts from the repo root
RUNS_DIR = os.path.join(BASE, "cnn_runs")
RUN = "acc_x+acc_y+acc_z+gyro_z__noaug"      # the run to report in detail
NAMES = ["dirt", "cobblestone", "asphalt"]
CARS = [1, 2, 3]
CAR_NAMES = {1: "Car 1 (VW Saveiro)", 2: "Car 2 (Fiat Bravo)", 3: "Car 3 (Fiat Palio)"}
CHANCE = 1 / len(NAMES)
pd.set_option("display.width", 140)


# ======================= metrics from a confusion matrix =======================
def metrics(cm):
    """cm[i, j] = windows of true class i predicted as class j."""
    cm = np.asarray(cm, dtype=float)
    tp = np.diag(cm)
    support = cm.sum(axis=1)                 # true windows per class
    predicted = cm.sum(axis=0)               # predicted windows per class
    total = cm.sum()
    precision = np.divide(tp, predicted, out=np.zeros_like(tp), where=predicted > 0)
    recall = np.divide(tp, support, out=np.zeros_like(tp), where=support > 0)
    f1 = np.divide(2 * precision * recall, precision + recall,
                   out=np.zeros_like(tp), where=(precision + recall) > 0)
    accuracy = tp.sum() / total
    expected = (support * predicted).sum() / total ** 2          # agreement by chance
    kappa = (accuracy - expected) / (1 - expected)
    w = support / total
    return {
        "per_class": pd.DataFrame({"precision": precision, "recall": recall, "f1": f1,
                                   "support": support.astype(int)}, index=NAMES),
        "accuracy": accuracy,
        "balanced_accuracy": recall.mean(),
        "kappa": kappa,
        "macro_precision": precision.mean(), "macro_recall": recall.mean(), "macro_f1": f1.mean(),
        "weighted_f1": (f1 * w).sum(),
    }


def load_run(run):
    path = os.path.join(RUNS_DIR, run)
    res = pd.read_csv(os.path.join(path, "results.csv"))
    with open(os.path.join(path, "confusion_matrices.json")) as f:
        cms = json.load(f)["matrices"]
    return res, cms


def cm_for_car(cms, car):
    """Sum the confusion matrices of all seeds for one test car."""
    keys = [k for k in cms if k.endswith(f"_test{car}")]
    return np.sum([np.array(cms[k]) for k in keys], axis=0)


# ======================= 1. compare all runs =======================
print("=" * 78)
print("1. ALL RUNS: macro F1 per test car (mean over seeds)")
print("=" * 78)
rows = []
for run in sorted(os.listdir(RUNS_DIR)):
    if not os.path.exists(os.path.join(RUNS_DIR, run, "results.csv")):
        continue
    r = pd.read_csv(os.path.join(RUNS_DIR, run, "results.csv"))
    per_car = r.groupby("test_car")["macro_f1"].mean()
    per_seed = r.groupby("seed")["macro_f1"].mean()
    rows.append({"run": run, **{f"car {c}": per_car.get(c, np.nan) for c in CARS},
                 "mean": per_seed.mean(), "± std (seeds)": per_seed.std(ddof=0),
                 "seeds": r["seed"].nunique()})
print(pd.DataFrame(rows).set_index("run").round(3).to_string())

# ======================= 2. detailed report for one run =======================
res, cms = load_run(RUN)
print("\n" + "=" * 78)
print(f"2. DETAILED REPORT: {RUN}   (seeds: {sorted(int(s) for s in res['seed'].unique())})")
print("=" * 78)

summary, per_car_metrics = [], {}
for c in CARS + ["all"]:
    cm = sum(cm_for_car(cms, k) for k in CARS) if c == "all" else cm_for_car(cms, c)
    m = metrics(cm)
    per_car_metrics[c] = (cm, m)
    label = "ALL CARS (pooled)" if c == "all" else f"TEST {CAR_NAMES[c]}"
    print(f"\n--- {label} ---")
    print(pd.DataFrame(cm.astype(int), index=[f"true {n}" for n in NAMES],
                       columns=[f"pred {n}" for n in NAMES]).to_string())
    print()
    print(m["per_class"].round(3).to_string())
    print(f"accuracy {m['accuracy']:.3f} | balanced accuracy {m['balanced_accuracy']:.3f} | "
          f"macro F1 {m['macro_f1']:.3f} | weighted F1 {m['weighted_f1']:.3f} | kappa {m['kappa']:.3f}")
    summary.append({"test": "all" if c == "all" else f"car {c}",
                    "accuracy": m["accuracy"], "balanced_acc": m["balanced_accuracy"],
                    "macro_precision": m["macro_precision"], "macro_recall": m["macro_recall"],
                    "macro_f1": m["macro_f1"], "weighted_f1": m["weighted_f1"], "kappa": m["kappa"]})

summary = pd.DataFrame(summary).set_index("test").round(3)
print("\nSUMMARY")
print(summary.to_string())
out_dir = os.path.join(RUNS_DIR, RUN)
summary.to_csv(os.path.join(out_dir, "report_summary.csv"))

# ======================= 3. plots =======================
fig, axes = plt.subplots(2, 4, figsize=(18, 8.5))
for col, c in enumerate(CARS + ["all"]):
    cm, _ = per_car_metrics[c]
    pct = cm / cm.sum(axis=1, keepdims=True) * 100
    title = "All cars (pooled)" if c == "all" else f"Test {CAR_NAMES[c]}"
    for row, (mat, fmt, sub) in enumerate([(cm, "{:.0f}", "counts"), (pct, "{:.0f}%", "% of true class")]):
        ax = axes[row, col]
        ax.imshow(pct, cmap="Purples", vmin=0, vmax=100)
        for i in range(3):
            for j in range(3):
                ax.text(j, i, fmt.format(mat[i, j]), ha="center", va="center", fontsize=11,
                        color="white" if pct[i, j] > 60 else "black")
        ax.set_xticks(range(3)); ax.set_xticklabels(NAMES, fontsize=9)
        ax.set_yticks(range(3)); ax.set_yticklabels(NAMES, fontsize=9)
        ax.set_xlabel("predicted"); ax.set_ylabel("true")
        ax.set_title(f"{title}\n{sub}", fontsize=10)
plt.suptitle(f"Confusion matrices: {RUN}", y=1.0)
plt.tight_layout()
plt.savefig(os.path.join(out_dir, "report_confusion_matrices.png"), dpi=130)
plt.show()

fig, axes = plt.subplots(1, 3, figsize=(15, 4.2), sharey=True)
for ax, metric in zip(axes, ["precision", "recall", "f1"]):
    x = np.arange(len(NAMES))
    for i, c in enumerate(CARS):
        vals = per_car_metrics[c][1]["per_class"][metric].values
        ax.bar(x + (i - 1) * 0.27, vals, width=0.27, label=CAR_NAMES[c],
               color=["#BA7517", "#534AB7", "#1D9E75"][i])
    ax.axhline(CHANCE, ls="--", c="gray", lw=1)
    ax.set_xticks(x); ax.set_xticklabels(NAMES); ax.set_ylim(0, 1); ax.set_title(metric)
axes[0].set_ylabel("score"); axes[0].legend(fontsize=8)
plt.suptitle("Per-class metrics for each left-out car", y=1.02)
plt.tight_layout()
plt.savefig(os.path.join(out_dir, "report_per_class.png"), dpi=130)
plt.show()

# ======================= 4. publication checklist =======================
print("\n" + "=" * 78)
print("4. PUBLICATION CHECKLIST")
print("=" * 78)
per_seed = res.groupby("seed")["macro_f1"].mean()
car_f1 = summary.loc[[f"car {c}" for c in CARS], "macro_f1"]
min_recall = min(per_car_metrics[c][1]["per_class"]["recall"].min() for c in CARS)

checks = [
    ("at least 3 seeds", res["seed"].nunique() >= 3, f"{res['seed'].nunique()} seed(s)"),
    ("seed spread small (std < 0.03)", res["seed"].nunique() >= 3 and per_seed.std(ddof=0) < 0.03,
     f"std {per_seed.std(ddof=0):.3f}" if res["seed"].nunique() > 1 else "needs >1 seed"),
    ("every test car well above chance (F1 > 0.5)", (car_f1 > 0.5).all(),
     f"worst car {car_f1.min():.3f}"),
    ("no class almost never found (recall > 0.3 everywhere)", min_recall > 0.3,
     f"lowest class recall {min_recall:.3f}"),
    ("gap between cars reported (max - min F1)", True, f"gap {car_f1.max() - car_f1.min():.3f}"),
]
for name, ok, detail in checks:
    print(f"[{'OK' if ok else '!!'}] {name:<52} {detail}")

print(f"\nfiles saved in {out_dir}")