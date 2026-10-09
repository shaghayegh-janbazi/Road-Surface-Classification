"""
Publication figures and README tables for the road surface classification project.
Reads only saved results (no training). Run after preprocess_v2, cnn_v2, deploy, rf_model, rf_online.

Figures (saved in figures/):
  fig1_dataset.png            windows per car and surface
  fig2_spectra.png            average spectrum per channel x surface, one line per car
  fig3_cnn_confusion.png      CNN confusion matrices per left-out car (3 seeds summed)
  fig4_rf_confusion.png       random forest confusion matrices per left-out car
  fig5_offline_comparison.png CNN vs RF macro F1 per left-out car, seed points shown
  fig6_int8.png               FP32 vs int8: macro F1 per car, and model size
  fig7_online.png             offline vs running scale, CNN vs RF
  fig8_timeline.png           one test drive second by second: true surface, CNN, RF
  fig9_rf_importance.png      top random forest features
  fig10_cnn_loss_curve.png    copy of one CNN validation loss curve
Tables (results/summary_tables.md): markdown, ready to paste into the README.
"""
import os
import json
import shutil
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
from scipy.signal import welch

# ======================= paths =======================
BASE = "data"                     # PVS folders and all outputs live here; run scripts from the repo root
DATA_FILE = os.path.join(BASE, "windows_v2.npz")
CNN_RUN = os.path.join(BASE, "cnn_runs", "acc_x+acc_y+acc_z+gyro_z__noaug")
DEPLOY = os.path.join(BASE, "deploy")
RF = os.path.join(BASE, "rf_runs")
FIG = "figures"                     # repo folder used by README.md
TAB = "results"
TIMELINE_PVS = 7                     # drive shown in fig8 (PVS 7 = Car 3, the hardest car)
SHOW = False                         # True: also display each figure (close its window to see the next one)

# ======================= style (validated colors) =======================
CHANNELS = ["acc_x_below_suspension", "acc_y_below_suspension",
            "acc_z_below_suspension", "gyro_z_below_suspension"]
SHORT = ["acc_x", "acc_y", "acc_z", "gyro_z"]
NAMES = ["dirt", "cobblestone", "asphalt"]
CARS = [1, 2, 3]
CAR_LABEL = {1: "Car 1 (VW Saveiro)", 2: "Car 2 (Fiat Bravo)", 3: "Car 3 (Fiat Palio)"}
CAR_COLOR = {1: "#2a78d6", 2: "#eb6834", 3: "#1baf7a"}          # categorical slots 1-3
MODEL_COLOR = {"CNN": "#6250d6", "RF": "#008300"}                # violet / green
CLASS_COLOR = ["#eda100", "#e34948", "#2a78d6"]                   # dirt / cobblestone / asphalt
INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#e6e5e1", "#fcfcfb"
CHANCE = 1 / 3

plt.rcParams.update({
    "figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF,
    "font.size": 10, "axes.titlesize": 11, "axes.labelsize": 10,
    "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "axes.axisbelow": True,
    "legend.frameon": False, "lines.linewidth": 2,
})
os.makedirs(FIG, exist_ok=True)
os.makedirs(TAB, exist_ok=True)


def save(fig, name):
    path = os.path.join(FIG, name)
    fig.savefig(path, dpi=200, bbox_inches="tight")
    print("saved", path)
    if SHOW:
        plt.show()
    plt.close(fig)


def load_cms(run_dir):
    with open(os.path.join(run_dir, "confusion_matrices.json")) as f:
        cms = json.load(f)["matrices"]
    return {c: np.sum([np.array(v) for k, v in cms.items() if k.endswith(f"_test{c}")], axis=0) for c in CARS}


def macro_f1_from_cm(cm):
    tp = np.diag(cm).astype(float)
    prec = np.divide(tp, cm.sum(0), out=np.zeros_like(tp), where=cm.sum(0) > 0)
    rec = np.divide(tp, cm.sum(1), out=np.zeros_like(tp), where=cm.sum(1) > 0)
    f1 = np.divide(2 * prec * rec, prec + rec, out=np.zeros_like(tp), where=(prec + rec) > 0)
    return f1.mean(), rec


# ======================= load =======================
d = np.load(DATA_FILE)
names_all = [str(c) for c in d["channels"]]
X = d["X"][:, [names_all.index(c) for c in CHANNELS], :]
y, car = d["y"].astype(int), d["car"].astype(int)

cnn_res = pd.read_csv(os.path.join(CNN_RUN, "results.csv"))
rf_res = pd.read_csv(os.path.join(RF, "rf_raw", "results.csv"))
rf_smooth = pd.read_csv(os.path.join(RF, "rf_smooth3", "results.csv"))
cnn_cm, rf_cm = load_cms(CNN_RUN), load_cms(os.path.join(RF, "rf_raw"))
int8 = pd.read_csv(os.path.join(DEPLOY, "onnx_int8_comparison.csv"))
cnn_on = pd.read_csv(os.path.join(DEPLOY, "online_replay_results.csv"))
rf_on = pd.read_csv(os.path.join(RF, "online", "rf_online_results.csv"))
rf_size = pd.read_csv(os.path.join(RF, "model_size.csv"))

# ======================= fig1: dataset =======================
fig, ax = plt.subplots(figsize=(7, 3.6))
w = 0.26
for k, name in enumerate(NAMES):
    counts = [np.sum((car == c) & (y == k)) for c in CARS]
    bars = ax.bar(np.arange(3) + (k - 1) * (w + 0.02), counts, width=w, color=CLASS_COLOR[k], label=name)
    ax.bar_label(bars, labels=[f"{v:,}" for v in counts], fontsize=8, color=INK2, padding=2)
ax.set_xticks(range(3)); ax.set_xticklabels([CAR_LABEL[c] for c in CARS])
ax.set_ylabel("2-s windows"); ax.grid(axis="x", visible=False)
ax.set_title(f"Dataset after preprocessing: {len(y):,} windows, below-suspension sensor", loc="left")
ax.legend(ncol=3, loc="upper left", bbox_to_anchor=(0, -0.12))
save(fig, "fig1_dataset.png")

# ======================= fig2: spectra =======================
freqs, psd = welch(X, fs=100, nperseg=64, axis=-1)
psd_db = 10 * np.log10(psd + 1e-12)
fig, axes = plt.subplots(4, 3, figsize=(11, 9.5), sharex=True, sharey="row")
for i, ch in enumerate(SHORT):
    for k, name in enumerate(NAMES):
        ax = axes[i, k]
        ax.axvspan(40, 45, color=GRID, alpha=0.9, lw=0)
        for c in CARS:
            m = (car == c) & (y == k)
            ax.plot(freqs, psd_db[m, i].mean(axis=0), color=CAR_COLOR[c], lw=1.6, label=CAR_LABEL[c])
        if i == 0:
            ax.set_title(name)
        if k == 0:
            ax.set_ylabel(f"{ch}\npower (dB, relative)")
for ax in axes[-1]:
    ax.set_xlabel("frequency (Hz)")
handles = [plt.Line2D([], [], color=CAR_COLOR[c], lw=2) for c in CARS] + [Patch(color=GRID)]
fig.legend(handles, [CAR_LABEL[c] for c in CARS] + ["40-45 Hz (cobblestone cue)"],
           ncol=4, loc="lower center", bbox_to_anchor=(0.5, -0.02))
fig.suptitle("Average spectrum per channel and surface (Welch, 2-s windows); each line is one car",
             x=0.01, ha="left", fontsize=12)
fig.tight_layout(rect=(0, 0.03, 1, 0.98))
save(fig, "fig2_spectra.png")


# ======================= fig3 / fig4: confusion matrices =======================
def confusion_figure(cms, title, fname):
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.9))
    for ax, c in zip(axes, CARS):
        cm = cms[c]
        pct = cm / cm.sum(axis=1, keepdims=True) * 100
        ax.imshow(pct, cmap="Blues", vmin=0, vmax=100)
        for i in range(3):
            for j in range(3):
                ax.text(j, i, f"{pct[i, j]:.0f}%\n({cm[i, j]:,})", ha="center", va="center", fontsize=8.5,
                        color="white" if pct[i, j] > 55 else INK)
        f1, _ = macro_f1_from_cm(cm)
        ax.set_xticks(range(3)); ax.set_xticklabels(NAMES)
        ax.set_yticks(range(3)); ax.set_yticklabels(NAMES)
        ax.set_xlabel("predicted"); ax.grid(False)
        ax.set_title(f"Test: {CAR_LABEL[c]}\nmacro F1 {f1:.3f}", fontsize=10)
    axes[0].set_ylabel("true surface")
    fig.suptitle(title, x=0.01, ha="left", fontsize=12)
    fig.tight_layout()
    save(fig, fname)


confusion_figure(cnn_cm, "1D CNN, leave-one-car-out (3 seeds summed); % of each true surface, window counts in brackets",
                 "fig3_cnn_confusion.png")
confusion_figure(rf_cm, "Random forest, leave-one-car-out (3 seeds summed); % of each true surface, window counts in brackets",
                 "fig4_rf_confusion.png")

# ======================= fig5: offline comparison =======================
fig, ax = plt.subplots(figsize=(7.5, 4))
w = 0.34
for j, (label, res) in enumerate([("CNN", cnn_res), ("RF", rf_res)]):
    means = res.groupby("test_car")["macro_f1"].mean().reindex(CARS)
    x = np.arange(3) + (j - 0.5) * (w + 0.03)
    bars = ax.bar(x, means.values, width=w, color=MODEL_COLOR[label],
                  label=f"{'1D CNN' if label == 'CNN' else 'Random forest'}  "
                        f"(overall {res.groupby('seed')['macro_f1'].mean().mean():.3f})")
    for xi, v in zip(x, means.values):                     # mean inside the bar, clear of the seed dots
        ax.text(xi, 0.03, f"{v:.3f}", ha="center", va="bottom", fontsize=8.5, color="white")
    for i, c in enumerate(CARS):
        pts = res[res.test_car == c]["macro_f1"].values
        ax.scatter(np.full(len(pts), x[i]), pts, s=14, color=INK, zorder=3, linewidths=0)
ax.axhline(CHANCE, ls="--", lw=1, color=INK2, label="chance (0.33)")
ax.set_xticks(range(3)); ax.set_xticklabels([CAR_LABEL[c] for c in CARS])
ax.set_ylabel("macro F1 on the unseen car"); ax.set_ylim(0, 1); ax.grid(axis="x", visible=False)
ax.set_title("Offline, leave-one-car-out: bars = mean of 3 seeds, dots = single seeds", loc="left")
ax.legend(loc="upper left", bbox_to_anchor=(0, -0.1), ncol=3)
save(fig, "fig5_offline_comparison.png")

# ======================= fig6: int8 =======================
fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 3.8), gridspec_kw={"width_ratios": [1.4, 1]})
w = 0.36
for j, (m, color, lbl) in enumerate([("pytorch_fp32", "#b8b3ef", "FP32 (PyTorch)"),
                                     ("onnx_int8", MODEL_COLOR["CNN"], "int8 (ONNX Runtime)")]):
    v = int8[int8.model == m].set_index("test_car")["macro_f1"].reindex(CARS)
    bars = a1.bar(np.arange(3) + (j - 0.5) * (w + 0.03), v.values, width=w, color=color, label=lbl)
    a1.bar_label(bars, labels=[f"{x:.3f}" for x in v.values], fontsize=8, color=INK2, padding=2)
a1.set_xticks(range(3)); a1.set_xticklabels([CAR_LABEL[c] for c in CARS], fontsize=9)
a1.set_ylim(0, 1); a1.set_ylabel("macro F1 (seed 0)"); a1.grid(axis="x", visible=False)
a1.set_title("Accuracy: int8 matches FP32", loc="left")
a1.legend(loc="upper left", bbox_to_anchor=(0, -0.1), ncol=2)
sizes = int8.groupby("model")["size_kb"].mean().reindex(["pytorch_fp32", "onnx_fp32", "onnx_int8"])
bars = a2.bar(["PyTorch\nFP32", "ONNX\nFP32", "ONNX\nint8"], sizes.values,
              color=["#b8b3ef", "#b8b3ef", MODEL_COLOR["CNN"]], width=0.55)
a2.bar_label(bars, labels=[f"{v:.0f} KB" for v in sizes.values], fontsize=9, color=INK2, padding=2)
a2.set_ylabel("file size (KB)"); a2.grid(axis="x", visible=False)
a2.set_ylim(0, sizes.max() * 1.18)
a2.set_title("Size: 2.8x smaller file", loc="left")
fig.suptitle("CNN quantization (static int8, calibrated on training cars only)", x=0.01, ha="left", fontsize=12)
fig.tight_layout()
save(fig, "fig6_int8.png")

# ======================= fig7: online =======================
cnn_o = (cnn_on[cnn_on.smoothing == "none"].groupby(["test_car", "scale"])["macro_f1"].mean().unstack())
rf_o = (rf_on[rf_on.smoothing == "none"].groupby(["test_car", "scale"])["macro_f1"].mean().unstack())
fig, ax = plt.subplots(figsize=(9, 4.2))
w = 0.19
spec = [("CNN", cnn_o, "offline", None), ("CNN", cnn_o, "running", "////"),
        ("RF", rf_o, "offline", None), ("RF", rf_o, "running", "////")]
for j, (m, tab, scale, hatch) in enumerate(spec):
    v = tab[scale].reindex(CARS).values
    x = np.arange(3) + (j - 1.5) * (w + 0.02)
    color = MODEL_COLOR[m]
    bars = ax.bar(x, v, width=w, color=color if hatch is None else SURF, edgecolor=color,
                  hatch=hatch, lw=1.2,
                  label=f"{'1D CNN int8' if m == 'CNN' else 'Random forest'}, "
                        f"{'whole-drive scale (offline)' if scale == 'offline' else 'running scale (on-device)'}"
                        f"  mean {np.mean(v):.3f}")
    ax.bar_label(bars, labels=[f"{x:.2f}" for x in v], fontsize=7.5, color=INK2, padding=2)
ax.axhline(CHANCE, ls="--", lw=1, color=INK2, label="chance (0.33)")
ax.set_xticks(range(3)); ax.set_xticklabels([CAR_LABEL[c] for c in CARS])
ax.set_ylim(0, 1); ax.set_ylabel("macro F1 (seed 0)"); ax.grid(axis="x", visible=False)
ax.set_title("Online replay: 10-sample chunks (0.1 s), IIR + sosfilt, one prediction per second", loc="left")
ax.legend(loc="upper left", bbox_to_anchor=(0, -0.1), ncol=2, fontsize=8.5)
save(fig, "fig7_online.png")

# ======================= fig8: timeline =======================
cnn_tl = pd.read_csv(os.path.join(DEPLOY, "online_replay_timeline.csv"))
rf_tl = pd.read_csv(os.path.join(RF, "online", "rf_online_timeline.csv"))
cnn_tl = cnn_tl[cnn_tl.recording == TIMELINE_PVS]
rf_tl = rf_tl[(rf_tl.recording == TIMELINE_PVS) & (rf_tl.scale == "running")]
t_max = int(max(cnn_tl.t_s.max(), rf_tl.t_s.max()))


def strip(df, col, truth=False):
    row = np.full(t_max + 1, -1)
    ok = df["moving"] & (df["label"] >= 0) if truth else np.ones(len(df), bool)
    row[df.loc[ok, "t_s"].astype(int).values] = df.loc[ok, col].values
    return row


rows = [strip(rf_tl, "label", truth=True), strip(cnn_tl, "pred"), strip(rf_tl, "pred")]
cmap = ListedColormap(["#ecebe7"] + CLASS_COLOR)
fig, ax = plt.subplots(figsize=(12, 2.6))
ax.imshow(np.array(rows) + 1, aspect="auto", cmap=cmap, vmin=0, vmax=3, interpolation="nearest",
          extent=(0, t_max / 60, 2.5, -0.5))
for yb in (0.5, 1.5):
    ax.axhline(yb, color=SURF, lw=4)
ax.set_yticks([0, 1, 2]); ax.set_yticklabels(["true surface", "1D CNN int8", "random forest"])
ax.set_xlabel("time (min)"); ax.grid(False)
car_of = (TIMELINE_PVS - 1) // 3 + 1
ax.set_title(f"PVS {TIMELINE_PVS} ({CAR_LABEL[car_of]}, never seen in training), online replay with running scale",
             loc="left")
ax.legend([Patch(color=c) for c in CLASS_COLOR] + [Patch(color="#ecebe7")],
          NAMES + ["not scored / no prediction yet"], ncol=4, loc="upper left", bbox_to_anchor=(0, -0.3))
save(fig, "fig8_timeline.png")

# ======================= fig9: RF importance =======================
imp = pd.read_csv(os.path.join(RF, "feature_importance.csv"), index_col=0).iloc[:, 0]
top = imp.sort_values(ascending=False).head(15)[::-1]
fig, ax = plt.subplots(figsize=(7, 5))
ax.barh(top.index, top.values, color=MODEL_COLOR["RF"], height=0.65)
ax.set_xlabel("mean importance (3 rounds)"); ax.grid(axis="y", visible=False)
ax.set_title("Random forest: top 15 features", loc="left")
save(fig, "fig9_rf_importance.png")

# ======================= fig10: CNN loss curve (copy) =======================
loss_png = os.path.join(CNN_RUN, "loss_seed0_test3_train1_val2.png")
if os.path.exists(loss_png):
    shutil.copy(loss_png, os.path.join(FIG, "fig10_cnn_loss_curve.png"))
    print("saved", os.path.join(FIG, "fig10_cnn_loss_curve.png"))
    if SHOW:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.imshow(plt.imread(loss_png)); ax.axis("off")
        plt.show()
        plt.close(fig)

# ======================= tables =======================
def per_car(res):
    s = res.groupby("test_car")["macro_f1"].mean().reindex(CARS)
    p = res.groupby("seed")["macro_f1"].mean()
    return [f"{v:.3f}" for v in s.values] + [f"{p.mean():.3f} ± {p.std(ddof=0):.3f}"]


lines = ["## Offline, leave-one-car-out (macro F1, mean of 3 seeds)", "",
         "| Model | Car 1 | Car 2 | Car 3 | Overall |", "|---|---|---|---|---|",
         "| 1D CNN | " + " | ".join(per_car(cnn_res)) + " |",
         "| Random forest | " + " | ".join(per_car(rf_res)) + " |",
         "| Random forest + smoothing (3 windows) | " + " | ".join(per_car(rf_smooth)) + " |", ""]

lines += ["## Online replay (macro F1, seed 0)", "",
          "| Model | Whole-drive scale (offline) | Running scale (on-device) |", "|---|---|---|",
          f"| 1D CNN int8 | {cnn_o['offline'].mean():.3f} | {cnn_o['running'].mean():.3f} |",
          f"| Random forest | {rf_o['offline'].mean():.3f} | {rf_o['running'].mean():.3f} |", ""]

i8 = int8.groupby("model")[["macro_f1", "size_kb", "latency_ms_mean"]].mean()
lines += ["## CNN quantization (seed 0, mean of 3 test cars)", "",
          "| Model | Macro F1 | File size (KB) | Latency per window (ms) |", "|---|---|---|---|"]
for m, lbl in [("pytorch_fp32", "PyTorch FP32"), ("onnx_fp32", "ONNX FP32"), ("onnx_int8", "ONNX int8")]:
    lat = "-" if np.isnan(i8.loc[m, "latency_ms_mean"]) else f"{i8.loc[m, 'latency_ms_mean']:.3f}"
    lines.append(f"| {lbl} | {i8.loc[m, 'macro_f1']:.3f} | {i8.loc[m, 'size_kb']:.0f} | {lat} |")

s0 = rf_size[rf_size.seed == 0]
lines += ["", "## Size and compute per prediction", "",
          "| Model | Memory | Work per window |", "|---|---|---|",
          "| 1D CNN int8 | ~31 KB weights (32,195 parameters) | ~2.0 million multiply-adds |",
          f"| Random forest | ~{s0.est_chip_kb.min():.0f}-{s0.est_chip_kb.max():.0f} KB (estimate, "
          f"{int(s0.nodes.min()):,}-{int(s0.nodes.max()):,} nodes) | "
          f"{int(s0.comparisons_worst_case.min())}-{int(s0.comparisons_worst_case.max())} comparisons "
          f"+ feature extraction |"]
md = "\n".join(lines)
with open(os.path.join(TAB, "summary_tables.md"), "w", encoding="utf-8") as f:
    f.write(md)
print("\n" + md)
print(f"\nfigures in {FIG}\ntables in {TAB}")
