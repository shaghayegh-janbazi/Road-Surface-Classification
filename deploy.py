"""
Deployment step for the final CNN (run G: acc x/y/z + gyro_z, below suspension).

Part A: ONNX + int8, for each leave-one-car-out model (seed 0)
  1. export PyTorch -> ONNX (FP32), check outputs match PyTorch
  2. static int8 quantization; calibration windows from the TRAINING cars only
  3. compare PyTorch FP32 / ONNX FP32 / ONNX int8: macro F1 on the test car, file size, latency

Part B: online replay, simulating the chip
  Raw CSV samples arrive in chunks of 10 (= every 0.1 s at 100 Hz):
    IIR high-pass per channel with sosfilt, filter state zi carried between chunks
    -> 200-sample ring buffer -> every 1 s: scale, normalize, int8 ONNX model -> class
  Two scale modes:
    "offline": per-recording scale from preprocessing (whole drive known) -> must match offline
    "running": median window RMS of the drive SO FAR (what a real chip can do)
  Also: check the chunked filter equals the offline filter, latency per prediction,
        and a 3-prediction majority vote.

Needs: pip install onnx onnxruntime
"""
import os
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import onnx
import onnxruntime as ort
from onnxruntime.quantization import (quantize_static, CalibrationDataReader,
                                      QuantFormat, QuantType)
from onnxruntime.quantization.shape_inference import quant_pre_process
from scipy.signal import butter, sosfilt, sosfilt_zi
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ======================= settings =======================
BASE = "data"                     # PVS folders and all outputs live here; run scripts from the repo root
DATA_FILE = os.path.join(BASE, "windows_v2.npz")
RUN_DIR = os.path.join(BASE, "cnn_runs", "acc_x+acc_y+acc_z+gyro_z__noaug")
OUT_DIR = os.path.join(BASE, "deploy")
SEED = 0
CARS = [1, 2, 3]
NAMES = ["dirt", "cobblestone", "asphalt"]
N_CALIB = 500                  # calibration windows for int8 (training cars only)
N_LATENCY = 500                # windows timed one by one

FS, WIN, STEP, CHUNK = 100, 200, 100, 10      # chunk of 10 samples = 0.1 s
MIN_SPEED = 1.0
WARMUP_WINDOWS = 30            # running scale: wait for 30 moving windows (~30 s) before predicting
SURFACES = ["dirt_road", "cobblestone_road", "asphalt_road"]
SOS = butter(2, 0.5, btype="highpass", fs=FS, output="sos")

os.makedirs(OUT_DIR, exist_ok=True)


# ======================= model (same as cnn_v2.py) =======================
class SurfaceCNN(nn.Module):
    def __init__(self, in_channels, n_classes=3):
        super().__init__()

        def block(c_in, c_out, k):
            return nn.Sequential(
                nn.Conv1d(c_in, c_out, kernel_size=k, padding=k // 2, bias=False),
                nn.BatchNorm1d(c_out), nn.ReLU(), nn.MaxPool1d(2), nn.Dropout(0.2))

        self.features = nn.Sequential(block(in_channels, 32, 5), block(32, 64, 3), block(64, 128, 3))
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Linear(128, n_classes)

    def forward(self, x):
        return self.head(self.pool(self.features(x)).squeeze(-1))


def load_trained(test_car):
    ckpt = torch.load(os.path.join(RUN_DIR, f"cnn_seed{SEED}_test{test_car}.pt"), weights_only=True)
    model = SurfaceCNN(len(ckpt["channels"]))
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    mean = np.array(ckpt["mean"], dtype=np.float32)
    std = np.array(ckpt["std"], dtype=np.float32)
    return model, mean, std, ckpt["channels"]


def normalize(X, mean, std):
    return ((X - mean[None, :, None]) / std[None, :, None]).astype(np.float32)


def kb(path):
    return os.path.getsize(path) / 1024


def latency_ms(sess, X):
    """Mean and 95th percentile time for ONE window (batch of 1), single thread."""
    name = sess.get_inputs()[0].name
    for i in range(20):                                   # warm-up
        sess.run(None, {name: X[i:i + 1]})
    t = []
    for i in range(min(N_LATENCY, len(X))):
        t0 = time.perf_counter()
        sess.run(None, {name: X[i:i + 1]})
        t.append((time.perf_counter() - t0) * 1000)
    return float(np.mean(t)), float(np.percentile(t, 95))


def onnx_predict(sess, X):
    return sess.run(None, {sess.get_inputs()[0].name: X})[0].argmax(1)


class CalibReader(CalibrationDataReader):
    """Feeds calibration windows one at a time to the int8 quantizer."""

    def __init__(self, X, input_name):
        self.it = iter([{input_name: X[i:i + 1]} for i in range(len(X))])

    def get_next(self):
        return next(self.it, None)


def session(path):
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = 1                         # closer to a single small core
    return ort.InferenceSession(path, opts, providers=["CPUExecutionProvider"])


# ======================= Part A: ONNX + int8 =======================
def part_a():
    d = np.load(DATA_FILE)
    names_all = [str(c) for c in d["channels"]]
    rows, models = [], {}

    for test_car in CARS:
        model, mean, std, channels = load_trained(test_car)
        idx = [names_all.index(c) for c in channels]
        X = d["X"][:, idx, :].astype(np.float32)
        y, car = d["y"], d["car"]
        Xte, yte = normalize(X[car == test_car], mean, std), y[car == test_car]

        tag = f"test{test_car}"
        fp32 = os.path.join(OUT_DIR, f"cnn_{tag}_fp32.onnx")
        pre = os.path.join(OUT_DIR, f"cnn_{tag}_pre.onnx")
        int8 = os.path.join(OUT_DIR, f"cnn_{tag}_int8.onnx")

        # 1. export
        dummy = torch.zeros(1, len(channels), WIN)
        kw = dict(input_names=["x"], output_names=["scores"], opset_version=17,
                  dynamic_axes={"x": {0: "batch"}, "scores": {0: "batch"}})
        try:
            torch.onnx.export(model, dummy, fp32, dynamo=False, **kw)
        except TypeError:                                  # older PyTorch has no 'dynamo' argument
            torch.onnx.export(model, dummy, fp32, **kw)
        onnx.checker.check_model(onnx.load(fp32))

        # 2. check ONNX FP32 == PyTorch
        with torch.no_grad():
            pt_scores = model(torch.from_numpy(Xte)).numpy()
        s_fp32 = session(fp32)
        onnx_scores = s_fp32.run(None, {"x": Xte})[0]
        max_diff = float(np.abs(pt_scores - onnx_scores).max())

        # 3. int8 static quantization, calibration on training cars only
        quant_pre_process(fp32, pre)
        rng = np.random.default_rng(0)
        Xtr = normalize(X[car != test_car], mean, std)
        calib = Xtr[rng.choice(len(Xtr), size=min(N_CALIB, len(Xtr)), replace=False)]
        quantize_static(pre, int8, CalibReader(calib, "x"), quant_format=QuantFormat.QDQ,
                        per_channel=True, weight_type=QuantType.QInt8, activation_type=QuantType.QInt8)
        s_int8 = session(int8)

        # 4. compare
        pred = {"pytorch_fp32": pt_scores.argmax(1), "onnx_fp32": onnx_scores.argmax(1),
                "onnx_int8": onnx_predict(s_int8, Xte)}
        pt_path = os.path.join(RUN_DIR, f"cnn_seed{SEED}_test{test_car}.pt")
        sizes = {"pytorch_fp32": kb(pt_path), "onnx_fp32": kb(fp32), "onnx_int8": kb(int8)}
        lat = {"onnx_fp32": latency_ms(s_fp32, Xte), "onnx_int8": latency_ms(s_int8, Xte)}

        for k, p in pred.items():
            rows.append({"test_car": test_car, "model": k,
                         "macro_f1": f1_score(yte, p, average="macro"),
                         "accuracy": accuracy_score(yte, p),
                         "agree_with_fp32": float((p == pred["pytorch_fp32"]).mean()),
                         "size_kb": sizes[k],
                         "latency_ms_mean": lat.get(k, (np.nan, np.nan))[0],
                         "latency_ms_p95": lat.get(k, (np.nan, np.nan))[1]})
        print(f"test car {test_car}: ONNX FP32 vs PyTorch max score difference {max_diff:.2e} "
              f"({'OK' if max_diff < 1e-3 else 'CHECK EXPORT'})")
        models[test_car] = (int8, mean, std, channels)

    table = pd.DataFrame(rows)
    table.to_csv(os.path.join(OUT_DIR, "onnx_int8_comparison.csv"), index=False)
    print("\nPART A: PyTorch FP32 vs ONNX FP32 vs ONNX int8 (test car, seed 0)")
    print(table.round(3).to_string(index=False))
    avg = table.groupby("model")[["macro_f1", "size_kb", "latency_ms_mean"]].mean().round(3)
    print("\naverage over the 3 test cars:\n", avg.to_string())
    n_params = sum(p.numel() for p in SurfaceCNN(4).parameters())
    print(f"\nparameters: {n_params} -> ~{n_params * 4 / 1024:.0f} KB as float32, "
          f"~{n_params / 1024:.0f} KB as int8 (weights only; files add graph overhead)")
    return models


# ======================= Part B: online replay =======================
def folder_for(n):
    for name in os.listdir(BASE):
        if name.replace(" ", "") == f"PVS{n}":
            return os.path.join(BASE, name)
    raise FileNotFoundError(f"PVS {n} folder not found")


def load_raw(n, channels):
    path = folder_for(n)
    data = pd.read_csv(os.path.join(path, "dataset_gps_mpu_left.csv"), usecols=channels + ["speed"])
    labels = pd.read_csv(os.path.join(path, "dataset_labels.csv"), usecols=SURFACES)
    onehot = labels[SURFACES].to_numpy()
    surface = np.where(onehot.sum(axis=1) == 1, onehot.argmax(axis=1), -1)
    return data[channels].to_numpy(np.float64).T, data["speed"].to_numpy(np.float64), surface


def replay(n, int8_path, mean, std, channels, scale_offline, mode):
    """Stream one recording in chunks of 10 samples, predict every 100 samples."""
    raw, speed, surface = load_raw(n, channels)
    C, N = raw.shape
    sess = session(int8_path)

    # offline reference filter (only for the equivalence check)
    ref = np.stack([sosfilt(SOS, raw[c], zi=sosfilt_zi(SOS) * raw[c, 0])[0] for c in range(C)])

    zi = [sosfilt_zi(SOS) * raw[c, 0] for c in range(C)]     # filter state, one per channel
    buf = np.zeros((C, WIN))                                 # ring buffer (newest at the end)
    seen, rms_history, out, max_filter_diff = 0, [], [], 0.0

    for start in range(0, N - CHUNK + 1, CHUNK):            # one loop = 0.1 s of data arriving
        chunk = raw[:, start:start + CHUNK]
        filt = np.empty_like(chunk)
        for c in range(C):
            filt[c], zi[c] = sosfilt(SOS, chunk[c], zi=zi[c])
        buf = np.concatenate([buf[:, CHUNK:], filt], axis=1)
        seen += CHUNK

        if seen < WIN or (seen - WIN) % STEP != 0:           # predict once per second
            continue
        w0 = seen - WIN
        max_filter_diff = max(max_filter_diff, float(np.abs(buf - ref[:, w0:seen]).max()))

        with np.errstate(all="ignore"):
            spd = np.nanmean(speed[w0:seen])
        moving = spd >= MIN_SPEED
        lab = surface[w0:seen]
        label = int(lab[0]) if (lab[0] != -1 and np.all(lab == lab[0])) else -1

        if mode == "offline":
            scale = scale_offline
        else:                                                # running median of moving windows so far
            if moving:
                rms_history.append(np.sqrt((buf ** 2).mean(axis=1)))
            if len(rms_history) < WARMUP_WINDOWS:
                continue
            scale = np.median(rms_history, axis=0)

        t0 = time.perf_counter()
        x = normalize((buf / scale[:, None])[None].astype(np.float32), mean, std)
        pred = int(sess.run(None, {"x": x})[0].argmax(1)[0])
        ms = (time.perf_counter() - t0) * 1000
        out.append({"recording": n, "t_s": seen / FS, "moving": moving, "label": label,
                    "pred": pred, "latency_ms": ms})

    df = pd.DataFrame(out)
    # majority vote over the last 3 predictions (only past and present, works online)
    df["pred_vote3"] = (df["pred"].rolling(3, min_periods=1)
                        .apply(lambda v: np.bincount(v.astype(int), minlength=3).argmax(), raw=True)
                        .astype(int))
    return df, max_filter_diff


def part_b(models):
    d = np.load(DATA_FILE)
    scales = d["scales"]
    scale_recs = list(d["scale_recordings"])
    names_all = [str(c) for c in d["channels"]]
    rows, timelines = [], []

    for test_car in CARS:
        int8_path, mean, std, channels = models[test_car]
        idx = [names_all.index(c) for c in channels]
        for n in [3 * (test_car - 1) + k for k in (1, 2, 3)]:
            s_off = scales[scale_recs.index(n)][idx]
            for mode in ["offline", "running"]:
                df, fdiff = replay(n, int8_path, mean, std, channels, s_off, mode)
                scored = df[df["moving"] & (df["label"] >= 0)]
                for col, smooth in [("pred", "none"), ("pred_vote3", "vote of 3")]:
                    rows.append({"test_car": test_car, "recording": n, "scale": mode, "smoothing": smooth,
                                 "scored_windows": len(scored),
                                 "macro_f1": f1_score(scored["label"], scored[col], average="macro"),
                                 "accuracy": accuracy_score(scored["label"], scored[col]),
                                 "latency_ms_mean": df["latency_ms"].mean(),
                                 "latency_ms_max": df["latency_ms"].max(),
                                 "filter_max_diff": fdiff})
                if mode == "running":
                    timelines.append(df)
                print(f"PVS {n} | {mode:<7} scale | {len(df)} predictions | "
                      f"chunked filter vs offline max diff {fdiff:.1e}")

    res = pd.DataFrame(rows)
    res.to_csv(os.path.join(OUT_DIR, "online_replay_results.csv"), index=False)
    pd.concat(timelines).to_csv(os.path.join(OUT_DIR, "online_replay_timeline.csv"), index=False)

    print("\nPART B: online replay (int8 model, 10-sample chunks, prediction every 1 s)")
    summary = (res.groupby(["test_car", "scale", "smoothing"])[["macro_f1", "accuracy"]]
               .mean().round(3).unstack(["scale", "smoothing"]))
    print(summary.to_string())
    print(f"\nlatency per prediction (scale + normalize + int8 inference): "
          f"mean {res['latency_ms_mean'].mean():.3f} ms, worst {res['latency_ms_max'].max():.3f} ms "
          f"(budget: 1000 ms between predictions)")
    print(f"chunked filter equals offline filter: max diff {res['filter_max_diff'].max():.1e}")

    # timeline figure: first recording of each test car, running scale
    fig, axes = plt.subplots(3, 1, figsize=(14, 8), sharey=True)
    for ax, test_car in zip(axes, CARS):
        n = 3 * (test_car - 1) + 1
        df = pd.concat(timelines)
        df = df[df["recording"] == n]
        lab = df["label"].where(df["label"] >= 0)
        ax.step(df["t_s"] / 60, lab, where="post", lw=3, alpha=0.4, label="true surface")
        ax.step(df["t_s"] / 60, df["pred_vote3"], where="post", lw=1, label="predicted (vote of 3)")
        ax.set_yticks([0, 1, 2]); ax.set_yticklabels(NAMES)
        ax.set_title(f"PVS {n} (Car {test_car}, model never saw this car), online replay")
        ax.grid(alpha=0.3)
    axes[-1].set_xlabel("time (min)"); axes[0].legend(loc="upper right")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "online_replay_timeline.png"), dpi=130)
    plt.close()
    print(f"\nfiles saved in {OUT_DIR}")


if __name__ == "__main__":
    models = part_a()
    part_b(models)