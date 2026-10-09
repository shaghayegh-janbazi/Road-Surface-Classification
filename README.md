Road Surface Classification from Vehicle Vibration

This project classifies the road surface (dirt, cobblestone or asphalt) every second, using an accelerometer and gyroscope mounted below a car's suspension.

It is designed for a sensor chip, not only for a laptop:

Every model is tested on a car it has never seen
Filtering uses an IIR Butterworth high-pass with sosfilt, which works online, so training and real time see the same signal
The CNN is compressed to int8, and the data is replayed every 0.1 s, the way a chip would receive it

Main results (macro F1 on an unseen car):

	1D CNN	Random forest
Offline (3 seeds)	0.757 ± 0.014	0.775 ± 0.002
Online replay, on-device normalization	0.633	0.696
Work per prediction	~2.0 M multiply-adds	150–350 comparisons + features
Memory	~31 KB (int8 weights)	~22–117 KB (estimate)

In short: a small random forest on signal features is as accurate as the CNN, more robust in real time, and much cheaper to run.

1. Data

Dataset: PVS – Passive Vehicular Sensors dataset:

3 cars, each with its own driver, each driving the same 3 routes → 9 recordings (PVS 1–9), about 3 hours
MPU-9250 IMU below the suspension, sampled at 100 Hz
A surface label for every sample

The data is not included in this repository (license CC BY-NC-ND 4.0). To run the code, download it and put the folders PVS 1 … PVS 9 in data/.

<img width="1265" height="764" alt="fig1_dataset" src="https://github.com/user-attachments/assets/6dcf11eb-43cd-490b-92eb-0a01010bdccc" />

Windows per car and surface after preprocessing. The three classes are fairly balanced in every car.

2. Preprocessing
Merge the sensor file and the label file by row position. The label file has no timestamp, so both files are first checked to have the same number of rows.
Filter each recording with a Butterworth high-pass, 0.5 Hz, order 2, using sosfilt. This removes gravity and gyroscope bias. IIR with sosfilt is used instead of sosfiltfilt because the system must work online: sosfiltfilt needs the whole recording, sosfilt only needs the samples that have already arrived. Training and real time then see exactly the same signal.
Cut 2-s windows (200 samples) with a 1-s step, so the system gives one answer per second.
Scale each channel by the recording's typical vibration level (median window RMS), so a car that shakes harder overall does not look like a rougher road.
Keep only windows where the car is moving (≥ 1 m/s) and the surface does not change inside the window. Result: 10,048 windows.
3. Choosing the inputs

A first model using the acceleration magnitude √(x² + y² + z²) reached only 0.56 macro F1. The average spectrum of every channel, for every surface and every car, shows why:

Cobblestone has extra energy at 40–45 Hz in acc_y and gyro_z, and this is true in all three cars
Car 3's suspension resonance is around 17 Hz, while Cars 1 and 2 are around 12 Hz
gyro_x and gyro_y look different in each car, so they do not transfer to a new car
The magnitude throws away the direction of the vibration, and that is where dirt and cobblestone differ

The final inputs are the channels whose patterns are the same across cars: acc_x, acc_y, acc_z, gyro_z. This raised macro F1 from 0.56 to 0.76.

<img width="2188" height="1926" alt="fig2_spectra" src="https://github.com/user-attachments/assets/52ed3b4b-e4a5-4b7d-bc9b-df94bcffa380" />

Average spectrum of each channel (rows) for each surface (columns); each line is one car. Cobblestone is higher than dirt at 40–45 Hz (shaded) in acc_y and gyro_z for all three cars. Car 3 (green) peaks near 17 Hz instead of ~12 Hz.

4. Evaluation
Leave-one-car-out: train on 2 cars, test on the third, and repeat so each car is the test car once
Every choice (number of epochs, model size, features) is made by validating one training car against the other; the test car is used only once, at the end
Each round is repeated with 3 seeds to show how stable the result is
Main metric: macro F1, so all three surfaces count equally
5. Model 1: 1D CNN
Layer	Output shape
Input: 4 channels × 200 samples	4 × 200
Conv 32, kernel 5 + BatchNorm + ReLU + MaxPool 2	32 × 100
Conv 64, kernel 3 + BatchNorm + ReLU + MaxPool 2	64 × 50
Conv 128, kernel 3 + BatchNorm + ReLU + MaxPool 2	128 × 25
Global average pooling → Linear	3 classes

32,195 parameters, dropout 0.2, Adam (learning rate 0.001), early stopping on validation loss. Kept small on purpose to fit on a chip.

<!-- fig3_cnn_confusion: delete this line and drag fig3_cnn_confusion.png here --> <img width="600" height="400" alt="fig10_cnn_loss_curve" src="https://github.com/user-attachments/assets/4b493577-6584-4651-92ed-4b7afbcbd833" />

Example validation run (train Car 1, validate Car 2): validation loss is lowest at epoch 2 and then rises while training loss keeps falling, so early stopping matters when the validation car is different from the training car.

Result: asphalt is almost always correct. Dirt and cobblestone are the hard pair. On Car 3, 85% of dirt is predicted as cobblestone, because that car passes more high-frequency vibration on every surface, so its dirt looks like the other cars' cobblestone.

6. Model 2: Random forest
48 features per window: RMS, peak-to-peak, kurtosis, crest factor and 8 frequency-band energies (Welch), for each of the 4 channels
Model type, size and number of features are chosen on the training cars; the smallest model within 0.01 F1 of the best is kept, because memory matters on a chip
Optional smoothing: average the probabilities of the last 3 predictions (only uses predictions already made, so it works online)
<img width="2381" height="801" alt="fig4_rf_confusion" src="https://github.com/user-attachments/assets/a848c0b3-6ad1-4de9-882e-b45aed096ff8" />

The forest finds more of Car 3's dirt (31% vs 14% for the CNN), but on Cars 1 and 2 it confuses more cobblestone with dirt.

<img width="1448" height="936" alt="fig9_rf_importance" src="https://github.com/user-attachments/assets/9545dfb3-c666-49ac-9064-4c9c7aa39ff0" />

The most used features are vibration strength (gyro_z RMS, peak-to-peak) and the 16–24 Hz and 12–16 Hz bands, around the suspension resonance.

7. Deployment

int8 quantization of the CNN (ONNX Runtime, calibrated only on training cars):

Model (seed 0, mean of 3 test cars)	Macro F1	File size	Latency per window*
PyTorch FP32	0.757	135 KB	–
ONNX FP32	0.757	127 KB	0.146 ms
ONNX int8	0.759	46 KB	0.083 ms

*Measured on a laptop CPU (batch 1, one thread), not on a microcontroller.

int8 makes the model 2.8× smaller with no loss in accuracy.

<img width="2188" height="743" alt="fig6_int8" src="https://github.com/user-attachments/assets/e14f2e9f-3f38-4769-938a-5c7a8e100598" />

Left: macro F1 per test car before and after int8 (seed 0). Right: model file size.

Online replay: the raw data is fed to the system 10 samples at a time (every 0.1 s). The filter keeps its memory between chunks, and its output was checked to be identical to the offline filter. The model predicts once per second from the last 2 s.

What the replay revealed: in preprocessing, step 4 uses the vibration level of the whole drive, which a device cannot know in advance. On a device it has to be estimated from the drive so far, and this lowers the score for both models. This is the most important lesson of the deployment step: training must use exactly what the chip will have.

<img width="1628" height="881" alt="fig7_online" src="https://github.com/user-attachments/assets/6675266c-7b03-4f92-b22b-0c512fad1f81" />

Solid bars: scale from the whole drive. Hatched bars: scale estimated from the drive so far, as on a device.

Model (seed 0, mean of 3 test cars)	Whole-drive scale (offline)	Running scale (on-device)	Drop
1D CNN int8	0.740	0.633	−0.11
Random forest	0.775	0.696	−0.08
<img width="2115" height="664" alt="fig8_timeline" src="https://github.com/user-attachments/assets/be68698a-6f5f-436a-8259-317c87fcbc1a" />

One drive of Car 3, second by second. The drive starts on asphalt, but both models call it cobblestone: the running scale has only seen this first stretch, so the vibration looks rough. After the long rough section, the final asphalt is recognized correctly. The forest finds part of the dirt; the CNN calls almost all of it cobblestone.

8. CNN vs random forest
<img width="1305" height="821" alt="fig5_offline_comparison" src="https://github.com/user-attachments/assets/bb6d259f-d7d3-4c31-bd0a-9ee2fc4c81d6" />

Bars: mean of 3 seeds. Dots: each seed. The forest is more stable across seeds and better on Car 3; the CNN is better on Car 1.

Offline, leave-one-car-out (macro F1, mean of 3 seeds):

Model	Car 1	Car 2	Car 3	Overall
1D CNN	0.837	0.800	0.635	0.757 ± 0.014
Random forest	0.793	0.818	0.715	0.775 ± 0.002
Random forest + smoothing (3 windows)	0.803	0.841	0.714	0.786 ± 0.005

Online replay (macro F1, seed 0): CNN 0.633, random forest 0.696 with the running scale (table in section 7).

Size and compute per prediction:

Model	Memory	Work per window
1D CNN int8	~31 KB weights (32,195 parameters)	~2.0 million multiply-adds
Random forest	~22–117 KB (estimate, 2,281–12,023 nodes)	150–350 comparisons + feature extraction

Conclusion: the two models are equal offline. The random forest is more robust in real time and needs far less computation, so it is the better choice for a sensor chip. The CNN uses less memory in two of the three rounds.

9. Limitations
Only 3 cars, and each car had one driver, so car and driver effects cannot be separated
Asphalt was driven about 3× faster than dirt and cobblestone, so surface and speed are partly mixed
The channels were chosen using spectra from all three cars, so that choice is not completely blind to the test car
On-device normalization costs 0.08–0.11 macro F1; the next step is to train with the same running estimate the device uses
Online results use one seed; the CNN's online score is averaged per recording, the forest's combines the 3 recordings of each car
A published leave-one-vehicle-out study on PVS reports 87.2 macro F1 using two sensor positions and 6-s windows (source); this project uses one position and 2-s windows
10. What I learned
Checking the spectra of each channel across cars improved the result more than any change to the model (0.56 → 0.76)
Testing on an unseen car is much harder than mixing cars; it shows how a sensor would perform in a new vehicle
The online replay exposed a training/deployment mismatch (normalization from the whole drive) that offline testing could not show
A small feature-based model can match a CNN at a fraction of the computation, so it is worth testing before choosing a deep model
11. How to run
bash
pip install -r requirements.txt
python src/preprocess_v2.py      # filtered windows -> windows_v2.npz
python src/explore_channels.py   # spectra and channel ranking
python src/cnn_v2.py             # CNN, leave-one-car-out, 3 seeds
python src/evaluate_runs.py      # full metrics report
python src/deploy.py             # ONNX, int8, online replay
python src/rf_model.py           # random forest
python src/rf_online.py          # random forest online replay
python src/make_figures.py       # figures and tables
Citation

J. Menegazzo and A. von Wangenheim, "Multi-Contextual and Multi-Aspect Analysis for Road Surface Type Classification Through Inertial Sensors and Deep Learning," 2020 X Brazilian Symposium on Computing Systems Engineering (SBESC), Florianópolis, 2020, pp. 1–8. doi:10.1109/SBESC51047.2020.9277846
