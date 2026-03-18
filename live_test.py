import numpy as np
import tensorflow as tf
from tensorflow import keras
from brainflow.board_shim import BoardShim, BrainFlowInputParams, BoardIds
import time
from scipy.signal import butter, sosfiltfilt

# ─────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────
WINDOW_SIZE            = 128     # 0.5 sec at 256Hz — must match training
FS                     = 256     # Muse S sample rate
THRESHOLD              = 0.5     # above = CLENCH, below = REST
MODEL_PATH             = "eeg_model2.keras"
INFERENCE_INTERVAL     = 0.25    # seconds between predictions

# ── Artifact rejection — must match mainModel.py + DataCollectScript.py ──
AMPLITUDE_THRESHOLD_UV = 9999999  # effectively disabled — Muse clips at 1000 always
VARIANCE_THRESHOLD     = 200000.0 # sitting still ~55k-120k, motion will spike above this

# ── Bandpass filter settings ──────────────────
BANDPASS_LOW_HZ        = 1.0
BANDPASS_HIGH_HZ       = 40.0
FILTER_ORDER           = 4


# ─────────────────────────────────────────────
# BANDPASS FILTER
# Handles DC offset removal (replaces detrend)
# ─────────────────────────────────────────────
def bandpass_filter(window):
    nyq = FS / 2.0
    sos = butter(FILTER_ORDER,
                 [BANDPASS_LOW_HZ / nyq, BANDPASS_HIGH_HZ / nyq],
                 btype='band', output='sos')
    filtered = np.zeros_like(window)
    for ch in range(window.shape[1]):
        filtered[:, ch] = sosfiltfilt(sos, window[:, ch])
    return filtered


# ─────────────────────────────────────────────
# ARTIFACT REJECTION
# ─────────────────────────────────────────────
def is_artifact(eeg_window):
    ch_vars = np.var(eeg_window, axis=0)
    # AF7 spiking = up-down head motion
    if ch_vars[1] > 60000:
        return True, "motion (AF7)"
    # TP9+TP10 both high with AF7 elevated = rotational motion
    if ch_vars[0] > 50000 and ch_vars[3] > 50000 and ch_vars[1] > 20000:
        return True, "motion (TP9+TP10)"
    return False, ""

# ─────────────────────────────────────────────
# FEATURE EXTRACTION
# ─────────────────────────────────────────────
def extract_band_powers(window):
    features = []
    freqs = np.fft.rfftfreq(window.shape[0], d=1.0 / FS)
    bands = {
        'delta': (0.5,  4),
        'theta': (4,    8),
        'alpha': (8,   13),
        'beta':  (13,  30),
    }
    for ch in range(window.shape[1]):
        fft_vals = np.abs(np.fft.rfft(window[:, ch])) ** 2
        for _, (lo, hi) in bands.items():
            idx   = np.where((freqs >= lo) & (freqs < hi))
            power = np.mean(fft_vals[idx]) if len(idx[0]) > 0 else 0.0
            features.append(np.log1p(power))
    return np.array(features, dtype=np.float32)


# ─────────────────────────────────────────────
# NORMALIZATION
# ─────────────────────────────────────────────
scaler_mean  = np.load('scaler_mean2.npy')
scaler_scale = np.load('scaler_scale2.npy')

def normalize(features):
    return (features - scaler_mean) / scaler_scale


# ─────────────────────────────────────────────
# LOAD MODEL
# ─────────────────────────────────────────────
print("Loading model...")
model = keras.models.load_model(MODEL_PATH)
print("Model loaded.\n")


# ─────────────────────────────────────────────
# CONNECT TO MUSE S
# ─────────────────────────────────────────────
params       = BrainFlowInputParams()
board        = BoardShim(BoardIds.MUSE_S_BOARD, params)
eeg_channels = BoardShim.get_eeg_channels(BoardIds.MUSE_S_BOARD)
print(f"EEG channels: {eeg_channels}")

print("\nConnecting to Muse S... make sure it's powered on and not connected elsewhere.")

try:
    board.prepare_session()
    board.start_stream()
    print("Connected! Streaming started.")
    print("Put the headset on and try clenching your jaw.\n")
    print("─" * 45)
    print("NOTE: Windows contaminated by head movement")
    print("      are skipped — no false triggers.")
    print("─" * 45 + "\n")

    time.sleep(1)   # let buffer fill

    n_artifact = 0
    n_total    = 0

    while True:
        data = board.get_current_board_data(WINDOW_SIZE)

        if data.shape[1] < WINDOW_SIZE:
            time.sleep(0.05)
            continue

        # ── Raw EEG shape: (WINDOW_SIZE, N_CHANNELS) ─────────────────────
        window = data[eeg_channels, :WINDOW_SIZE].astype(np.float32).T

        # TEMPORARY DEBUG — per channel
        ch_vars = np.var(window, axis=0)
        ch_names = ['TP9', 'AF7', 'AF8', 'TP10']
        var_str = '  '.join([f"{n}={v:.0f}" for n, v in zip(ch_names, ch_vars)])
        print(f"  {var_str}")

        # ── Artifact check on raw signal ──────────────────────────────────
        bad, reason = is_artifact(window)
        if bad:
            n_artifact += 1
            print(f"  [SKIP — {reason}]  "
                  f"(artifacts: {n_artifact}/{n_total})")
            time.sleep(INFERENCE_INTERVAL)
            continue

        # ── Bandpass filter (removes DC offset + motion noise) ────────────
        window_filtered = bandpass_filter(window)

        # ── Feature extraction → normalize → predict ─────────────────────
        features   = extract_band_powers(window_filtered)
        features   = normalize(features)
        inp        = features.reshape(1, -1)
        prediction = model.predict(inp, verbose=0)[0][0]

        # ── Output ────────────────────────────────────────────────────────
        if prediction > THRESHOLD:
            print(f"  CLENCH  ████████  ({prediction:.2f})")
        else:
            print(f"  rest    ░░░░░░░░  ({prediction:.2f})")

        time.sleep(INFERENCE_INTERVAL)

except KeyboardInterrupt:
    print("\nStopped by user.")

finally:
    print("Closing BrainFlow session...")
    board.stop_stream()
    board.release_session()
    print("Done.")