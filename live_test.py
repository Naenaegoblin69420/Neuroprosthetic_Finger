import numpy as np
import tensorflow as tf
from tensorflow import keras
from brainflow.board_shim import BoardShim, BrainFlowInputParams, BoardIds
from brainflow.data_filter import DataFilter, FilterTypes, DetrendOperations
import time

# ─────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────
WINDOW_SIZE  = 128       # 0.5 sec of data
FS           = 256       # Muse S sample rate
THRESHOLD    = 0.5       # above this = CLENCH, below = REST
MODEL_PATH   = "eeg_model.keras"

# ─────────────────────────────────────────────
# FEATURE EXTRACTION — must match training exactly
# ─────────────────────────────────────────────
def extract_band_powers(window, fs=256):
    features = []
    freqs = np.fft.rfftfreq(window.shape[0], d=1.0/fs)
    bands = {
        'delta': (0.5, 4),
        'theta': (4,   8),
        'alpha': (8,  13),
        'beta':  (13, 30),
    }
    for ch in range(window.shape[1]):
        fft_vals = np.abs(np.fft.rfft(window[:, ch])) ** 2
        for _, (lo, hi) in bands.items():
            idx = np.where((freqs >= lo) & (freqs < hi))
            power = np.mean(fft_vals[idx]) if len(idx[0]) > 0 else 0.0
            features.append(np.log1p(power))
    return np.array(features, dtype=np.float32)

# ─────────────────────────────────────────────
# LOAD SCALER
# ─────────────────────────────────────────────
scaler_mean  = np.load('scaler_mean.npy')
scaler_scale = np.load('scaler_scale.npy')

def normalize(features):
    return (features - scaler_mean) / scaler_scale

# ─────────────────────────────────────────────
# LOAD KERAS MODEL (full TF, no TFLite needed on laptop)
# ─────────────────────────────────────────────
print("Loading model...")
model = keras.models.load_model(MODEL_PATH)
print("Model loaded.\n")

# ─────────────────────────────────────────────
# CONNECT TO MUSE S VIA BLUETOOTH
# ─────────────────────────────────────────────
BoardShim.enable_dev_board_logger()

params = BrainFlowInputParams()
# params.mac_address = "Muse-0889"  # uncomment if needed

board        = BoardShim(BoardIds.MUSE_S_BOARD, params)
eeg_channels = BoardShim.get_eeg_channels(BoardIds.MUSE_S_BOARD)
print(f"EEG channels: {eeg_channels}")

# ─────────────────────────────────────────────
# MAIN LOOP
# ─────────────────────────────────────────────
print("\nConnecting to Muse S... make sure it's powered on and not connected to anything else")

try:
    board.prepare_session()
    board.start_stream()
    print("Connected! Streaming started.")
    print("Put the headset on and try clenching your jaw.\n")
    print("─" * 40)

    # Let buffer fill for 1 full second before starting
    time.sleep(1)

    while True:
        # Get latest WINDOW_SIZE samples from buffer
        data = board.get_current_board_data(WINDOW_SIZE)

        # Skip if buffer doesn't have enough samples yet
        if data.shape[1] < WINDOW_SIZE:
            time.sleep(0.05)
            continue

        # Pull the 4 EEG channels, transpose to shape (WINDOW_SIZE, 4)
        window = data[eeg_channels, :WINDOW_SIZE].T  # shape: (128, 4)

        # Extract features → normalize → predict
        features   = extract_band_powers(window)         # shape: (16,)
        features   = normalize(features)                 # shape: (16,)
        inp        = features.reshape(1, -1)             # shape: (1, 16) for model
        prediction = model.predict(inp, verbose=0)[0][0] # single float 0.0–1.0

        # Print result
        if prediction > THRESHOLD:
            print(f"  CLENCH  ████████  ({prediction:.2f})")
        else:
            print(f"  rest    ░░░░░░░░  ({prediction:.2f})")

        # Run inference every 0.25 seconds
        time.sleep(0.25)

except KeyboardInterrupt:
    print("\nStopped by user.")

finally:
    print("Closing BrainFlow session...")
    board.stop_stream()
    board.release_session()
    print("Done.")