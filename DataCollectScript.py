import numpy as np
from brainflow.board_shim import BoardShim, BrainFlowInputParams, BoardIds
import time
import keyboard  # pip install keyboard

# ─────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────
WINDOW_SIZE   = 128       # must match training
FS            = 256       # Muse S sample rate
SAVE_INTERVAL = 0.5       # collect one sample every 0.5 seconds

# Label definition:
# 0 = REST  (default, no key held)
# 1 = CLENCH (hold SPACE while clenching)

OUTPUT_DATA   = "inputData.csv"
OUTPUT_LABELS = "inputDataLabel.csv"

# ─────────────────────────────────────────────
# FEATURE EXTRACTION — identical to mainModel.py
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
    return np.array(features, dtype=np.float32)  # shape: (16,)


# ─────────────────────────────────────────────
# CONNECT TO MUSE S
# ─────────────────────────────────────────────
params = BrainFlowInputParams()
# params.mac_address = "XX:XX:XX:XX:XX:XX"  # uncomment if needed

board        = BoardShim(BoardIds.MUSE_S_BOARD, params)
eeg_channels = BoardShim.get_eeg_channels(BoardIds.MUSE_S_BOARD)

print("Connecting to Muse S...")
board.prepare_session()
board.start_stream()
print("Connected!\n")
print("─" * 50)
print("HOW TO COLLECT DATA:")
print("  Just sit still        → label 0 (REST)")
print("  Hold SPACE + clench   → label 1 (CLENCH)")
print("  Press Q               → stop and save")
print("─" * 50)
print("Starting in 3 seconds — put the headset on...\n")
time.sleep(3)

# ─────────────────────────────────────────────
# COLLECTION LOOP
# ─────────────────────────────────────────────
all_features = []
all_labels   = []

try:
    while True:
        # Check for quit
        if keyboard.is_pressed('q'):
            print("\nQ pressed — stopping collection.")
            break

        # Get latest window of samples
        data = board.get_current_board_data(WINDOW_SIZE)

        if data.shape[1] < WINDOW_SIZE:
            time.sleep(0.05)
            continue

        # Extract the 4 EEG channels → shape (128, 4)
        window   = data[eeg_channels, :WINDOW_SIZE].T

        # Extract features → shape (16,)
        features = extract_band_powers(window)

        # Label: 1 if holding space, 0 otherwise
        label = 1 if keyboard.is_pressed('space') else 0

        all_features.append(features)
        all_labels.append(label)

        # Count how many of each class collected so far
        n_clench = sum(all_labels)
        n_rest   = len(all_labels) - n_clench

        state = "CLENCH ████" if label == 1 else "rest   ░░░░"
        print(f"  {state}  |  total: {len(all_labels)} samples  "
              f"(rest={n_rest}, clench={n_clench})")

        time.sleep(SAVE_INTERVAL)

except KeyboardInterrupt:
    print("\nCtrl+C detected — stopping.")

finally:
    board.stop_stream()
    board.release_session()

    if len(all_features) == 0:
        print("No data collected.")
    else:
        # Save to CSV
        X = np.array(all_features)   # shape: (N, 16)
        Y = np.array(all_labels)     # shape: (N,)

        np.savetxt(OUTPUT_DATA,   X, delimiter=',')
        np.savetxt(OUTPUT_LABELS, Y, delimiter=',')

        n_clench = int(Y.sum())
        n_rest   = len(Y) - n_clench

        print(f"\nSaved {len(X)} samples to {OUTPUT_DATA} and {OUTPUT_LABELS}")
        print(f"  REST   samples: {n_rest}")
        print(f"  CLENCH samples: {n_clench}")

        if n_rest < 50 or n_clench < 50:
            print("\n⚠️  WARNING: You need at least 50 of each class for good training.")
            print("   Run this script again to collect more data.")
        else:
            print("\n✅ Good dataset! Run mainModel.py to retrain.")