import numpy as np
from brainflow.board_shim import BoardShim, BrainFlowInputParams, BoardIds
import time
import keyboard  # pip install keyboard

# ─────────────────────────────────────────────
# CONFIGURATION  — must match mainModel.py
# ─────────────────────────────────────────────
WINDOW_SIZE   = 128       # samples per window (0.5s at 256Hz)
FS            = 256       # Muse S sample rate (Hz)
N_CHANNELS    = 4         # TP9, AF7, AF8, TP10
SAVE_INTERVAL = 0.5       # collect one window every 0.5 seconds

# ── Artifact rejection — same as live_test.py ─
AMPLITUDE_THRESHOLD_UV = 150.0
VARIANCE_THRESHOLD     = 2000.0

# ── Output files ──────────────────────────────
# APPEND MODE: if these files already exist, new
# data is appended — existing data is NOT lost.
OUTPUT_EEG    = "inputDataTest3.csv"
OUTPUT_LABELS = "inputDataLabelTest3.csv"

# Label definition:
# 0 = REST  (sit still OR shake head side to side — anything not a clench)
# 1 = CLENCH (hold SPACE while jaw clenching)


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
# CONNECT TO MUSE S
# ─────────────────────────────────────────────
params       = BrainFlowInputParams()
# params.mac_address = "XX:XX:XX:XX:XX:XX"  # uncomment if needed

board        = BoardShim(BoardIds.MUSE_S_BOARD, params)
eeg_channels = BoardShim.get_eeg_channels(BoardIds.MUSE_S_BOARD)

print(f"EEG channels: {eeg_channels}")
print("\nConnecting to Muse S...")
board.prepare_session()
board.start_stream()
print("Connected!\n")
time.sleep(2)

print("─" * 55)
print("HOW TO COLLECT DATA:")
print("  Sit still OR shake head side-to-side → label 0 (REST)")
print("  Hold SPACE + jaw clench              → label 1 (CLENCH)")
print("  Press Q                              → stop and save")
print("─" * 55)

# ── Check for existing data and warn user ─────
import os
existing_samples = 0
if os.path.exists(OUTPUT_EEG):
    try:
        existing = np.loadtxt(OUTPUT_EEG, delimiter=',')
        existing_samples = existing.shape[0] if existing.ndim == 2 else 1
        print(f"\n  APPEND MODE: found {existing_samples} existing samples in {OUTPUT_EEG}")
        print(f"  New data will be added to the existing dataset.")
    except Exception:
        print(f"\n  NOTE: {OUTPUT_EEG} exists but could not be read — will overwrite.")
else:
    print(f"\n  NEW SESSION: {OUTPUT_EEG} not found — starting fresh.")

print("\nStarting in 3 seconds — put the headset on...\n")
time.sleep(3)


# ─────────────────────────────────────────────
# COLLECTION LOOP
# ─────────────────────────────────────────────
all_eeg    = []
all_labels = []
n_rejected = 0

try:
    while True:
        if keyboard.is_pressed('q'):
            print("\nQ pressed — stopping collection.")
            break

        data = board.get_current_board_data(WINDOW_SIZE)

        if data.shape[1] < WINDOW_SIZE:
            time.sleep(0.05)
            continue

        # ── Raw EEG shape: (WINDOW_SIZE, N_CHANNELS) ─────────────────────
        window = data[eeg_channels, :WINDOW_SIZE].astype(np.float32).T

        # ── Artifact check ────────────────────────────────────────────────
        bad, reason = is_artifact(window)
        if bad:
            n_rejected += 1
            print(f"  [REJECTED — {reason}]  total rejected: {n_rejected}")
            time.sleep(SAVE_INTERVAL)
            continue

        # ── Label: 1 if holding SPACE, else 0 ────────────────────────────
        label = 1 if keyboard.is_pressed('space') else 0

        all_eeg.append(window.flatten())
        all_labels.append(label)

        n_clench   = sum(all_labels)
        n_rest     = len(all_labels) - n_clench
        state      = "CLENCH ████" if label == 1 else "rest   ░░░░"
        total_saved = existing_samples + len(all_labels)
        print(f"  {state}  |  new: {len(all_labels)}"
              f"  (rest={n_rest}, clench={n_clench})"
              f"  total in file: {total_saved}"
              f"  rejected: {n_rejected}")

        time.sleep(SAVE_INTERVAL)

except KeyboardInterrupt:
    print("\nCtrl+C detected — stopping.")

finally:
    board.stop_stream()
    board.release_session()

    if len(all_eeg) == 0:
        print("No data collected — nothing saved.")
    else:
        X_new = np.array(all_eeg,    dtype=np.float32)
        Y_new = np.array(all_labels, dtype=np.float32)

        # ── Append to existing CSVs if they exist ─────────────────────────
        if existing_samples > 0:
            try:
                X_old = np.loadtxt(OUTPUT_EEG,    delimiter=',').astype(np.float32)
                Y_old = np.loadtxt(OUTPUT_LABELS, delimiter=',').astype(np.float32)

                if X_old.ndim == 1:
                    X_old = X_old.reshape(1, -1)

                X_save = np.vstack([X_old, X_new])
                Y_save = np.concatenate([Y_old, Y_new])
                print(f"\n  Appended {len(X_new)} new samples to {existing_samples} existing.")
            except Exception as e:
                print(f"\n  WARNING: Could not load existing data ({e}) — saving new data only.")
                X_save = X_new
                Y_save = Y_new
        else:
            X_save = X_new
            Y_save = Y_new

        np.savetxt(OUTPUT_EEG,    X_save, delimiter=',')
        np.savetxt(OUTPUT_LABELS, Y_save, delimiter=',')

        n_clench_total = int(Y_save.sum())
        n_rest_total   = len(Y_save) - n_clench_total
        n_new_clench   = int(Y_new.sum())
        n_new_rest     = len(Y_new) - n_new_clench

        print(f"\n  This session:  {len(Y_new)} samples  "
              f"(rest={n_new_rest}, clench={n_new_clench})")
        print(f"  Total in file: {len(Y_save)} samples  "
              f"(rest={n_rest_total}, clench={n_clench_total})")
        print(f"  Rejected this session: {n_rejected}")

        if n_rest_total < 50 or n_clench_total < 50:
            print("\n  WARNING: Aim for at least 50 of each class.")
        elif abs(n_rest_total - n_clench_total) > 0.3 * len(Y_save):
            short = 'CLENCH' if n_clench_total < n_rest_total else 'REST'
            print(f"\n  WARNING: Classes are imbalanced — collect more {short} samples.")
        else:
            print("\n  Good dataset!")
            print("  Set FEATURES_PRECOMPUTED = False in mainModel.py and retrain.")