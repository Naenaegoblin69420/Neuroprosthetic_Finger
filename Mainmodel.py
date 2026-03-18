import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers
import os

# ─────────────────────────────────────────────
# 1. CONFIGURATION
# ─────────────────────────────────────────────
WINDOW_SIZE    = 128
N_CHANNELS     = 4
N_FEATURES     = 16
EPOCHS         = 150
BATCH_SIZE     = 32
LEARNING_RATE  = 1e-3
MODEL_PATH     = "eeg_model2.keras"
TFLITE_PATH    = "eeg_model2.tflite"
FS             = 256

# ── Bandpass filter settings ──────────────────
# 1–40Hz:
#   - Below 1Hz  : slow motion drift removed
#   - Above 40Hz : high-freq motion noise removed
#   - Keeps delta, theta, alpha, beta intact
BANDPASS_LOW_HZ  = 1.0
BANDPASS_HIGH_HZ = 40.0
FILTER_ORDER     = 4

# ── Artifact rejection thresholds ─────────────
# Must match DataCollectScript.py and live_test.py
# AF7-based: motion causes AF7 to spike > 60k
# TP9/TP10-based: catches rotational motion

# ── CSV filenames ─────────────────────────────
EEG_CSV   = 'inputData3.csv'
LABEL_CSV = 'inputDataLabel3.csv'

# Set to True  if CSV already has 16 band-power features (old data format)
# Set to False if CSV has raw EEG windows from new DataCollectScript
FEATURES_PRECOMPUTED = False


# ─────────────────────────────────────────────
# 2. BANDPASS FILTER
# ─────────────────────────────────────────────
def bandpass_filter(window, fs=FS, lowcut=BANDPASS_LOW_HZ,
                    highcut=BANDPASS_HIGH_HZ, order=FILTER_ORDER):
    """
    Zero-phase Butterworth bandpass filter.
    window : np.ndarray shape (WINDOW_SIZE, N_CHANNELS)
    returns: filtered window, same shape
    """
    from scipy.signal import butter, sosfiltfilt
    nyq  = fs / 2.0
    sos  = butter(order, [lowcut / nyq, highcut / nyq], btype='band', output='sos')
    filtered = np.zeros_like(window)
    for ch in range(window.shape[1]):
        filtered[:, ch] = sosfiltfilt(sos, window[:, ch])
    return filtered


# ─────────────────────────────────────────────
# 3. ARTIFACT REJECTION
# ─────────────────────────────────────────────
def is_artifact(eeg_window):
    """
    Returns True if window should be rejected.
    Uses per-channel variance — same logic as live_test.py.
    eeg_window : shape (WINDOW_SIZE, N_CHANNELS)
    """
    ch_vars = np.var(eeg_window, axis=0)
    # AF7 (index 1) spiking = up-down head motion
    if ch_vars[1] > 60000:
        return True
    # TP9 + TP10 both high with AF7 elevated = rotational motion
    if ch_vars[0] > 50000 and ch_vars[3] > 50000 and ch_vars[1] > 20000:
        return True
    return False


# ─────────────────────────────────────────────
# 4. FEATURE EXTRACTION
# ─────────────────────────────────────────────
def extract_band_powers(window, fs=FS):
    """
    Log band power for delta/theta/alpha/beta per channel.
    Expects window already bandpass-filtered.
    Returns 16 features: 4 channels x 4 bands.
    """
    features = []
    freqs = np.fft.rfftfreq(window.shape[0], d=1.0 / fs)
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
# 5. LOAD DATA
# ─────────────────────────────────────────────
print("Loading data...")

if FEATURES_PRECOMPUTED:
    # ── Old format: CSV already has 16 band-power features ───────────────
    X_raw = np.loadtxt(EEG_CSV,   delimiter=',').astype(np.float32)  # (N, 16)
    Y_raw = np.loadtxt(LABEL_CSV, delimiter=',').astype(np.float32)  # (N,)
    print(f"  Loaded pre-extracted features: {X_raw.shape}")

else:
    # ── New format: raw EEG from DataCollectScript ────────────────────────
    # inputData.csv shape: (N, WINDOW_SIZE * N_CHANNELS)
    eeg_flat = np.loadtxt(EEG_CSV,   delimiter=',').astype(np.float32)
    Y_all    = np.loadtxt(LABEL_CSV, delimiter=',').astype(np.float32)
    print(f"  Loaded raw EEG: {eeg_flat.shape}")

    # Reshape to (N, WINDOW_SIZE, N_CHANNELS)
    eeg_wins = eeg_flat.reshape(-1, WINDOW_SIZE, N_CHANNELS)

    # ── Artifact rejection ────────────────────────────────────────────────
    keep_mask  = np.array([not is_artifact(eeg_wins[i]) for i in range(len(eeg_wins))])
    n_rejected = np.sum(~keep_mask)
    print(f"  Artifact rejection: {n_rejected}/{len(eeg_wins)} windows removed "
          f"({100*n_rejected/len(eeg_wins):.1f}%)")
    eeg_wins = eeg_wins[keep_mask]
    Y_all    = Y_all[keep_mask]

    # ── Bandpass + feature extraction ─────────────────────────────────────
    print("  Applying bandpass filter and extracting features...")
    X_raw = np.array([
        extract_band_powers(bandpass_filter(eeg_wins[i]))
        for i in range(len(eeg_wins))
    ], dtype=np.float32)
    Y_raw = Y_all
    print(f"  Feature matrix: {X_raw.shape}")


# ─────────────────────────────────────────────
# 6. LABEL CHECK + NORMALIZE
# ─────────────────────────────────────────────
unique, counts = np.unique(Y_raw, return_counts=True)
print("\nLabel distribution:", dict(zip(unique, counts)))
print("Expected: 0=rest/head movement, 1=jaw clench (finger open)")

scaler = StandardScaler()
X = scaler.fit_transform(X_raw).astype(np.float32)

np.save('scaler_mean2.npy',  scaler.mean_)
np.save('scaler_scale2.npy', scaler.scale_)

Y = Y_raw.astype(np.float32)


# ─────────────────────────────────────────────
# 7. TRAIN / VAL / TEST SPLIT
# ─────────────────────────────────────────────
# 10% held out as completely unseen test set
# Remaining 90% split into 80% train / 20% val
X_temp,  X_test,  Y_temp,  Y_test  = train_test_split(
    X, Y, test_size=0.10, random_state=42, stratify=Y
)
X_train, X_val,   Y_train, Y_val   = train_test_split(
    X_temp, Y_temp, test_size=0.20, random_state=42, stratify=Y_temp
)

print(f"\nSplit  Train: {X_train.shape} | Val: {X_val.shape} | Test: {X_test.shape}")


# ─────────────────────────────────────────────
# 8. MODEL ARCHITECTURE
# ─────────────────────────────────────────────
def build_model(n_features):
    inputs = keras.Input(shape=(n_features,), name="eeg_input")

    x = layers.Dense(
        64, activation='gelu',
        kernel_regularizer=regularizers.l2(1e-4),
        kernel_initializer='glorot_uniform',
        name="dense_1"
    )(inputs)
    x = layers.BatchNormalization(name="bn_1")(x)
    x = layers.Dropout(0.4, name="drop_1")(x)

    x = layers.Dense(
        32, activation='gelu',
        kernel_regularizer=regularizers.l2(1e-4),
        name="dense_2"
    )(x)
    x = layers.BatchNormalization(name="bn_2")(x)
    x = layers.Dropout(0.3, name="drop_2")(x)

    x = layers.Dense(
        16, activation='gelu',
        kernel_regularizer=regularizers.l2(1e-4),
        name="dense_3"
    )(x)
    x = layers.Dropout(0.2, name="drop_3")(x)

    # >0.5 → CLENCH (finger opens)
    # <0.5 → REST   (finger stays closed)
    outputs = layers.Dense(1, activation='sigmoid', name="output")(x)

    return keras.Model(inputs, outputs, name="EEG_FingerNet_v2")

model = build_model(N_FEATURES)
model.summary()


# ─────────────────────────────────────────────
# 9. COMPILE
# ─────────────────────────────────────────────
model.compile(
    loss      = 'binary_crossentropy',
    optimizer = keras.optimizers.Adam(
        learning_rate=LEARNING_RATE,
        beta_1=0.9, beta_2=0.999
    ),
    metrics   = ['accuracy']
)


# ─────────────────────────────────────────────
# 10. CALLBACKS
# ─────────────────────────────────────────────
callbacks = [
    keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=20,
        restore_best_weights=True,
        verbose=1
    ),
    keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=10,
        min_lr=1e-6,
        verbose=1
    ),
    keras.callbacks.ModelCheckpoint(
        MODEL_PATH,
        monitor='val_accuracy',
        save_best_only=True,
        verbose=1
    ),
]


# ─────────────────────────────────────────────
# 11. TRAIN
# ─────────────────────────────────────────────
history = model.fit(
    X_train, Y_train,
    validation_data = (X_val, Y_val),
    epochs          = EPOCHS,
    batch_size      = BATCH_SIZE,
    callbacks       = callbacks,
    verbose         = 1
)


# ─────────────────────────────────────────────
# 12. EVALUATE — val + unseen test
# ─────────────────────────────────────────────
val_loss,  val_acc  = model.evaluate(X_val,  Y_val,  verbose=0)
test_loss, test_acc = model.evaluate(X_test, Y_test, verbose=0)

print(f"\nVal  Accuracy: {val_acc:.4f}  | Val  Loss: {val_loss:.4f}")
print(f"Test Accuracy: {test_acc:.4f}  | Test Loss: {test_loss:.4f}")

if abs(val_acc - test_acc) > 0.05:
    print("  WARNING: Val and Test accuracy differ by >5% — possible overfitting.")
else:
    print("  OK: Val and Test accuracy are close — model generalises well.")

preds = model.predict(X_val[:10]).flatten()
print("\nSample predictions (first 10 val samples):")
for i, (p, true) in enumerate(zip(preds, Y_val[:10])):
    label      = "CLENCH" if p    > 0.5 else "REST"
    true_label = "CLENCH" if true == 1  else "REST"
    match      = "OK" if label == true_label else "XX"
    print(f"  [{i}] {match}  output={p:.3f} -> predicted={label} | actual={true_label}")


# ─────────────────────────────────────────────
# 13. PLOT
# ─────────────────────────────────────────────
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

ax1.plot(history.history['accuracy'],     label='Train Acc')
ax1.plot(history.history['val_accuracy'], label='Val Acc')
ax1.set_title('Accuracy'); ax1.legend(); ax1.grid(True)

ax2.plot(history.history['loss'],     label='Train Loss')
ax2.plot(history.history['val_loss'], label='Val Loss')
ax2.set_title('Loss'); ax2.legend(); ax2.grid(True)

plt.tight_layout()
plt.savefig('training_curves.png', dpi=150)
plt.show()


# ─────────────────────────────────────────────
# 14. CONVERT TO TFLITE
# ─────────────────────────────────────────────
print("\nConverting to TensorFlow Lite...")

converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]

def representative_dataset():
    for i in range(min(100, len(X_train))):
        yield [X_train[i:i+1]]

converter.representative_dataset       = representative_dataset
converter.target_spec.supported_ops    = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.inference_input_type         = tf.float32
converter.inference_output_type        = tf.float32

tflite_model = converter.convert()

with open(TFLITE_PATH, 'wb') as f:
    f.write(tflite_model)

size_kb = os.path.getsize(TFLITE_PATH) / 1024
print(f"TFLite model saved: {TFLITE_PATH} ({size_kb:.1f} KB)")
print("\nFiles to copy to Pi:")
print(f"  - {TFLITE_PATH}")
print(f"  - scaler_mean2.npy")
print(f"  - scaler_scale2.npy")