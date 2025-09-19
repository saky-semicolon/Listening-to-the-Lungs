#!/usr/bin/env python
# coding: utf-8

# # Multi Class Lungs Disease Prediction

# In[ ]:


import os, random, numpy as np, tensorflow as tf
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"           # quieter logs
# Optional if ONEDNN layout logs are noisy on CPU:
# os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

# Force channels_last (you’re using 2D convs on spectrograms):
tf.keras.backend.set_image_data_format("channels_last")

# Seeds
SEED = 42
random.seed(SEED); np.random.seed(SEED); tf.random.set_seed(SEED)

# GPU memory growth (avoid OOM spikes)
gpus = tf.config.experimental.list_physical_devices('GPU')
for g in gpus: tf.config.experimental.set_memory_growth(g, True)
print(tf.__version__)


# ## 0) Setup & Config

# ### 0.1) Imports

# In[ ]:


import os, random, math, time, glob, json, itertools
from pathlib import Path
from typing import Tuple, Dict, List

import numpy as np
import pandas as pd
import librosa, soundfile as sf
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

print("TF version:", tf.__version__)


# ### 0.2) Reproducibility

# In[ ]:


SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)
os.environ['PYTHONHASHSEED'] = str(SEED)


# ### 0.3) Config

# In[ ]:


class CFG:
    # Paths
    data_dir        = "/kaggle/input/asthma-detection-dataset-version-2/Asthma Detection Dataset Version 2/Asthma Detection Dataset Version 2"
    work_dir        = "work_tf"

    # Classes
    classes         = ("Bronchial", "asthma", "copd", "healthy", "pneumonia")
    num_classes     = len(classes)

    # Audio params
    sample_rate     = 16000
    duration_sec    = 4.0
    n_fft           = 1024
    hop_length      = 256
    n_mels          = 128
    fmin            = 20
    fmax            = 8000
    n_mfcc          = 20

    # Augmentations
    aug_prob        = 0.8
    time_stretch    = (0.9, 1.1)    # +/- 10%
    pitch_steps     = 2             # +/- 2 semitones
    noise_snr_db    = (15, 30)      # target SNR range

    # Training
    batch_size      = 16
    epochs          = 100
    lr              = 3e-4
    weight_decay    = 1e-4
    early_stop_pat  = 12
    mixed_precision = False
    label_smoothing = 0.05
    class_weights   = None

    # Splits
    test_size       = 0.15
    val_size        = 0.15
    seed            = 42


# ### 0.4) Setup

# In[ ]:


Path(CFG.work_dir).mkdir(parents=True, exist_ok=True)
random.seed(CFG.seed); np.random.seed(CFG.seed); tf.random.set_seed(CFG.seed)

if CFG.mixed_precision:
    from tensorflow.keras import mixed_precision as mp
    mp.set_global_policy("mixed_float16")


# ### 0.5) Derived values

# In[ ]:


def sec_to_samples(seconds, sr):
    return int(round(seconds * sr))

TARGET_SAMPLES = sec_to_samples(CFG.duration_sec, CFG.sample_rate)

print(f"Duration: {CFG.duration_sec}s | Sample Rate: {CFG.sample_rate}Hz | TARGET_SAMPLES: {TARGET_SAMPLES}")
print(f"Classes: {CFG.classes} | Total: {CFG.num_classes}")


# ## 1) Index Files & Stratified Split

# ### 1.1) Dataset Indexing

# In[ ]:


def build_index(data_dir: str) -> pd.DataFrame:
    """Scan dataset and build a dataframe with paths and labels."""
    rows = []
    for lbl in CFG.classes:
        folder = Path(data_dir) / lbl
        if not folder.exists():
            print(f"Warning: folder not found for class '{lbl}' -> {folder}")
            continue
        for wav in folder.glob("*.wav"):
            rows.append({"path": str(wav), "label": lbl})

    if len(rows) == 0:
        raise RuntimeError("No WAV files found. Please check data directory structure.")

    df = pd.DataFrame(rows)

    # Map labels to integer IDs
    label2id = {c: i for i, c in enumerate(CFG.classes)}
    id2label = {i: c for c, i in label2id.items()}
    df["label_id"] = df["label"].map(label2id)

    print(f"Found {len(df)} files across {len(df['label'].unique())} classes")
    return df, label2id, id2label

# Build dataset index
df_all, label2id, id2label = build_index(CFG.data_dir)

df_all.head(), df_all["label"].value_counts()


# ### 1.2) Train/Val/Test Split

# In[ ]:


from sklearn.model_selection import train_test_split

# Build dataset index
df_all, label2id, id2label = build_index(CFG.data_dir)

# Split stratified
df_trainval, df_test = train_test_split(
    df_all,
    test_size=CFG.test_size,
    stratify=df_all["label_id"],
    random_state=CFG.seed
)

val_rel = CFG.val_size / (1.0 - CFG.test_size)
df_train, df_val = train_test_split(
    df_trainval,
    test_size=val_rel,
    stratify=df_trainval["label_id"],
    random_state=CFG.seed
)

# Save splits
df_train.to_csv(f"{CFG.work_dir}/train.csv", index=False)
df_val.to_csv(f"{CFG.work_dir}/val.csv", index=False)
df_test.to_csv(f"{CFG.work_dir}/test.csv", index=False)

# Save label mapping
with open(f"{CFG.work_dir}/labels.json", "w") as f:
    json.dump({"label2id": label2id, "id2label": id2label}, f, indent=2)

# Summary
print("Dataset split completed")
print(f"Train: {len(df_train)} | Val: {len(df_val)} | Test: {len(df_test)}")
print("\nTrain distribution:\n", df_train["label"].value_counts())
print("\nVal distribution:\n", df_val["label"].value_counts())
print("\nTest distribution:\n", df_test["label"].value_counts())


# ## 2) Audio I/O, Augmentation, and Feature Extraction

# ### 2.1) Optional: quiet some TF graph layout warnings (set BEFORE TF imports)

# In[ ]:


import os
os.environ.setdefault("TF_ENABLE_LAYOUT_OPTIMIZER", "0")  # purely optional

import numpy as np
import librosa
import soundfile as sf


# ### 2.2) CFG-safe accessors (prevents AttributeError during tf.data)

# In[ ]:


def cfg_get(name, default):
    # Assumes a global CFG object exists elsewhere in your notebook/script
    return getattr(CFG, name, default)

# Helper
def sec_to_samples(s: float, sr: int) -> int:
    return int(round(float(s) * int(sr)))

TARGET_SAMPLES = sec_to_samples(cfg_get("duration_s", 5.0), cfg_get("sample_rate", 16000))


# ### 2.3) Loading and Padding

# In[ ]:


def load_wav(path: str, target_sr=None) -> np.ndarray:
    if target_sr is None:
        target_sr = cfg_get("sample_rate", 16000)

    y, sr = sf.read(path)
    if y.ndim > 1:                   # stereo → mono
        y = y.mean(axis=1)
    if sr != target_sr:
        y = librosa.resample(y, orig_sr=sr, target_sr=target_sr)
    return y.astype(np.float32)

def pad_or_trim(y: np.ndarray, target_len: int = None) -> np.ndarray:
    if target_len is None:
        target_len = TARGET_SAMPLES
    n = len(y)
    if n < target_len:
        y = np.pad(y, (0, target_len - n), mode="constant")
    else:
        y = y[:target_len]
    return y


# ### 2.4) Augmentation

# In[ ]:


def maybe_time_stretch(y: np.ndarray, sr=None) -> np.ndarray:
    """Random time stretching; mono-safe; librosa-compatible."""
    if np.random.rand() < 0.5:
        if sr is None:
            sr = cfg_get("sample_rate", 16000)
        span = cfg_get("time_stretch", (0.9, 1.1))
        rate = float(np.random.uniform(*span))
        if y.ndim > 1:
            y = librosa.to_mono(y)
        y = np.ascontiguousarray(y, dtype=np.float32)
        y = librosa.effects.time_stretch(y, rate=rate)
        y = pad_or_trim(y, TARGET_SAMPLES)
    return y

def maybe_pitch_shift(y: np.ndarray, sr=None) -> np.ndarray:
    """
    Random pitch shift.
    - Uses CFG.pitch_shift_steps = (lo, hi) if present (float range, e.g., (-3.0, 3.0))
    - Otherwise falls back to CFG.pitch_steps = k (int), sampling from [-k, k] uniformly
    """
    if np.random.rand() < 0.5:
        if sr is None:
            sr = cfg_get("sample_rate", 16000)

        ps_range = cfg_get("pitch_shift_steps", None)
        if ps_range is not None:
            n_steps = float(np.random.uniform(*ps_range))
        else:
            k = int(cfg_get("pitch_steps", 2))
            n_steps = float(np.random.randint(-k, k + 1))  # includes 0

        y = np.ascontiguousarray(y, dtype=np.float32)
        y = librosa.effects.pitch_shift(y, sr=sr, n_steps=n_steps)
        y = pad_or_trim(y, TARGET_SAMPLES)
    return y

def maybe_add_noise(y: np.ndarray) -> np.ndarray:
    """Random noise addition based on SNR (in dB)."""
    if np.random.rand() < 0.5:
        snr_db_lo, snr_db_hi = cfg_get("noise_snr_db", (10.0, 30.0))
        snr_db = float(np.random.uniform(snr_db_lo, snr_db_hi))
        sp = float(np.mean(y ** 2)) + 1e-9
        npow = sp / (10 ** (snr_db / 10.0))
        noise = np.random.normal(0.0, np.sqrt(npow), size=y.shape).astype(np.float32)
        y = (y + noise).astype(np.float32)
    return y

def apply_augs(y: np.ndarray, sr=None) -> np.ndarray:
    """Apply all augmentations with a single coin flip gate (CFG.aug_prob)."""
    if sr is None:
        sr = cfg_get("sample_rate", 16000)
    if np.random.rand() < cfg_get("aug_prob", 0.8):
        y = maybe_time_stretch(y, sr)
        y = maybe_pitch_shift(y, sr)
        y = maybe_add_noise(y)
    return y


# ### 2.5) Features

# In[ ]:


def compute_melspec(y: np.ndarray, sr=None) -> np.ndarray:
    if sr is None:
        sr = cfg_get("sample_rate", 16000)
    S = librosa.feature.melspectrogram(
        y=y,
        sr=sr,
        n_fft=cfg_get("n_fft", 1024),
        hop_length=cfg_get("hop_length", 256),
        n_mels=cfg_get("n_mels", 128),
        fmin=cfg_get("fmin", 20.0),
        fmax=cfg_get("fmax", sr // 2),
        power=2.0,
    )
    S_db = librosa.power_to_db(S, ref=np.max)
    S_db = (S_db - S_db.mean()) / (S_db.std() + 1e-6)  # standardize per sample
    return S_db.astype(np.float32)  # (n_mels, time)

def compute_handcrafted(y: np.ndarray, sr=None) -> np.ndarray:
    if sr is None:
        sr = cfg_get("sample_rate", 16000)

    n_fft = cfg_get("n_fft", 1024)
    hop = cfg_get("hop_length", 256)
    fmin = cfg_get("fmin", 20.0)
    fmax = cfg_get("fmax", sr // 2)
    n_mfcc = cfg_get("n_mfcc", 20)

    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc, n_fft=n_fft, hop_length=hop, fmin=fmin, fmax=fmax)
    chroma = librosa.feature.chroma_stft(y=y, sr=sr, n_fft=n_fft, hop_length=hop)
    zcr = librosa.feature.zero_crossing_rate(y, hop_length=hop)
    centroid = librosa.feature.spectral_centroid(y=y, sr=sr, n_fft=n_fft, hop_length=hop)
    bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr, n_fft=n_fft, hop_length=hop)

    def agg(X):
        # mean+std across frames
        return np.concatenate([X.mean(axis=1), X.std(axis=1)])

    feats = np.concatenate([agg(mfcc), agg(chroma), agg(zcr), agg(centroid), agg(bandwidth)]).astype(np.float32)
    return feats  # ~ 2*(n_mfcc + 12 + 1 + 1 + 1) = 2*(n_mfcc + 15)


# ## 3) EDA

# In[ ]:


import os
import librosa
import librosa.display
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from collections import Counter
import pandas as pd

# Dataset path
data_path = "/kaggle/input/asthma-detection-dataset-version-2/Asthma Detection Dataset Version 2/Asthma Detection Dataset Version 2"

# Class distribution
classes = os.listdir(data_path)
samples_per_class = {cls: len(os.listdir(os.path.join(data_path, cls))) for cls in classes}

print("Dataset Summary:")
print(f"Total Classes: {len(classes)}")
print(f"Classes: {classes}")
print(f"Samples per class: {samples_per_class}")


# ### 3.1) Plot class distribution

# In[ ]:


plt.figure(figsize=(8,5))
sns.barplot(x=list(samples_per_class.keys()), y=list(samples_per_class.values()))
plt.title("Class Distribution")
plt.ylabel("Number of Samples")
plt.xlabel("Class")
plt.xticks(rotation=45)
plt.show()


# ### 3.2) Show waveform and spectrogram for a random sample

# In[ ]:


example_class = classes[0]
example_file = os.path.join(data_path, example_class, os.listdir(os.path.join(data_path, example_class))[0])


# ### 3.3) Waveform

# In[ ]:


y, sr = librosa.load(example_file)
plt.figure(figsize=(12,4))
librosa.display.waveshow(y, sr=sr)
plt.title(f"Waveform - {example_class}")
plt.show()


# ### 3.4) Spectogram

# In[ ]:


plt.figure(figsize=(12,4))
D = librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref=np.max)
librosa.display.specshow(D, sr=sr, x_axis='time', y_axis='log')
plt.colorbar(format='%+2.0f dB')
plt.title(f"Spectrogram - {example_class}")
plt.show()


# ## 4) tf.data Pipelines (Generators with Augment for Train)

# ### 4.1) Inspect handcrafted feature dimension

# In[ ]:


tmp_path = df_train.iloc[0]["path"]
y_tmp = pad_or_trim(load_wav(tmp_path))
hand_dim = compute_handcrafted(y_tmp).shape[0]
print("Handcrafted feature dim:", hand_dim)

# Label mappings (already saved in labels.json earlier)
label2id = {c: i for i, c in enumerate(CFG.classes)}
id2label = {v: k for k, v in label2id.items()}


# ### 4.2) Example extraction

# In[ ]:


def example_from_row(row, training=False):
    """Extract mel-spectrogram + handcrafted features + label from a single row."""
    y = pad_or_trim(load_wav(row["path"]))
    if training:
        y = apply_augs(y)

    mel = compute_melspec(y)          # (n_mels, T)
    hand = compute_handcrafted(y)     # (hand_dim,)
    lbl = int(row["label_id"])

    # CNN expects (H, W, 1)
    mel = np.expand_dims(mel, axis=-1)   # (n_mels, T, 1)
    return mel, hand, lbl

def generator_from_df(df: pd.DataFrame, training=False):
    for _, row in df.iterrows():
        yield example_from_row(row, training=training)


# ### 4.3) Dataset Wrapper

# In[ ]:


def make_dataset(df: pd.DataFrame, training=False, batch_size=CFG.batch_size):
    ds = tf.data.Dataset.from_generator(
        lambda: generator_from_df(df, training=training),
        output_signature=(
            tf.TensorSpec(shape=(CFG.n_mels, None, 1), dtype=tf.float32),   # mel
            tf.TensorSpec(shape=(hand_dim,), dtype=tf.float32),             # handcrafted
            tf.TensorSpec(shape=(), dtype=tf.int32),                        # label
        )
    )

    if training:
        ds = ds.shuffle(1024, seed=CFG.seed, reshuffle_each_iteration=True)

    # Pad mel time dimension across batch
    ds = ds.padded_batch(
        batch_size,
        padded_shapes=(
            (CFG.n_mels, None, 1),   # mel spectrograms (pad time dim only)
            (hand_dim,),             # handcrafted features (fixed)
            ()                       # labels
        ),
        padding_values=(
            0.0,                     # mel
            0.0,                     # handcrafted
            np.int32(0)              # labels
        )
    ).prefetch(tf.data.AUTOTUNE)

    return ds


# ### 4.4) Build datasets

# In[ ]:


train_ds = make_dataset(df_train, training=True)
val_ds   = make_dataset(df_val, training=False)
test_ds  = make_dataset(df_test, training=False)

print("tf.data pipelines ready:")
print("Train batches:", len(list(train_ds)))
print("Val batches:", len(list(val_ds)))
print("Test batches:", len(list(test_ds)))


# ## 5) Model: CNN → BiLSTM → Attention (Mel branch) + MLP (handcrafted) → Late Fusion

# ### 5.1) Attention layer (temporal)

# In[ ]:


class AdditiveAttention(layers.Layer):
    def __init__(self, d_model, **kwargs):
        super().__init__(**kwargs)
        self.w = layers.Dense(d_model, activation="tanh")
        self.v = layers.Dense(1, use_bias=False)

    def call(self, x):  # x: (B,T,D)
        s = self.w(x)                 # (B,T,D)
        s = self.v(s)                 # (B,T,1)
        a = tf.nn.softmax(s, axis=1)  # (B,T,1)
        ctx = tf.reduce_sum(x * a, axis=1)  # (B,D)
        return ctx, a


# ### 5.2) Convolutional Block

# In[ ]:


def conv_block(x, filters):
    x = layers.Conv2D(filters, (3, 3), padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.Conv2D(filters, (3, 3), padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.MaxPool2D(pool_size=(2, 2))(x)
    x = layers.Dropout(0.2)(x)
    return x


# ### 5.3) Model Builder

# In[ ]:


def build_model(hand_dim: int, num_classes=len(CFG.classes)) -> keras.Model:
    # Inputs
    mel_in  = layers.Input(shape=(CFG.n_mels, None, 1), name="mel")     # (M,T,1)
    hand_in = layers.Input(shape=(hand_dim,), name="handcrafted")       # (D,)

    # CNN over Mel
    x = conv_block(mel_in, 32)
    x = conv_block(x, 64)
    x = conv_block(x, 128)  # final conv block

    # Save last conv for Grad-CAM++
    x = layers.Lambda(lambda z: z, name="conv_tail_identity")(x)

    # (B, M', T', C) -> (B, T', M', C)
    x = layers.Permute((2, 1, 3))(x)

    # Flatten freq*channels -> features, time stays
    x = layers.TimeDistributed(layers.Flatten())(x)  # (B, T', F)

    # Temporal modeling
    x = layers.Bidirectional(layers.LSTM(128, return_sequences=True))(x)  # (B,T,256)

    # Attention
    ctx, attn = AdditiveAttention(256, name="temporal_attention")(x)      # ctx: (B,256)

    mel_emb = layers.Dropout(0.3)(ctx)

    # --- Handcrafted branch ---
    h = layers.Dense(256)(hand_in)
    h = layers.BatchNormalization()(h)
    h = layers.ReLU()(h)
    h = layers.Dropout(0.3)(h)
    h = layers.Dense(128, activation="relu")(h)

    # Late fusion
    z = layers.Concatenate()([mel_emb, h])        # (B, 256+128)
    z = layers.Dense(256, activation="relu")(z)
    z = layers.Dropout(0.3)(z)
    out = layers.Dense(num_classes, activation="softmax", dtype="float32")(z)

    # Compile model
    model = keras.Model(inputs=[mel_in, hand_in], outputs=[out], name="HybridCNN_BiLSTM_Attn")
    opt = keras.optimizers.Adam(learning_rate=CFG.lr)
    model.compile(optimizer=opt, loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model


# ### 5.4) Dataset Formatting and Splitting

# In[ ]:


# Dataset formatting
def split_xy(ds):
    # Map from (mel, hand, y) → ((mel, hand), y)
    return ds.map(lambda mel, hand, y: ((mel, hand), y), 
                  num_parallel_calls=tf.data.AUTOTUNE)


# In[ ]:


# Ensure hand_dim is computed correctly
tmp_path = df_train.iloc[0]["path"]
y_tmp = pad_or_trim(load_wav(tmp_path))
hand_dim = compute_handcrafted(y_tmp).shape[0]
print("Handcrafted feature dim:", hand_dim)  # should be > 0, e.g. 70

# Rebuild model with correct input shape
model = build_model(hand_dim=hand_dim, num_classes=len(CFG.classes))
model.summary()


# In[ ]:


# Prepare datasets again (they must match new model inputs)
train_xy = split_xy(train_ds)
val_xy   = split_xy(val_ds)
test_xy  = split_xy(test_ds)


# In[ ]:


# Training setup
ckpt_path = f"{CFG.work_dir}/best_model.keras"
callbacks = [
    keras.callbacks.ModelCheckpoint(
        ckpt_path, save_best_only=True, monitor="val_accuracy", mode="max"
    ),
    keras.callbacks.EarlyStopping(
        patience=CFG.early_stop_pat, restore_best_weights=True,
        monitor="val_accuracy", mode="max"
    ),
    keras.callbacks.ReduceLROnPlateau(
        factor=0.5, patience=4, monitor="val_loss", mode="min", verbose=1
    ),
]


# ### 5.5) Train model

# In[ ]:


history = model.fit(
    train_xy,
    validation_data=val_xy,
    epochs=CFG.epochs,
    callbacks=callbacks,
    verbose=1
)


# ## 6) Evaluation

# In[ ]:


from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import pandas as pd
import numpy as np
from typing import Tuple

# Collect predictions
def collect_preds(model, ds) -> Tuple[np.ndarray, np.ndarray]:
    y_true, y_prob = [], []
    for (mel, hand), y in ds:
        p = model.predict((mel, hand), verbose=0)
        y_true.append(y.numpy())
        y_prob.append(p)
    y_true = np.concatenate(y_true)
    y_prob = np.concatenate(y_prob)
    return y_true, y_prob


# In[ ]:


# Validation
yv_true, yv_prob = collect_preds(model, val_xy)
yv_pred = yv_prob.argmax(axis=1)

print("Validation Report:")
print(classification_report(yv_true, yv_pred, target_names=CFG.classes, zero_division=0))

cm_val = confusion_matrix(yv_true, yv_pred)
print("\n Validation Confusion Matrix:")
display(pd.DataFrame(cm_val, index=CFG.classes, columns=CFG.classes))


# In[ ]:


# Test
yt_true, yt_prob = collect_preds(model, test_xy)
yt_pred = yt_prob.argmax(axis=1)

print("\n Test Report:")
print(classification_report(yt_true, yt_pred, target_names=CFG.classes, zero_division=0))

cm_test = confusion_matrix(yt_true, yt_pred)
print("\nTest Confusion Matrix:")
display(pd.DataFrame(cm_test, index=CFG.classes, columns=CFG.classes))


# In[ ]:


# One-vs-rest ROC-AUC (macro)
Y_ovr_val = np.eye(len(CFG.classes))[yv_true]
Y_ovr_test = np.eye(len(CFG.classes))[yt_true]

val_aucs, test_aucs = [], []
for c in range(len(CFG.classes)):
    try:
        val_aucs.append(roc_auc_score(Y_ovr_val[:, c], yv_prob[:, c]))
        test_aucs.append(roc_auc_score(Y_ovr_test[:, c], yt_prob[:, c]))
    except ValueError:
        # Happens if a class is missing in y_true
        continue

if val_aucs:
    print(f"\nVal Macro ROC-AUC: {np.mean(val_aucs):.4f}")
if test_aucs:
    print(f"Test Macro ROC-AUC: {np.mean(test_aucs):.4f}")


# In[ ]:


import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

# Training Curves
plt.figure(figsize=(8, 5))
pd.DataFrame(history.history)[["loss", "val_loss", "accuracy", "val_accuracy"]].plot(grid=True, linewidth=2)
plt.title("Training Curves", fontsize=14)
plt.ylabel("Loss / Accuracy")
plt.xlabel("Epoch")
plt.tight_layout()
plt.show()


# In[ ]:


# Confusion Matrix (Test)
cm = confusion_matrix(yt_true, yt_pred)
cm_norm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]  # normalized version

plt.figure(figsize=(6, 5))
sns.heatmap(
    cm_norm, annot=True, fmt=".2f", cmap="Blues",
    xticklabels=CFG.classes, yticklabels=CFG.classes,
    cbar=True, square=True
)
plt.xlabel("Predicted", fontsize=12)
plt.ylabel("True", fontsize=12)
plt.title("Test Confusion Matrix (Normalized)", fontsize=14)
plt.tight_layout()
plt.show()


# In[ ]:


# Extract final metrics from history safely
final_train_acc = history.history.get("accuracy", [None])[-1]
final_train_loss = history.history.get("loss", [None])[-1]
final_val_acc = history.history.get("val_accuracy", [None])[-1]
final_val_loss = history.history.get("val_loss", [None])[-1]

print("\n Final Training & Validation Metrics: ")
print(f" Training Accuracy   : {final_train_acc:.4f} ({final_train_acc*100:.2f}%)")
print(f" Training Loss       : {final_train_loss:.4f}")
print(f" Validation Accuracy : {final_val_acc:.4f} ({final_val_acc*100:.2f}%)")
print(f" Validation Loss     : {final_val_loss:.4f}")


# In[ ]:


plt.figure(figsize=(12,5))

# Accuracy
plt.subplot(1,2,1)
plt.plot(history.history['accuracy'], label='Train Accuracy', marker='o')
plt.plot(history.history['val_accuracy'], label='Val Accuracy', marker='s')
plt.title("Accuracy Over Epochs")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.grid(True, linestyle="--", alpha=0.6)
plt.legend()

# Loss
plt.subplot(1,2,2)
plt.plot(history.history['loss'], label='Train Loss', marker='o')
plt.plot(history.history['val_loss'], label='Val Loss', marker='s')
plt.title("Loss Over Epochs")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.grid(True, linestyle="--", alpha=0.6)
plt.legend()

plt.tight_layout()
plt.show()


# In[ ]:


import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
from sklearn.preprocessing import label_binarize
import itertools

# 1) Evaluate on validation set
val_loss, val_acc = model.evaluate(val_xy, verbose=0)
print(f"Validation Loss: {val_loss:.4f}")
print(f"Validation Accuracy: {val_acc:.4f}")


# In[ ]:


# 2) Get predictions & true labels from val_xy
y_pred_probs = model.predict(val_xy, verbose=0)   # shape: (N, num_classes)
y_true = np.concatenate([y.numpy() for (_, _), y in val_xy], axis=0)
y_pred_classes = np.argmax(y_pred_probs, axis=1)

# Class names (from CFG)
classes = list(CFG.classes)
num_classes = len(classes)


# In[ ]:


# 3) Classification report
print("\nClassification Report (Validation):")
print(classification_report(y_true, y_pred_classes, target_names=classes, digits=4))


# In[ ]:


# 4) Confusion matrix
cm = confusion_matrix(y_true, y_pred_classes)
plt.figure(figsize=(6,5))
plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
plt.title("Confusion Matrix (Validation)")
plt.colorbar()
tick_marks = np.arange(num_classes)
plt.xticks(tick_marks, classes, rotation=45, ha='right')
plt.yticks(tick_marks, classes)

# Annotate cells
thresh = cm.max() / 2.0
for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
    plt.text(j, i, cm[i, j],
             ha="center",
             color="white" if cm[i, j] > thresh else "black",
             fontsize=9)
plt.ylabel("True label")
plt.xlabel("Predicted label")
plt.tight_layout()
plt.show()


# In[ ]:


# 5) ROC curves (one-vs-rest)
y_true_bin = label_binarize(y_true, classes=np.arange(num_classes))  # shape: (N, C)

plt.figure(figsize=(8,6))
auc_per_class = []

for i, cls_name in enumerate(classes):
    fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_pred_probs[:, i])
    roc_auc = auc(fpr, tpr)
    auc_per_class.append(roc_auc)
    plt.plot(fpr, tpr, lw=2, label=f"{cls_name} (AUC={roc_auc:.3f})")

# Micro-average ROC
fpr_micro, tpr_micro, _ = roc_curve(y_true_bin.ravel(), y_pred_probs.ravel())
auc_micro = auc(fpr_micro, tpr_micro)
plt.plot(fpr_micro, tpr_micro, linestyle="--", lw=2, label=f"Micro-average (AUC={auc_micro:.3f})")

# Macro-average ROC
auc_macro = np.mean(auc_per_class)
plt.plot([0,1],[0,1],"k--", lw=1)
plt.xlim([0.0, 1.0]); plt.ylim([0.0, 1.05])
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curves (Validation)")
plt.legend(loc="lower right", fontsize=9)
plt.tight_layout()
plt.show()

print(f"\nPer-class AUC: {dict(zip(classes, [round(a, 4) for a in auc_per_class]))}")
print(f"Micro-average AUC: {auc_micro:.4f}")
print(f"Macro-average AUC: {auc_macro:.4f}")


# In[ ]:


import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
from sklearn.preprocessing import label_binarize
import itertools

def evaluate_and_plot(model, ds, classes, set_name="Validation"):
    """Evaluate model on a dataset, print metrics, and plot confusion matrix + ROC curves."""
    # ----- 1) Evaluate -----
    loss, acc = model.evaluate(ds, verbose=0)
    print(f"\n{set_name} Loss: {loss:.4f}")
    print(f"{set_name} Accuracy: {acc:.4f}")

    # ----- 2) Predictions -----
    y_pred_probs = model.predict(ds, verbose=0)
    y_true = np.concatenate([y.numpy() for (_, _), y in ds], axis=0)
    y_pred_classes = np.argmax(y_pred_probs, axis=1)
    num_classes = len(classes)

    # ----- 3) Classification Report -----
    print(f"\nClassification Report ({set_name}):")
    print(classification_report(y_true, y_pred_classes, target_names=classes, digits=4))

    # ----- 4) Confusion Matrix -----
    cm = confusion_matrix(y_true, y_pred_classes)
    plt.figure(figsize=(6,5))
    plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    plt.title(f"Confusion Matrix ({set_name})")
    plt.colorbar()
    tick_marks = np.arange(num_classes)
    plt.xticks(tick_marks, classes, rotation=45, ha='right')
    plt.yticks(tick_marks, classes)

    thresh = cm.max() / 2.0
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 ha="center",
                 color="white" if cm[i, j] > thresh else "black",
                 fontsize=9)
    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    plt.tight_layout()
    plt.show()

    # ----- 5) ROC Curves -----
    y_true_bin = label_binarize(y_true, classes=np.arange(num_classes))
    plt.figure(figsize=(8,6))
    auc_per_class = []

    for i, cls_name in enumerate(classes):
        fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_pred_probs[:, i])
        roc_auc = auc(fpr, tpr)
        auc_per_class.append(roc_auc)
        plt.plot(fpr, tpr, lw=2, label=f"{cls_name} (AUC={roc_auc:.3f})")

    # Micro-average ROC
    fpr_micro, tpr_micro, _ = roc_curve(y_true_bin.ravel(), y_pred_probs.ravel())
    auc_micro = auc(fpr_micro, tpr_micro)
    plt.plot(fpr_micro, tpr_micro, linestyle="--", lw=2, label=f"Micro-average (AUC={auc_micro:.3f})")

    # Macro-average ROC
    auc_macro = np.mean(auc_per_class)
    plt.plot([0,1],[0,1],"k--", lw=1)
    plt.xlim([0.0, 1.0]); plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC Curves ({set_name})")
    plt.legend(loc="lower right", fontsize=9)
    plt.tight_layout()
    plt.show()

    print(f"\nPer-class AUC ({set_name}): {dict(zip(classes, [round(a, 4) for a in auc_per_class]))}")
    print(f"Micro-average AUC ({set_name}): {auc_micro:.4f}")
    print(f"Macro-average AUC ({set_name}): {auc_macro:.4f}")

    return loss, acc


# Run for Validation and Test
classes = list(CFG.classes)

val_loss, val_acc = evaluate_and_plot(model, val_xy, classes, set_name="Validation")
test_loss, test_acc = evaluate_and_plot(model, test_xy, classes, set_name="Test")


# In[ ]:


import seaborn as sns
from sklearn.metrics import confusion_matrix
import numpy as np
import matplotlib.pyplot as plt

# Compute confusion matrix
cm = confusion_matrix(y_true, y_pred_classes)

# Normalize by row (true labels)
cm_norm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]

plt.figure(figsize=(6,5))
sns.heatmap(cm_norm, annot=True, fmt=".2f", 
            xticklabels=CFG.classes, yticklabels=CFG.classes, 
            cmap="Blues", cbar=True)

plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Normalized Confusion Matrix")
plt.tight_layout()
plt.show()


# ## 7) XAI: Grad-CAM (for dual-input model: mel + handcrafted features)

# In[ ]:


import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt


def _find_last_conv_before_rnn(model):
    """Return the last Conv2D layer before any RNN. If none, return last Conv2D overall."""
    rnn_types = (tf.keras.layers.LSTM, tf.keras.layers.GRU, tf.keras.layers.SimpleRNN, tf.keras.layers.RNN)
    last_rnn_idx, last_conv_idx = None, None
    layers_list = list(model.layers)

    for i, layer in enumerate(layers_list):
        if isinstance(layer, tf.keras.layers.Conv2D):
            last_conv_idx = i
        if isinstance(layer, rnn_types):
            last_rnn_idx = i
            break

    if last_rnn_idx is None:
        if last_conv_idx is None:
            raise ValueError("No Conv2D layer found in model.")
        return layers_list[last_conv_idx].name
    else:
        for i in range(last_rnn_idx - 1, -1, -1):
            if isinstance(layers_list[i], tf.keras.layers.Conv2D):
                return layers_list[i].name
        if last_conv_idx is None:
            raise ValueError("No Conv2D layer found in model.")
        return layers_list[last_conv_idx].name


def _pack_inputs_for_model(model, mel_batch, hand_batch):
    """Return inputs in correct form for the model (dict or list)."""
    mel_batch = tf.convert_to_tensor(mel_batch)
    hand_batch = tf.convert_to_tensor(hand_batch)
    try:
        names = [str(inp.name).split(":")[0].split("/")[-1] for inp in model.inputs]
        if "mel" in names and "handcrafted" in names:
            return {"mel": mel_batch, "handcrafted": hand_batch}
    except Exception:
        pass
    return [mel_batch, hand_batch]


def grad_cam(model, mel_batch, hand_batch, class_idx=None, conv_layer_name=None):
    """
    Compute Grad-CAM heatmaps for a batch.
    Returns:
      heatmaps_resized: np.array (B, M, T)
      chosen_classes: np.array (B,)
      preds_np: np.array (B, C)
    """
    if conv_layer_name is None:
        conv_layer_name = _find_last_conv_before_rnn(model)

    grad_model = tf.keras.Model(
        inputs=model.inputs,
        outputs=[model.get_layer(conv_layer_name).output, model.output]
    )

    batch_inputs = _pack_inputs_for_model(model, mel_batch, hand_batch)
    conv_out0, preds0 = grad_model(batch_inputs, training=False)
    if isinstance(preds0, (list, tuple)):
        preds0 = preds0[0]

    preds0_np = preds0.numpy()
    B, C = preds0_np.shape

    # Decide target class for each sample
    if class_idx is None:
        cls_np = np.argmax(preds0_np, axis=1).astype(np.int32)
    else:
        cls_np = np.array(class_idx).reshape(-1).astype(np.int32)

    class_one_hot = tf.convert_to_tensor(np.eye(C, dtype=np.float32)[cls_np])

    # Gradient calculation
    with tf.GradientTape() as tape:
        conv_out, preds = grad_model(batch_inputs, training=False)
        if isinstance(preds, (list, tuple)):
            preds = preds[0]
        tape.watch(conv_out)
        y_c = tf.reduce_sum(preds * class_one_hot, axis=1)

    grads = tape.gradient(y_c, conv_out)
    if grads is None:
        raise RuntimeError("Gradients are None. Check conv layer connectivity.")

    # Global average pooling over gradients
    weights = tf.reduce_mean(grads, axis=[1, 2])  # (B,K)
    cam = tf.reduce_sum(tf.reshape(weights, (-1, 1, 1, tf.shape(conv_out)[-1])) * conv_out, axis=-1)
    cam = tf.nn.relu(cam)

    # Normalize each heatmap
    eps = 1e-8
    cam_min = tf.reduce_min(cam, axis=[1, 2], keepdims=True)
    cam_max = tf.reduce_max(cam, axis=[1, 2], keepdims=True)
    cam_norm = (cam - cam_min) / (cam_max - cam_min + eps)

    # Resize CAMs to mel spectrogram shape
    M, T = tf.shape(mel_batch)[1], tf.shape(mel_batch)[2]
    cam_resized = tf.image.resize(cam_norm[..., tf.newaxis], size=(M, T), method="bilinear")
    return tf.squeeze(cam_resized, axis=-1).numpy(), cls_np, preds0_np


def visualize_cam_on_mel(mel, cam, title="Grad-CAM", cmap_mel="magma", cmap_cam="jet", alpha=0.4):
    """Overlay Grad-CAM heatmap on mel spectrogram."""
    mel2d = mel[..., 0] if mel.ndim == 3 else mel
    plt.figure(figsize=(8, 4))
    plt.imshow(mel2d, origin="lower", aspect="auto", cmap=cmap_mel)
    plt.imshow(cam, origin="lower", aspect="auto", cmap=cmap_cam, alpha=alpha)
    plt.title(title)
    plt.tight_layout()
    plt.show()


# In[ ]:


# Demo: run Grad-CAM on one validation batch
for (mel_b, hand_b), y_b in val_xy.take(1):
    heatmaps, chosen_classes, preds = grad_cam(model, mel_b, hand_b)
    n = min(10, mel_b.shape[0])  # Show up to 10 samples
    for i in range(n):
        pred_idx = int(np.argmax(preds[i]))
        true_idx = int(y_b.numpy()[i])
        title = f"True: {CFG.classes[true_idx]} | Pred: {CFG.classes[pred_idx]}"
        visualize_cam_on_mel(mel_b[i].numpy(), heatmaps[i], title=title)
    break


# 10 Samples

# In[ ]:


# Note: val_xy should be your tf.data dataset returning ((mel, hand), label)
for (mel_b, hand_b), y_b in val_xy.take(1):
    # Run Grad-CAM (auto-detects last Conv2D if conv_layer_name=None)
    heatmaps, chosen_classes, preds = grad_cam(model, mel_b, hand_b, conv_layer_name=None)

    # Number of samples to visualize (up to 10)
    n = min(10, mel_b.shape[0])
    for i in range(n):
        pred_idx = int(np.argmax(preds[i]))
        # Handle tensor or numpy labels
        true_idx = int(y_b.numpy()[i]) if hasattr(y_b, "numpy") else int(y_b[i])
        title = f"True: {CFG.classes[true_idx]} | Pred: {CFG.classes[pred_idx]}"
        visualize_cam_on_mel(mel_b[i].numpy(), heatmaps[i], title=title)

    break


# In[ ]:


get_ipython().system('pip install tf-keras-vis')


# In[ ]:


# Integrated Gradients for spectrogram input
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tf_keras_vis.saliency import Saliency
from tf_keras_vis.utils.scores import CategoricalScore
from tf_keras_vis.utils.model_modifiers import ReplaceToLinear

def integrated_gradients(model, mel_batch, hand_batch, target_class=None, n_samples=50):
    attributions = []

    for i in range(len(mel_batch)):
        mel_input = tf.convert_to_tensor(mel_batch[i:i+1])
        hand_input = tf.convert_to_tensor(hand_batch[i:i+1])

        # Baseline: all zeros
        mel_baseline = tf.zeros_like(mel_input)
        hand_baseline = tf.zeros_like(hand_input)

        # Interpolate
        alphas = tf.linspace(0.0, 1.0, n_samples)
        mel_interp = mel_baseline + alphas[:, None, None, None] * (mel_input - mel_baseline)
        hand_interp = hand_baseline + alphas[:, None] * (hand_input - hand_baseline)

        # Expand batch
        mel_interp = tf.cast(mel_interp, tf.float32)
        hand_interp = tf.cast(hand_interp, tf.float32)

        with tf.GradientTape() as tape:
            tape.watch([mel_interp, hand_interp])
            preds = model([mel_interp, hand_interp], training=False)

            if target_class is None:
                target_class = tf.argmax(preds[0])

            scores = preds[:, target_class]

        grads = tape.gradient(scores, [mel_interp, hand_interp])
        mel_grads, hand_grads = grads

        # Average over steps
        avg_mel_grads = tf.reduce_mean(mel_grads, axis=0).numpy()
        avg_hand_grads = tf.reduce_mean(hand_grads, axis=0).numpy()

        # Normalize mel
        mel_attr = avg_mel_grads * (mel_input[0].numpy() - mel_baseline[0].numpy())
        mel_attr = (mel_attr - mel_attr.min()) / (mel_attr.max() - mel_attr.min() + 1e-8)

        # Normalize hand
        hand_attr = avg_hand_grads * (hand_input[0].numpy() - hand_baseline[0].numpy())
        hand_attr = (hand_attr - hand_attr.min()) / (hand_attr.max() - hand_attr.min() + 1e-8)

        attributions.append((mel_attr, hand_attr))

    return attributions

def visualize_ig_on_mel(mel, ig, title="Integrated Gradients"):
    """Overlay IG heatmap on mel spectrogram."""
    mel2d = mel[..., 0] if mel.ndim == 3 else mel
    plt.figure(figsize=(8,4))
    plt.imshow(mel2d, origin="lower", aspect="auto", cmap="magma")
    plt.imshow(ig, origin="lower", aspect="auto", cmap="hot", alpha=0.4)
    plt.title(title)
    plt.tight_layout()
    plt.show()


# In[ ]:


# SHAP for handcrafted features
import shap

def explain_handcrafted_with_shap(model, hand_samples, background_samples, max_display=15):
    """
    Use SHAP to explain handcrafted features.
    Args:
        model              : trained dual-input model
        hand_samples       : handcrafted feature array (N, F)
        background_samples : background handcrafted features (for SHAP baseline)
        max_display        : number of features to display
    """
    # Wrap the model so it only takes handcrafted input
    def f_handcrafted(x):
        # Expand x to match batch size and pass dummy mel
        dummy_mel = np.zeros((x.shape[0], *model.input[0].shape[1:]), dtype=np.float32)
        preds = model([dummy_mel, x], training=False)
        return preds.numpy()

    explainer = shap.Explainer(f_handcrafted, background_samples)
    shap_values = explainer(hand_samples)

    # Plot summary
    shap.summary_plot(shap_values, hand_samples, plot_type="bar", max_display=max_display)
    return shap_values


# In[ ]:


for (mel_b, hand_b), y_b in val_xy.take(5):
    heatmaps, _, _ = grad_cam(model, mel_b, hand_b)
    visualize_cam_on_mel(mel_b[0].numpy(), heatmaps[0])


# In[ ]:


# Note: val_xy should be your tf.data dataset returning ((mel, hand), label)
for (mel_b, hand_b), y_b in val_xy.take(1):
    # Run Grad-CAM (auto-detects last Conv2D if conv_layer_name=None)
    heatmaps, chosen_classes, preds = grad_cam(model, mel_b, hand_b, conv_layer_name=None)

    # Number of samples to visualize (up to 10)
    n = min(10, mel_b.shape[0])
    for i in range(n):
        pred_idx = int(np.argmax(preds[i]))
        # Handle tensor or numpy labels
        true_idx = int(y_b.numpy()[i]) if hasattr(y_b, "numpy") else int(y_b[i])
        title = f"True: {CFG.classes[true_idx]} | Pred: {CFG.classes[pred_idx]}"
        visualize_cam_on_mel(mel_b[i].numpy(), heatmaps[i], title=title)

    break


# In[ ]:


for (mel_b, hand_b), y_b in val_xy.take(1):
    attributions = integrated_gradients(model, mel_b, hand_b)
    mel_attr, hand_attr = attributions[0]  # take first sample

    visualize_ig_on_mel(mel_b[0].numpy(), mel_attr)  # heatmap on spectrogram
    print("Hand feature attribution:", hand_attr)    # show feature importances


# In[ ]:


# Note: val_xy should be your tf.data dataset returning ((mel, hand), label)
for (mel_b, hand_b), y_b in val_xy.take(1):
    # --- Grad-CAM ---
    heatmaps, chosen_classes, preds = grad_cam(model, mel_b, hand_b, conv_layer_name=None)

    # --- Integrated Gradients ---
    attributions = integrated_gradients(model, mel_b, hand_b)

    # Number of samples to visualize (up to 10)
    n = min(10, mel_b.shape[0])
    for i in range(n):
        pred_idx = int(np.argmax(preds[i]))
        true_idx = int(y_b.numpy()[i]) if hasattr(y_b, "numpy") else int(y_b[i])
        title = f"True: {CFG.classes[true_idx]} | Pred: {CFG.classes[pred_idx]}"

        # Grad-CAM heatmap visualization
        visualize_cam_on_mel(mel_b[i].numpy(), heatmaps[i], title=title)

        # Integrated Gradients visualization
        mel_attr, hand_attr = attributions[i]
        visualize_ig_on_mel(mel_b[i].numpy(), mel_attr, title=title)
        print(f"Hand-crafted feature attribution for sample {i}: {hand_attr}")

    break


# ### Notes & tips
# 
# If you prefer a specific conv layer (e.g. the last conv block), pass its name in conv_layer_name="conv_tail_identity" (or whatever your layer is named).
# 
# If your model's inputs are named differently than "mel" and "handcrafted", the _pack_inputs_for_model will pass positional inputs; you can adapt to named keys as needed.
# 
# This version uses first-order Grad-CAM which is reliable, widely used, and compatible with GPU/CuDNN RNN acceleration. Grad-CAM++ (higher-order) gives finer localization but requires ops that may not support second/third derivatives on GPU; for reproducible training & visualization I recommend this first-order Grad-CAM.
# 
# If any file is silent and librosa prints warnings (empty frequency tuning), you may ignore or filter those silent files during preprocessing.
# 
# If you paste this block into your notebook it should run without the previous cuDNN / stacking / dtype errors. If you still see any error, paste the full traceback here and I’ll patch it immediately.
