import librosa
import numpy as np

def extract_mel(y, sr, n_mels=128):
    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels)
    mel_db = librosa.power_to_db(mel, ref=np.max)
    return mel_db

def extract_mfcc(y, sr, n_mfcc=20):
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    return mfcc

def extract_handcrafted(y, sr):
    zcr = np.mean(librosa.feature.zero_crossing_rate(y))
    centroid = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))
    bandwidth = np.mean(librosa.feature.spectral_bandwidth(y=y, sr=sr))
    return np.array([zcr, centroid, bandwidth])

# --- I/O helpers expected by the app and runner ---
def load_wav(path, target_sr=16000):
    """Load an audio file as mono float32 at target sample rate."""
    y, sr = librosa.load(path, sr=None, mono=True)
    if sr != target_sr:
        y = librosa.resample(y, orig_sr=sr, target_sr=target_sr)
        sr = target_sr
    return y.astype(np.float32)

def pad_or_trim(y, target_len=int(4.0 * 16000)):
    """Pad with zeros or trim to a fixed number of samples."""
    n = len(y)
    if n < target_len:
        y = np.pad(y, (0, target_len - n), mode="constant")
    return y[:target_len]
