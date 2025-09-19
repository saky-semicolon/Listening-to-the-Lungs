# app.py
"""
Streamlit demo app for Listening-to-the-Lungs
Features:
 - Upload or record audio (.wav)
 - Display waveform and mel-spectrogram
 - Extract & display handcrafted features (MFCC, ZCR, centroid...)
 - Run model inference (loads work_tf/best_model.keras if available)
 - Explainability: Grad-CAM overlay, Integrated Gradients, SHAP for handcrafted features
 - Demo examples & About section
"""

import io
import os
import tempfile
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import librosa
import librosa.display
import soundfile as sf
import streamlit as st

# UI styling
st.set_page_config(page_title="Listening to the Lungs — Demo", layout="wide")

# Import project modules (must exist in repo src/)
try:
    from src.features import extract_mel, extract_handcrafted, load_wav, pad_or_trim
    from src.model import build_model
    from src.xai import grad_cam, integrated_gradients, shap_explain
    from src.data import CFG
except Exception as e:
    # Graceful fallback: we'll try to use local helper functions if src not available
    st.warning("Could not import project modules from src/. Some functionality might be limited.\n"
               "Error: {}".format(e))
    # Provide minimal local implementations
    def load_wav(path, target_sr=16000):
        y, sr = sf.read(path)
        if y.ndim > 1:
            y = y.mean(axis=1)
        if sr != target_sr:
            y = librosa.resample(y, sr, target_sr)
        return y.astype(np.float32)

    def pad_or_trim(y, target_len=int(4.0 * 16000)):
        n = len(y)
        if n < target_len:
            y = np.pad(y, (0, target_len - n), mode="constant")
        return y[:target_len]

    def extract_mel(y, sr, n_mels=128):
        S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels)
        return librosa.power_to_db(S, ref=np.max)

    def extract_handcrafted(y, sr):
        zcr = np.mean(librosa.feature.zero_crossing_rate(y))
        centroid = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))
        bandwidth = np.mean(librosa.feature.spectral_bandwidth(y=y, sr=sr))
        return np.array([zcr, centroid, bandwidth])

    class DummyCFG:
        sample_rate = 16000
        n_mels = 128
        work_dir = "work_tf"
        classes = ("Bronchial", "asthma", "copd", "healthy", "pneumonia")
    CFG = DummyCFG()

# Utilities
def bytes_to_tempfile(uploaded_file) -> str:
    """Save uploaded BytesIO to a temp .wav file and return path."""
    suffix = ".wav"
    tf = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    tf.write(uploaded_file.read())
    tf.flush()
    tf.close()
    return tf.name

@st.cache_resource
def load_tf_model(path):
    import tensorflow as tf
    if not os.path.exists(path):
        return None
    try:
        model = tf.keras.models.load_model(path, compile=False)
        return model
    except Exception as e:
        st.error(f"Failed to load model at {path}: {e}")
        return None

def plot_waveform(y, sr, ax=None):
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 2))
    else:
        fig = ax.figure
    librosa.display.waveshow(y, sr=sr, ax=ax)
    ax.set_ylabel("Amplitude")
    ax.set_xlabel("Time (s)")
    return fig

def plot_mel_spec(mel, sr, hop_length=512, ax=None):
    # mel is in dB (n_mels, t)
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 3))
    else:
        fig = ax.figure
    img = librosa.display.specshow(mel, sr=sr, x_axis='time', y_axis='mel', ax=ax)
    ax.set_ylabel("Mel bins")
    plt.colorbar(img, ax=ax, format="%+2.0f dB")
    return fig

def show_feature_table(handcraft_vec):
    # Simple mapping for display
    cols = ["zcr", "centroid", "bandwidth"] if len(handcraft_vec) == 3 else [f"f{i}" for i in range(len(handcraft_vec))]
    df = pd.DataFrame([handcraft_vec], columns=cols)
    st.table(df.T.rename(columns={0: "value"}))

def predict_and_show(model, mel, hand):
    """
    mel: (M, T) floating db
    hand: 1D vector
    Convert mel to model input shape (1, M, T, 1), hand: (1, D)
    """
    import numpy as _np
    if model is None:
        st.info("No model available — running demo random prediction.")
        probs = _np.random.dirichlet(np.ones(len(CFG.classes))).reshape(1, -1)
    else:
        # prepare
        X_m = _np.expand_dims(np.transpose(mel, (0,1)), axis=(0, -1))  # (1,M,T,1)
        X_h = _np.expand_dims(hand, axis=0).astype(_np.float32)
        try:
            probs = model.predict([X_m, X_h], verbose=0)
        except Exception as e:
            st.error(f"Model prediction failed: {e}")
            probs = _np.random.dirichlet(np.ones(len(CFG.classes))).reshape(1, -1)
    top_idx = int(probs.argmax(axis=1)[0])
    table = pd.DataFrame({
        "class": list(CFG.classes),
        "prob": probs.ravel()
    }).sort_values("prob", ascending=False).reset_index(drop=True)
    st.subheader("Prediction")
    st.write(f"Predicted: **{CFG.classes[top_idx]}**  (prob={float(probs[0, top_idx]):.3f})")
    st.bar_chart(table.set_index("class"))
    st.write(table)

def run_gradcam_and_show(model, mel, hand):
    st.subheader("Grad-CAM (spectrogram overlay)")
    try:
        # model may be None: handle gracefully
        if model is None:
            st.info("No model loaded, Grad-CAM demo unavailable.")
            return
        # prepare inputs
        X_m = np.expand_dims(np.transpose(mel, (0,1)), axis=(0, -1)).astype(np.float32)
        X_h = np.expand_dims(hand, axis=0).astype(np.float32)
        cams, chosen_classes, preds = grad_cam(model, X_m, X_h, conv_layer_name=None)
        cam = cams[0]  # (M,T)
        # show overlay
        fig, ax = plt.subplots(1,1, figsize=(8,3))
        librosa.display.specshow(mel, sr=CFG.sample_rate, x_axis='time', y_axis='mel', ax=ax)
        ax.imshow(cam, origin='lower', aspect='auto', cmap='jet', alpha=0.5)
        ax.set_title("Grad-CAM overlay")
        st.pyplot(fig)
    except Exception as e:
        st.error(f"Grad-CAM failed: {e}")

def run_ig_and_show(model, mel, hand):
    st.subheader("Integrated Gradients (mel contribution)")
    try:
        if model is None:
            st.info("No model loaded, Integrated Gradients demo unavailable.")
            return
        X_m = np.expand_dims(np.transpose(mel, (0,1)), axis=(0, -1)).astype(np.float32)
        X_h = np.expand_dims(hand, axis=0).astype(np.float32)
        # choose predicted class
        preds = model.predict([X_m, X_h])
        c = int(np.argmax(preds, axis=1)[0])
        ig_map = integrated_gradients(model, X_m, X_h, target_class=c, steps=30)
        ig = ig_map[0].mean(axis=-1) if ig_map.ndim == 4 else ig_map[0]
        fig, ax = plt.subplots(1,1, figsize=(8,3))
        librosa.display.specshow(mel, sr=CFG.sample_rate, x_axis='time', y_axis='mel', ax=ax)
        ax.imshow(ig, origin='lower', aspect='auto', cmap='inferno', alpha=0.6)
        ax.set_title("Integrated Gradients overlay")
        st.pyplot(fig)
    except Exception as e:
        st.error(f"Integrated Gradients failed: {e}")

def run_shap_for_handcrafted(model, mel, hand):
    st.subheader("SHAP for Handcrafted Features")
    try:
        if model is None:
            st.info("No model loaded, SHAP demo unavailable.")
            return
        # shap may be heavy; run with single sample approximation
        shap_vals = shap_explain(model, np.expand_dims(np.transpose(mel, (0,1)), axis=(0,-1)).astype(np.float32),
                                 np.expand_dims(hand, axis=0).astype(np.float32), nsamples=50)
        # shap_values structure depends on explainer; try to display for handcrafted (index 1)
        if hasattr(shap_vals, "values"):
            hv = shap_vals.values[1]  # handcrafted contributions
            hv = np.array(hv).squeeze()
            if hv.ndim == 2:  # (samples, features)
                hv = hv[0]
            df = pd.DataFrame({"feature": [f"f{i}" for i in range(len(hv))], "shap": hv})
            st.bar_chart(df.set_index("feature"))
        else:
            st.write("SHAP returned type:", type(shap_vals))
    except Exception as e:
        st.error(f"SHAP explanation failed or is heavy in this environment: {e}")

# App layout
st.title("Listening to the Lungs — Demo")
st.markdown("Upload a lung sound (.wav) and run inference + explanations. "
            "This demo uses the project's model (if available) or runs a lightweight fallback.")

# Sidebar controls
with st.sidebar:
    st.header("Controls")
    model_path = st.text_input("Model path", value=os.path.join(CFG.work_dir, "best_model.keras"))
    load_model_btn = st.button("Load model")
    if "model_obj" not in st.session_state:
        st.session_state.model_obj = None
    if load_model_btn:
        st.session_state.model_obj = load_tf_model(model_path)
        if st.session_state.model_obj is None:
            st.warning("Model not loaded. Using fallback/dummy behavior.")
        else:
            st.success("Model loaded successfully.")

    st.markdown("---")
    st.markdown("### Examples")
    if st.button("Use example audio (from dataset)"):
        # try to load an example file if dataset available
        sample_dir = os.path.join("data", "raw")
        if os.path.isdir(sample_dir):
            # pick first wav
            files = [f for f in os.listdir(sample_dir) if f.lower().endswith(".wav")]
            if files:
                st.session_state.example_file = os.path.join(sample_dir, files[0])
                st.success(f"Selected example: {files[0]}")
            else:
                st.warning("No .wav files found in data/raw/")
        else:
            st.warning("data/raw/ not present in repo.")
    st.markdown("---")
    st.write("About:")
    st.caption("Demo: upload .wav → visualize → predict → explain\nModel architecture: Hybrid CNN + BiLSTM + Attention + handcrafted features.")

# Main: file upload and display
col1, col2 = st.columns([1, 1])
with col1:
    uploaded = st.file_uploader("Upload WAV file", type=["wav", "mp3", "flac"])
    if "example_file" in st.session_state and st.session_state.example_file:
        use_example = st.checkbox("Use example file", value=False)
        if use_example:
            uploaded = open(st.session_state.example_file, "rb")

    if uploaded is not None:
        # get local path
        if isinstance(uploaded, str):
            wav_path = uploaded
        else:
            # uploaded is UploadedFile object
            wav_path = bytes_to_tempfile(uploaded)

        # play audio
        try:
            st.audio(wav_path)
        except Exception:
            # if bytes-like object
            with open(wav_path, "rb") as f:
                st.audio(f.read())

        # load audio
        try:
            y = load_wav(wav_path, target_sr=CFG.sample_rate)
            y = pad_or_trim(y)
        except Exception as e:
            st.error(f"Failed to load audio: {e}")
            st.stop()

        # Visualizations
        st.subheader("Waveform")
        fig_w = plot_waveform(y, sr=CFG.sample_rate)
        st.pyplot(fig_w)

        st.subheader("Mel-Spectrogram (dB)")
        mel = extract_mel(y, sr=CFG.sample_rate, n_mels=CFG.n_mels)
        fig_m = plot_mel_spec(mel, sr=CFG.sample_rate)
        st.pyplot(fig_m)

        st.subheader("Handcrafted Features")
        hand = extract_handcrafted(y, sr=CFG.sample_rate)
        show_feature_table(hand)

        # Prediction & XAI tabs
        model_to_use = st.session_state.model_obj or load_tf_model(model_path)
        predict_and_show(model_to_use, mel, hand)

        tabs = st.tabs(["Grad-CAM", "Integrated Gradients", "SHAP (handcrafted)"])
        with tabs[0]:
            run_gradcam_and_show(model_to_use, mel, hand)
        with tabs[1]:
            run_ig_and_show(model_to_use, mel, hand)
        with tabs[2]:
            st.write("SHAP may be slow; please be patient.")
            run_shap_for_handcrafted(model_to_use, mel, hand)

        # Clean up temp file if created
        if not isinstance(uploaded, str):
            try:
                os.unlink(wav_path)
            except Exception:
                pass
    else:
        st.info("Please upload a WAV file to get started. Or use the example from data/raw if present.")

with col2:
    st.header("Quick Demo")
    st.markdown("Try the small utilities below:")
    if st.button("Run smoke inference (random)"):
        model_tmp = load_tf_model(model_path)
        # create dummy sample
        X_m = np.random.randn(1, CFG.n_mels, 64, 1).astype(np.float32)
        X_h = np.random.randn(1, 70).astype(np.float32)
        if model_tmp is None:
            st.write("No model loaded. Example random prediction:")
            probs = np.random.dirichlet(np.ones(len(CFG.classes)))
        else:
            try:
                preds = model_tmp.predict([X_m, X_h], verbose=0)
                probs = preds[0]
            except Exception as e:
                st.error(f"Inference failed: {e}")
                probs = np.random.dirichlet(np.ones(len(CFG.classes)))
        table = pd.DataFrame({'class': CFG.classes, 'prob': probs}).sort_values('prob', ascending=False)
        st.write(table)
        st.bar_chart(table.set_index("class"))

    st.markdown("---")
    st.header("About / Methodology")
    st.markdown("""
    - Dataset: Asthma Detection Dataset v2 (Kaggle)
    - Model: Hybrid CNN (mel) + BiLSTM + Attention + Handcrafted features (late fusion)
    - Explanations: Grad-CAM (spectrogram), Integrated Gradients, SHAP (handcrafted)
    """)
    st.markdown("**Notes:**\n- Explanations (IG/SHAP) can be computationally heavy. Use small steps/samples in demo mode.")
    st.markdown("**Citation:** Tawfik et al. (2022) Asthma Detection System ... (see DATASET.md)")

st.caption("If you want a persistent demo, consider deploying this Streamlit app to HuggingFace Spaces (free) or Streamlit Cloud.")
