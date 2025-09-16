# Lung Disease Detection from Respiratory Sounds

## 📌 Overview
This repository implements a **hybrid deep learning framework** for automatic **multi-class lung disease detection** from respiratory sounds.  
The model integrates **deep audio features (mel-spectrogram + CNN–BiLSTM–Attention)** with **handcrafted acoustic features** (MFCCs, chroma, ZCR, spectral centroid, bandwidth).  
Explainability is achieved using **Grad-CAM, Integrated Gradients, and SHAP** for different feature branches.

### Target Diseases
- Bronchial
- Asthma
- COPD
- Healthy
- Pneumonia

## 🏗️ Model Architecture
The model consists of two parallel branches:

1. **Mel-Spectrogram Branch**
   - Input: 4s audio → Mel-Spectrogram (128 × ~250)
   - 3 Conv2D blocks with BatchNorm, ReLU, MaxPooling, Dropout
   - Flattened via `TimeDistributed`
   - Bidirectional LSTM (128 units × 2 directions)
   - Additive Attention → temporal context vector
   - 256-dim embedding

2. **Handcrafted Feature Branch**
   - Features: MFCC, Chroma, ZCR, Spectral Centroid, Bandwidth
   - Total dimension ≈ 70
   - Fully connected network (Dense(256) → Dense(128))

3. **Fusion + Classification**
   - Concatenate embeddings (256 + 128 = 384)
   - Dense(256) + Dropout
   - Output Softmax layer (5 classes)


## ⚙️ Features
- End-to-end **deep + handcrafted feature fusion**
- Robust **data augmentations**: pitch shift, time-stretch, noise injection
- Explainable AI (XAI) methods:
  - **Grad-CAM** on mel spectrogram
  - **Integrated Gradients** on mel spectrogram
  - **SHAP** values on handcrafted features
- Evaluation metrics:
  - Accuracy, Loss, ROC-AUC, Confusion Matrix, Classification Report
  - Per-class AUC, Micro- and Macro-averaged ROC curves

## 📂 Dataset
- **Asthma Detection Dataset Version 2** (from Kaggle)
- Structure:
  ```
  dataset/
  ├── Bronchial/*.wav
  ├── asthma/*.wav
  ├── copd/*.wav
  ├── healthy/*.wav
  ├── pneumonia/*.wav
  ```

## 🚀 Training
- Optimizer: Adam (`lr=3e-4`, weight decay = 1e-4)
- Loss: Sparse Categorical Crossentropy
- Regularization: Dropout + Early Stopping
- Batch size: 16
- Epochs: 100 (with early stopping at 70)

## 📊 Results
- Strong validation and test accuracy across all classes
- ROC-AUC > 0.90 for all of the classes
- Grad-CAM & IG show meaningful attention on disease-relevant regions
- SHAP highlights important handcrafted features (MFCCs, spectral properties)

## 🔍 Explainability Examples
- **Grad-CAM** overlays class activation maps on mel-spectrograms
- **Integrated Gradients** highlights frequency bands most influential
- **SHAP** plots show feature importance of handcrafted features

## 📦 Installation
```bash
# Install dependencies (if on Colab/Kaggle, adjust as needed)
pip install numpy scipy pandas matplotlib seaborn librosa soundfile scikit-learn tensorflow==2.15.0 shap
```

## ▶️ Usage
1. Clone repo:
   ```bash
   git clone https://github.com/yourusername/lung-disease-detection.git
   cd lung-disease-detection
   ```
2. Prepare dataset under `data_dir` path inside `CFG` class
3. Run notebook or training script
4. Evaluate using built-in metrics
5. Visualize XAI results

## 📈 Visualization
- **Training Curves**: Accuracy & loss over epochs
- **Confusion Matrix**: Per-class classification performance
- **ROC Curves**: One-vs-rest, micro/macro average
- **XAI Visualizations**: Grad-CAM overlays, Integrated Gradients, SHAP barplots

## 🧠 Future Work
- Expand dataset with more diseases (e.g., Tuberculosis, COVID-19 coughs)
- Deploy as **web app** with real-time inference
- Use **transformer-based encoders** (AST, Wav2Vec2) for stronger embeddings

## 👨‍💻 Author
- **S M Asiful Islam Saky**  
Bachelor of Computer Science (Specialization: Data Science)  
Researcher in AI/ML/DL  
Skills: Python, TensorFlow, NLP, Data Science, Explainable AI

## 📜 License
This repository is licensed under the MIT License.
