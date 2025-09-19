# 📂 Dataset: Asthma Detection Dataset (Version 2)

## Overview  
The **Asthma Detection Dataset: Version 2** is a publicly available respiratory audio dataset hosted on **Kaggle**. It is designed for research into diagnosing asthma and other lung conditions using ML / DL methods.  

This dataset consists of 1,211 audio samples (1.5–5 seconds long) with clean recordings, classified into five lung condition categories.

---

## Dataset Link  
You can download the dataset from Kaggle:

[Asthma Detection Dataset: Version 2 on Kaggle](https://www.kaggle.com/datasets/mohammedtawfikmusaed/asthma-detection-dataset-version-2) :contentReference[oaicite:2]{index=2}

---

## Composition  
- Total Samples: **1,211**  
- Duration per sample: **1.5–5 seconds**  
- Classes & sample counts:  
  - Asthma: 288  
  - Bronchial: 104  
  - COPD: 401  
  - Healthy: 133  
  - Pneumonia: 255  

---

## Preprocessing Summary  
- Noise reduction, amplification (WavePad)  
- Segmentation to standard duration (5–6 seconds approx)  
- Removal of incomplete / noisy samples  

---

## Usage in this Project  

We use **DVC** to version the dataset while keeping code repository light.  
Steps to obtain data:

```bash
# Clone this repo
git clone <your-github-repo-url>
cd <repo-folder>

# Install dependencies, ensure Kaggle credentials are set
pip install dvc kaggle

# If using Kaggle API option:
kaggle datasets download -d mohammedtawfikmusaed/asthma-detection-dataset-version-2 -p data/raw/ --unzip

# Or, if dataset already tracked via DVC remote:
dvc pull
