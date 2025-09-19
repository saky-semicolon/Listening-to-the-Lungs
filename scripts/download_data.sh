#!/usr/bin/env bash
set -e
echo "Downloading Asthma Detection Dataset (Kaggle)..."
if ! command -v kaggle &> /dev/null; then
  echo "kaggle CLI not found. Install: pip install kaggle"
  exit 1
fi

# target path
mkdir -p data/raw
kaggle datasets download -d mohammedtawfikmusaed/asthma-detection-dataset-version-2 -p data/raw --unzip
echo "Downloaded to data/raw/"

# optionally track with DVC (user decision)
echo "To track with DVC: run 'dvc add data/raw' then 'git add data/raw.dvc' and commit."
