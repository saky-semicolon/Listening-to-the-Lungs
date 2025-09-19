#!/usr/bin/env python3
"""
run.py - simple entrypoint for common tasks:
  - python run.py train    (runs a smoke training run if real data not present)
  - python run.py xai      (runs a small Grad-CAM demo on a random batch)
  - python run.py infer --wav path/to.wav  (demo inference)
"""
import argparse
import os
import numpy as np

from src import build_model, train_model, CFG
from src.xai import grad_cam
from src.features import load_wav, pad_or_trim, extract_mel, extract_handcrafted

def make_dummy_dataset(batch_size=4, time_steps=64, hand_dim=70, num_classes=5):
    # Creates small random tensors for a smoke-run
    X_mel = np.random.randn(batch_size, CFG.n_mels, time_steps, 1).astype(np.float32)
    X_hand = np.random.randn(batch_size, hand_dim).astype(np.float32)
    y = np.random.randint(0, num_classes, size=(batch_size,)).astype(np.int32)
    return (X_mel, X_hand), y

def train(args):
    # If real data present (work_tf/train.csv) instruct user to run full pipeline.
    if not os.path.exists(os.path.join(CFG.work_dir, "train.csv")):
        print("No train.csv found in work_dir. Running a small smoke training run with dummy data.")
        (X_mel, X_hand), y = make_dummy_dataset(batch_size=4, time_steps=64, hand_dim=70, num_classes=len(CFG.classes))
        model = build_model(hand_dim=70, num_classes=len(CFG.classes), lr=1e-3)
        # Convert to tf datasets (small)
        import tensorflow as tf
        ds = tf.data.Dataset.from_tensor_slices(((X_mel, X_hand), y)).batch(2)
        val_ds = ds.take(1)
        history = train_model(model, ds, val_ds, CFG, work_dir=CFG.work_dir)
        print("Smoke training finished. Best model (if saved) located at:", os.path.join(CFG.work_dir, "best_model.keras"))
    else:
        print("Found train.csv. Please run the full training pipeline (not implemented in run.py).")
        print("Use src/data.py and src/train.py to build full TF pipelines.")

def xai_demo(args):
    print("Running small Grad-CAM demo on random batch...")
    (X_mel, X_hand), y = make_dummy_dataset(batch_size=4, time_steps=64, hand_dim=70, num_classes=len(CFG.classes))
    model = build_model(hand_dim=70, num_classes=len(CFG.classes))
    preds = model.predict([X_mel, X_hand])
    cams = grad_cam(model, X_mel, X_hand)
    print("Produced CAMs shape:", cams.shape)
    # Save one CAM + mel image
    import matplotlib.pyplot as plt
    plt.imsave(os.path.join(CFG.work_dir, "gradcam_demo.png"), cams[0].mean(axis=0), cmap="jet")
    print("Saved demo image:", os.path.join(CFG.work_dir, "gradcam_demo.png"))

def infer(args):
    wav = args.wav
    if not wav:
        print("Please provide --wav path")
        return
    y = load_wav(wav)
    y = pad_or_trim(y)
    mel = extract_mel(y, sr=CFG.sample_rate)
    mel = np.expand_dims(np.transpose(mel, (0,1)), axis=-1)[None,...]  # (1, M, T, 1)
    hand = extract_handcrafted(y, sr=CFG.sample_rate)[None,...]
    model_path = os.path.join(CFG.work_dir, "best_model.keras")
    if not os.path.exists(model_path):
        print("No trained model found at", model_path)
        return
    import tensorflow as tf
    model = tf.keras.models.load_model(model_path, compile=False)
    pred = model.predict([mel, hand])
    cls = int(pred.argmax(-1)[0])
    print("Predicted class:", CFG.classes[cls], "prob:", float(pred[0,cls]))

def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd")
    sub.add_parser("train")
    sub.add_parser("xai")
    p_inf = sub.add_parser("infer")
    p_inf.add_argument("--wav", type=str, help="Path to WAV file")
    args = parser.parse_args()

    if args.cmd == "train":
        train(args)
    elif args.cmd == "xai":
        xai_demo(args)
    elif args.cmd == "infer":
        infer(args)
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
