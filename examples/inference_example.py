"""
Example: run inference on a WAV file:
python examples/inference_example.py path/to/file.wav
"""
import sys
from src.features import load_wav, pad_or_trim, extract_mel, extract_handcrafted
from src import CFG
import numpy as np
import tensorflow as tf

def main(wav_path):
    y = load_wav(wav_path)
    y = pad_or_trim(y)
    mel = extract_mel(y, sr=CFG.sample_rate)
    mel = np.expand_dims(np.transpose(mel, (0,1)), axis=-1)[None,...]
    hand = extract_handcrafted(y, sr=CFG.sample_rate)[None,...]
    model = tf.keras.models.load_model(f"{CFG.work_dir}/best_model.keras", compile=False)
    pred = model.predict([mel, hand])
    cls = int(pred.argmax(-1)[0])
    print("Predicted:", CFG.classes[cls], pred[0, cls])

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python examples/inference_example.py path/to/file.wav")
    else:
        main(sys.argv[1])
