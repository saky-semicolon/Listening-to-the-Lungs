import numpy as np
from src.model import build_model

def test_build_and_predict():
    hand_dim = 70
    C = 5
    model = build_model(hand_dim=hand_dim, num_classes=C, lr=1e-3)
    # small random batch: (B, 128, T, 1)
    X_mel = np.random.randn(2, 128, 32, 1).astype('float32')
    X_hand = np.random.randn(2, hand_dim).astype('float32')
    preds = model.predict([X_mel, X_hand], verbose=0)
    assert preds.shape == (2, C)
