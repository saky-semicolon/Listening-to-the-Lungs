import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

# Grad-CAM
def grad_cam(model, mel_batch, hand_batch, conv_layer_name=None):
    conv_layer = conv_layer_name or "conv_tail_identity"
    grad_model = tf.keras.Model(
        inputs=model.inputs,
        outputs=[model.get_layer(conv_layer).output, model.output]
    )
    with tf.GradientTape() as tape:
        conv_out, preds = grad_model([mel_batch, hand_batch], training=False)
        class_idx = tf.argmax(preds, axis=1)
        one_hot = tf.one_hot(class_idx, preds.shape[-1])
        loss = tf.reduce_sum(preds * one_hot, axis=1)
    grads = tape.gradient(loss, conv_out)
    weights = tf.reduce_mean(grads, axis=(1,2))
    cam = tf.reduce_sum(weights[:, None, None, :] * conv_out, axis=-1)
    cam = tf.nn.relu(cam)
    cam /= (tf.reduce_max(cam, axis=(1,2), keepdims=True) + 1e-8)
    return cam.numpy()

def visualize_cam_on_mel(mel, cam, title="Grad-CAM"):
    mel2d = mel[...,0] if mel.ndim==3 else mel
    plt.imshow(mel2d, origin="lower", aspect="auto", cmap="magma")
    plt.imshow(cam, origin="lower", aspect="auto", cmap="jet", alpha=0.4)
    plt.title(title)
    plt.show()

# Integrated Gradients (simple)
def integrated_gradients(model, mel, hand, target_class, steps=50):
    baseline = tf.zeros_like(mel)
    alphas = tf.linspace(0.0, 1.0, steps)
    integrated = tf.zeros_like(mel, dtype=tf.float32)
    for alpha in alphas:
        x = baseline + alpha * (mel - baseline)
        with tf.GradientTape() as tape:
            tape.watch(x)
            preds = model([x, hand], training=False)
            loss = preds[:, target_class]
        grads = tape.gradient(loss, x)
        integrated += grads
    avg_grads = integrated / steps
    return ((mel - baseline) * avg_grads).numpy()

# SHAP (for handcrafted features)
import shap
def shap_explain(model, mel, hand, nsamples=50):
    explainer = shap.Explainer(model, [mel, hand])
    shap_values = explainer([mel, hand], nsamples=nsamples)
    return shap_values
