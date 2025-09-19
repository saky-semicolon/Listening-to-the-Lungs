import tensorflow as tf
from tensorflow import keras

def train_model(model, train_xy, val_xy, cfg, work_dir="work_tf"):
    ckpt_path = f"{work_dir}/best_model.keras"
    callbacks = [
        keras.callbacks.ModelCheckpoint(ckpt_path, save_best_only=True,
                                        monitor="val_accuracy", mode="max"),
        keras.callbacks.EarlyStopping(patience=cfg.early_stop_pat,
                                      restore_best_weights=True,
                                      monitor="val_accuracy", mode="max"),
        keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=4,
                                          monitor="val_loss", mode="min", verbose=1),
    ]
    history = model.fit(
        train_xy,
        validation_data=val_xy,
        epochs=cfg.epochs,
        callbacks=callbacks,
        verbose=1
    )
    return history
