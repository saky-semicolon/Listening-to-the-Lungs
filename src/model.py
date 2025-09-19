import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

class AdditiveAttention(layers.Layer):
    def __init__(self, d_model, **kwargs):
        super().__init__(**kwargs)
        self.w = layers.Dense(d_model, activation="tanh")
        self.v = layers.Dense(1, use_bias=False)

    def call(self, x):
        s = self.w(x)
        s = self.v(s)
        a = tf.nn.softmax(s, axis=1)
        ctx = tf.reduce_sum(x * a, axis=1)
        return ctx, a

def conv_block(x, filters):
    x = layers.Conv2D(filters, (3,3), padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.Conv2D(filters, (3,3), padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.MaxPool2D(pool_size=(2,2))(x)
    x = layers.Dropout(0.2)(x)
    return x

def build_model(hand_dim, num_classes, lr=3e-4):
    mel_in  = layers.Input(shape=(128, None, 1), name="mel")
    hand_in = layers.Input(shape=(hand_dim,), name="handcrafted")

    x = conv_block(mel_in, 32)
    x = conv_block(x, 64)
    x = conv_block(x, 128)
    x = layers.Lambda(lambda z: z, name="conv_tail_identity")(x)

    x = layers.Permute((2,1,3))(x)
    x = layers.TimeDistributed(layers.Flatten())(x)

    x = layers.Bidirectional(layers.LSTM(128, return_sequences=True))(x)
    ctx, _ = AdditiveAttention(256, name="temporal_attention")(x)

    mel_emb = layers.Dropout(0.3)(ctx)

    h = layers.Dense(256)(hand_in)
    h = layers.BatchNormalization()(h)
    h = layers.ReLU()(h)
    h = layers.Dropout(0.3)(h)
    h = layers.Dense(128, activation="relu")(h)

    z = layers.Concatenate()([mel_emb, h])
    z = layers.Dense(256, activation="relu")(z)
    z = layers.Dropout(0.3)(z)
    out = layers.Dense(num_classes, activation="softmax", dtype="float32")(z)

    model = keras.Model(inputs=[mel_in, hand_in], outputs=[out])
    opt = keras.optimizers.Adam(learning_rate=lr)
    model.compile(optimizer=opt, loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model
