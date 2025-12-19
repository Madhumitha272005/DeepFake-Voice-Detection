import tensorflow as tf
import numpy as np

X = np.load("data/processed/X.npy")
y = np.load("data/processed/y.npy")

X = X.reshape(-1, 40, 1)

model = tf.keras.Sequential([
    tf.keras.layers.Conv1D(32, 3, activation='relu', input_shape=(40,1)),
    tf.keras.layers.MaxPooling1D(),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.fit(X, y, epochs=20, batch_size=32)

model.save("models/cnn_model.h5")
