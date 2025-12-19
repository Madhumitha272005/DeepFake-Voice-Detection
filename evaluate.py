from sklearn.metrics import classification_report
import numpy as np
import tensorflow as tf

model = tf.keras.models.load_model("models/cnn_model.h5")
X = np.load("data/processed/X.npy")
y = np.load("data/processed/y.npy")

pred = (model.predict(X) > 0.5).astype(int)
print(classification_report(y, pred))
