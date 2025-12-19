# inference/predict.py

import librosa
import numpy as np
import tensorflow as tf

MODEL_PATH = "models/cnn_model.h5"

model = tf.keras.models.load_model(MODEL_PATH)

def predict_audio(audio_file):
    # Load audio
    y, sr = librosa.load(audio_file, sr=16000)

    # Extract MFCC
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
    mfcc = np.mean(mfcc.T, axis=0)

    # Reshape for model
    mfcc = mfcc.reshape(1, 40, 1)

    # Prediction
    prob = model.predict(mfcc)[0][0]

    label = "Deepfake Voice" if prob > 0.5 else "Real Voice"

    return label, float(prob)
