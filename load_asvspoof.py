import os
import librosa
import numpy as np

def load_asvspoof(data_path, label_file, max_files=None):
    X, y = [], []

    with open(label_file, "r") as f:
        lines = f.readlines()

    for i, line in enumerate(lines):
        if max_files and i >= max_files:
            break

        parts = line.strip().split()
        file_id = parts[0]
        label_text = parts[-1]

        label = 0 if label_text == "bonafide" else 1
        audio_path = os.path.join(data_path, "flac", file_id + ".flac")

        if not os.path.exists(audio_path):
            continue

        audio, sr = librosa.load(audio_path, sr=16000)
        mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=40)
        mfcc = np.mean(mfcc.T, axis=0)

        X.append(mfcc)
        y.append(label)

    return np.array(X), np.array(y)
