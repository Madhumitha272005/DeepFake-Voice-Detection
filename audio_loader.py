import librosa

def load_audio(path):
    y, sr = librosa.load(path, sr=16000)
    return y, sr
