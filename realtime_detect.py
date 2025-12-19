import sounddevice as sd
import queue

q = queue.Queue()

def callback(indata, frames, time, status):
    q.put(indata.copy())

with sd.InputStream(callback=callback, samplerate=16000):
    while True:
        chunk = q.get()
        # preprocess & predict
