import librosa
import soundfile as sf
from pathlib import Path
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import time
from scipy.signal import correlate
import pandas as pd
from scipy.io import wavfile

SAMPLE_RATE = 44100

HOP_LENGTH = 512

#index of highest 10 matched samples in full audio, take the same place in the full recording

def give_ten_potential_answer(similarity):
    top_10_sample = np.argpartition(similarity, -10)[-10:]
    top_10_idx = top_10_sample[np.argsort(similarity[top_10_sample])[::-1]]
    top_10_seconds = top_10_idx * HOP_LENGTH / SAMPLE_RATE

    with np.printoptions(formatter={'float': lambda x: f"{x:.2f} seconds"}):
        print(np.array(top_10_seconds))


full = np.array([0.20, 0.25, 0.28, 0.2, 0.2, 0.40, 0.30, 0.26, 0.328, 0.620, 0.33, 0.520, 0.320, 0.4, 0.36, 0.52, 0.480, 0.2, 0.2, 0.3, 0.3, 0.4, 0.4])

give_ten_potential_answer(full)

