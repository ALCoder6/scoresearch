import librosa
import soundfile
from pathlib import Path
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import time


file_path_full = "videoplayback.wav"

file_path_match = "matchingplayback.wav"

def plot_values_histogram(spectrogram_freq):
    plt.hist(spectrogram_freq.ravel(), bins=100, range=(1, spectrogram_freq.max()))
    plt.show()

def signal_to_db(signal):
    stft = librosa.stft(signal)
    spectrogram = np.abs(stft)
    spectrogram_db = librosa.amplitude_to_db(spectrogram)
    return spectrogram_db

def signal_to_freq(signal):
    stft = librosa.stft(signal, n_fft=2048)
    spectrogram = np.abs(stft)
    return spectrogram

def main(): 
    signal_full, sample_rate = librosa.load(file_path_full, sr=None, duration=10.0)
    signal_match, _ = librosa.load(file_path_match, sr=sample_rate)

    spectrogram_freq = signal_to_freq(signal_full)
    full = np.array(spectrogram_freq)

    spectrogram_freq_cut = np.where(full < 2, 0, full)
    

    plot_values_histogram(spectrogram_freq_cut)

if __name__ == "__main__":
    main()