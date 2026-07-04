import librosa
from pathlib import Path
import librosa.display
import matplotlib.pyplot as plt
import numpy as np

file_path = "videoplayback5.mp3"

def plot_spectrogram_and_save(signal, sample_rate, output_path: Path):
    stft = librosa.stft(signal)
    spectrogram = np.abs(stft)
    spectrogram_db = librosa.amplitude_to_db(spectrogram)

    plt.figure(figsize=(10,4))
    img = librosa.display.specshow(spectrogram_db, 
    y_axis='log', x_axis='time', sr=sample_rate, cmap='inferno')
    plt.show()

def main(): 
    signal, sample_rate = librosa.load(file_path, sr=None)

    print(f"File loaded successfully!")
    print(f"Sample Rate: {sample_rate} Hz")
    print(f"Total Audio Samples: {len(signal)}")

    duration = len(signal) / sample_rate
    print(f"Length of Audio: {duration:.2f} seconds")
    plot_spectrogram_and_save(signal, sample_rate, Path('img') / 'spectrogram.png')

if __name__ == "__main__":
    main()