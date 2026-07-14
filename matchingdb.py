import librosa
import soundfile
from pathlib import Path
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import time
from scipy.signal import correlate

file_path_full = "videoplayback.wav"

file_path_match = "matchingplayback.wav"

def matches_fully(spectrogram_full, spectrogram_match):
    full = np.array(spectrogram_full)
    match = np.array(spectrogram_match)
    
    for (i, j), value in np.ndenumerate(match):
        if (value != full[i][j]):
            return False
        
    return True

def find_all_matches(signal_full, signal_match):
    spectrogram_full = signal_to_db(signal_full)
    spectrogram_match = signal_to_db(signal_match)
    
    full = np.array(spectrogram_full)
    match = np.array(spectrogram_match)
    
    full_w = full.shape[1]
    match_w = match.shape[1]
    
    max_col = full_w - match_w + 1
    
    all_matches = []
    
    for c in range(max_col):
        window = full[:, c : c + match_w]
        
        if np.allclose(window, match, rtol=1e-1):
            all_matches.append(c)
                
    return all_matches

def find_matches_cross_correlation(signal_full, signal_match, threshold_percent=99.99999):
    # Ensure they are numpy arrays
    full = np.array([1, 2, 3, 4, 5, 6])#signal_to_db(signal_full))
    match = np.array([2, 3])#signal_to_db(signal_match))
    
    # 1. Perform 2D Cross-Correlation (Valid mode keeps it within bounds)
    # We correlate along the time axis (columns)
    correlation = correlate(full, match, mode='valid')
    
    # Because full and match have the same height, 'correlation' becomes a 1D array
    # representing how well they match at each column shift.
    correlation_1d = correlation[0] 
    
    # 2. Find the peaks (where the snippet matches best)
    # A perfect match will create a massive peak in the correlation values.
    max_correlation_val = np.max(correlation_1d)
    
    # Define a threshold (e.g., 95% of the absolute maximum match strength)
    # This accounts for tiny noise or rounding differences.
    threshold = max_correlation_val * (threshold_percent / 100.0)
    
    # Find all column indices where the correlation crosses our threshold
    match_indices = np.where(correlation_1d >= threshold)[0]
    
    return match_indices.tolist()

def signal_to_db(signal):
    stft = librosa.stft(signal)
    spectrogram = np.abs(stft)
    spectrogram_db = librosa.amplitude_to_db(spectrogram)
    return spectrogram_db

def signal_to_freq(signal):
    stft = librosa.stft(signal)
    spectrogram = np.abs(stft)
    return spectrogram

def plot_spectrogram_and_save(signal, sample_rate, output_path: Path):
    spectrogram_db = signal_to_db(signal)
    #print(spectrogram_db)
    plt.figure(figsize=(10,4))
    img = librosa.display.specshow(spectrogram_db, 
    y_axis='log', x_axis='time', sr=sample_rate, cmap='inferno')

def main(): 
    start = time.time()
    #sfo = soundfile.SoundFile(file_path_full)
    #signal_full, sample_rate = librosa.load(sfo, sr=None)
    signal_full, sample_rate = librosa.load(file_path_full, sr=None, duration=10.0)
    signal_match, _ = librosa.load(file_path_match, sr=sample_rate)
    end = time.time()

    print(f"File loaded successfully!")
    print(f"Load Time: {(end - start):.2f} seconds")
    print(f"Sample Rate: {sample_rate} Hz")
    print(f"Total Audio Samples: {len(signal_full)}")

    duration_full = len(signal_full) / sample_rate
    duration_match = len(signal_match) / sample_rate


    print(f"Length of Audio: {duration_full:.2f} seconds")
    print(f"Length of Audio: {duration_match:.2f} seconds")

    #all_matches = find_matches_cross_correlation(signal_full, signal_match)
# print(f"Snippet found starting at column(s): {matching_columns}")

    all_matches = find_all_matches(signal_full, signal_match)
    print(all_matches)

    spectrogram_db = signal_to_db(signal_full)
    match_db = signal_to_db(signal_match)

    #plot_spectrogram_and_save(signal_full, sample_rate, Path('img') / 'spectrogram.png')
    print()
    #plot_spectrogram_and_save(signal_match, sample_rate, Path('img2') / 'spectrogram2.png')
    plt.show()

if __name__ == "__main__":
    main()