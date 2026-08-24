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

file_path_full = "bernsteinfull.wav"

file_path_match = "academyhorns5.wav"

measures_to_time = dict()

SAMPLE_RATE = 44100

HOP_LENGTH = 512

def signal_to_db(signal):
    stft = librosa.stft(signal)
    spectrogram = np.abs(stft)
    spectrogram_db = librosa.amplitude_to_db(spectrogram)
    return spectrogram_db

def signal_to_ampl(signal):
    stft = librosa.stft(signal)
    spectrogram = np.abs(stft)
    return spectrogram

def plot_spectrogram_and_save(signal, sample_rate, output_path: Path):
    spectrogram_ampl = signal_to_ampl(signal)
    plt.figure(figsize=(10,4))
    img = librosa.display.specshow(spectrogram_ampl, 
    y_axis='log', x_axis='time', sr=sample_rate, cmap='inferno')
    plt.hist(spectrogram_ampl)

def plot_values_histogram(spectrogram_ampl):
    plt.hist(spectrogram_ampl.ravel(), bins=100, range=(1, spectrogram_ampl.max()))
    plt.show()


def cosine_similarity(window, match):
    sums = np.sum(window * match, axis=1, keepdims=True)
    window_norm = np.linalg.norm(window, axis=1, keepdims=True)
    match_norm = np.linalg.norm(match, axis=1, keepdims=True)
    denominator = window_norm * match_norm

    output = np.zeros_like(sums)
    np.divide(sums, denominator, out=output, where=denominator != 0)

    ampltiude_factor_1 = np.zeros_like(sums)
    np.divide(window_norm, match_norm, out=ampltiude_factor_1, where=match_norm != 0)
    ampltiude_factor_2 = np.zeros_like(sums)
    np.divide(match_norm, window_norm, out=ampltiude_factor_2, where=window_norm != 0)
    ampltiude_factor = np.minimum(ampltiude_factor_1, ampltiude_factor_2)

    return output * ampltiude_factor


def correlate(full, match):
    full_w = full.shape[1]
    match_w = match.shape[1]
    
    max_col = full_w - match_w + 1
    
    all_windows = []
    
    for c in range(max_col):
        window = full[:, c : c + match_w]
        all_windows.append(cosine_similarity(window, match))
                
    return np.concat(all_windows, axis=1)

def similarity_significance(window, match, cosine_similarity):
    # librosa.display.specshow(window, y_axis='log', x_axis='time', sr=SAMPLE_RATE, cmap='inferno')
    # plt.show()
    # librosa.display.specshow(match, y_axis='log', x_axis='time', sr=SAMPLE_RATE, cmap='inferno')
    # plt.show()
    
    window_max = np.max(window, axis=1, keepdims=True)
    match_max = np.max(match, axis=1, keepdims=True)

    SIGNIFICANCE_LIMIT = 5
    window_max = np.clip(window_max, a_max=SIGNIFICANCE_LIMIT, a_min=0)
    match_max = np.clip(match_max, a_max=SIGNIFICANCE_LIMIT, a_min=0)
    # librosa.display.specshow(np.repeat(window_max, window.shape[1], axis=1), y_axis='log', x_axis='time', sr=SAMPLE_RATE, cmap='inferno')
    # plt.show()
    # librosa.display.specshow(np.repeat(match_max, window.shape[1], axis=1), y_axis='log', x_axis='time', sr=SAMPLE_RATE, cmap='inferno')
    # plt.show()

    avg_importance = (window_max + match_max) / 2.0
    total_importance = np.sum(avg_importance)
    weighted_arr = cosine_similarity * avg_importance
    total_weight = np.sum(weighted_arr)
    weight = total_weight / total_importance
    
    return weight

def weighing(full, match, cosine_similarity):
    full_w = full.shape[1]
    match_w = match.shape[1]

    max_col = full_w - match_w + 1
        
    all_windows = []

    for c in range(max_col):
        window = full[:, c : c + match_w]
        all_windows.append(similarity_significance(window, match, cosine_similarity[:, c : c + 1]))

    return all_windows

def top_k_non_adjacent(arr, k, min_dist=1):
    """
    Finds the top k values and indices that are at least `min_dist` apart.
    """
    # Create a working copy to modify and mask out values
    similarity = np.array(arr)
    working_arr = similarity.astype(float, copy=True)
    
    selected_indices = []
    selected_values = []
    
    for _ in range(k):
        # 1. Find the index of the current maximum value
        max_idx = np.nanargmax(working_arr)
        
        # Break early if all remaining elements have been masked out
        if np.isnan(working_arr[max_idx]):
            break
            
        # 2. Save the valid maximum value and its original index
        selected_indices.append(max_idx)
        selected_values.append(arr[max_idx])
        
        # 3. Determine the masking window bounds
        start = max(0, max_idx - min_dist)
        end = min(len(arr), max_idx + min_dist + 1)
        
        # 4. Mask the chosen element and its neighbors with NaN
        working_arr[start:end] = np.nan
        
    return np.array(selected_values), np.array(selected_indices)

def give_ten_potential_answer(similarity):
    similarity = np.array(similarity)
    top_10_sample = np.argpartition(similarity, -10)[-10:]
    top_10_idx = top_10_sample[np.argsort(similarity[top_10_sample])[::-1]]
    top_10_seconds = sample_position_to_seconds(top_10_idx)

    with np.printoptions(formatter={'float': lambda x: f"{x:.2f} seconds"}):
        print(np.array(top_10_seconds))

def sample_position_to_seconds(position):
    return (position * HOP_LENGTH / SAMPLE_RATE)
    
def play_potential_answer(timeframe, match_signal, full_signal):
    signal_cropped = full_signal[timeframe * HOP_LENGTH : timeframe * HOP_LENGTH + len(match_signal)]
    output_path = "potentialanswer.wav"

    sf.write(output_path, signal_cropped, SAMPLE_RATE)
    print(f"Audio successfully saved to {output_path}")    

def main(): 
    start = time.time()

    full_signal, _ = librosa.load(file_path_full, sr=SAMPLE_RATE)
    match_signal, _ = librosa.load(file_path_match, sr=SAMPLE_RATE)
    #, offset=260, duration=5

    end = time.time()
    print(f"File loaded successfully!")
    print(f"Load Time: {(end - start):.2f} seconds")
    print(f"Sample Rate: {SAMPLE_RATE} Hz")
    print(f"Total Audio Samples: {len(full_signal)}")
    duration_full = len(full_signal) / SAMPLE_RATE
    duration_match = len(match_signal) / SAMPLE_RATE
    print(f"Length of Audio: {duration_full:.2f} seconds")
    print(f"Length of Audio: {duration_match:.2f} seconds")

    #best_matches = find_best_matches(full_signal, match_signal)

    #plot_values_histogram(signal_to_ampl(full_signal))
    #print()
    #plot_values_histogram(signal_to_ampl(match_signal))
    #plt.show()

    #similarity_df = pd.DataFrame(weighted_similarity)
    #similarity_df.to_excel('similarity435.xlsx', index=False)

    full_ampl = signal_to_ampl(full_signal)
    match_ampl = signal_to_ampl(match_signal)

    cosine_sim = correlate(full_ampl, match_ampl)


    weighted_similarity = weighing(full_ampl, match_ampl, cosine_sim)
    idx = int(np.argmax(weighted_similarity))
    play_potential_answer(idx, match_signal, full_signal)

    give_ten_potential_answer(weighted_similarity)
    top_similarity, top_indicies = top_k_non_adjacent(weighted_similarity, 20, min_dist=200)

    for idx, (index, similarity) in enumerate(zip(top_indicies, top_similarity)):
        print(f"Rank {idx + 1}: Index = {index}, Similarity = {similarity:.4f}, Time = {sample_position_to_seconds(index):.2f} seconds")


if __name__ == "__main__":
    main()