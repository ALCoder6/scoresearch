# import pyaudio

# # Define audio properties
# CHUNK = 1024
# FORMAT = pyaudio.paInt16
# CHANNELS = 1
# RATE = 44100

# p = pyaudio.PyAudio()

# # Open the live microphone stream
# stream = p.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK)

# print("Streaming live mic data. Press Ctrl+C to stop.")

# try:
#     while True:
#         # Read raw binary data from the microphone buffer
#         data = stream.read(CHUNK)
#         # Print the first few bytes of the live byte stream
#         print(data[:10]) 
# except KeyboardInterrupt:
#     print("\nStreaming stopped.")
#     stream.stop_stream()
#     stream.close()
#     p.terminate()


import librosa
import soundfile as sf
from pathlib import Path
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import time
import pandas as pd
from scipy.io import wavfile
import argparse
import queue
import sys
import numpy as np
import sounddevice as sd

# Thread-safe queue to pass audio blocks from background thread to main loop
q = queue.Queue()



file_path_full = "bernsteinfull.wav"

SAMPLE_RATE = 44100

HOP_LENGTH = 512

def signal_to_ampl(signal):
    stft = librosa.stft(signal)
    spectrogram = np.abs(stft)
    return spectrogram

full_signal, _ = librosa.load(file_path_full, sr=SAMPLE_RATE)
full_ampl = signal_to_ampl(full_signal)

def audio_callback(indata, frames, time, status):
    """Callback function called by sounddevice for each new audio block."""
    if status:
        print(status, file=sys.stderr)
    # Put a copy of the incoming audio block (NumPy ndarray) into the queue
    q.put(indata.copy())


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
    window_max = np.max(window, axis=1, keepdims=True)
    match_max = np.max(match, axis=1, keepdims=True)
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


def sample_position_to_seconds(position):
    return (position * HOP_LENGTH / SAMPLE_RATE)

def seconds_to_readable_time(seconds):
    minutes = int(seconds // 60)
    remaining_seconds = seconds % 60
    if minutes > 0:
        return f"{minutes}m {remaining_seconds:.2f}s"
    else:
        return f"{remaining_seconds:.2f}s"

def top_k_non_adjacent(arr, k, min_dist=100):
    similarity = np.array(arr)
    working_arr = similarity.astype(float, copy=True)
    
    selected_indices = []
    selected_values = []
    
    for _ in range(k):
        max_idx = np.nanargmax(working_arr)
        
        if np.isnan(working_arr[max_idx]):
            break
            
        selected_indices.append(max_idx)
        selected_values.append(arr[max_idx])
        
        start = max(0, max_idx - min_dist)
        end = min(len(arr), max_idx + min_dist + 1)
        
        working_arr[start:end] = np.nan
        
    return np.array(selected_values), np.array(selected_indices)


def process_signal(signal_block):
    """
    Your custom real-time signal processing function.
    `signal_block` is a 2D NumPy array with shape (frames, channels).
    """

    match_ampl = signal_to_ampl(signal_block)

    cosine_sim = correlate(full_ampl, match_ampl)

    weighted_similarity = weighing(full_ampl, match_ampl, cosine_sim)

    top_similarity, top_indicies = top_k_non_adjacent(weighted_similarity, 10)
    for idx, (index, similarity) in enumerate(zip(top_indicies, top_similarity)):
        match_location_seconds = sample_position_to_seconds(index)
        readable_time = seconds_to_readable_time(match_location_seconds)
        print(f"Rank {idx + 1}: Index = {index}, Similarity = {similarity:.4f}, Time = {readable_time} seconds")
    
    

    # Example 1: Calculate RMS (Volume / Energy)
    #rms = np.sqrt(np.mean(signal_block**2))
    
    # Example 2: Peak Amplitude
    #peak = np.max(np.abs(signal_block))

    # Output signal stats (Replace this block with your custom DSP / FFT / model logic)
    #print(f"RMS Energy: {rms:.4f} | Peak Amplitude: {peak:.4f}", end="\r")

def main():
    parser = argparse.ArgumentParser(description="Live Microphone Signal Capture")
    parser.add_argument('-d', '--device', type=int, help='input device ID')
    parser.add_argument('-c', '--channels', type=int, default=1, help='number of channels')
    parser.add_argument('-r', '--samplerate', type=float, default=44100, help='sampling rate (Hz)')
    parser.add_argument('-b', '--blocksize', type=int, default=5, help='block size in seconds')
    args = parser.parse_args()

    print("Starting live audio capture... Press Ctrl+C to stop.\n")

    try:
        # Open live audio stream
        stream = sd.InputStream(
            device=args.device,
            channels=args.channels,
            samplerate=args.samplerate,
            blocksize=args.blocksize * args.samplerate,
            callback=audio_callback
        )
        
        with stream:
            while True:
                # Retrieve incoming signal block from queue
                print("Getting audio...")
                signal_block = q.get()
                signal_block = signal_block.flatten()  # Flatten to 1D if needed

                print(f"Processing capture...")
                
                # Perform live signal operations
                process_signal(signal_block)

    except KeyboardInterrupt:
        print("\nStopped live audio capture.")
    except Exception as e:
        print(f"\nError: {e}")

if __name__ == "__main__":
    main()