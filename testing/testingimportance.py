import librosa
import soundfile
from pathlib import Path
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import time

similarity = np.array([[90],
                       [80]])

full = np.array([[20, 25, 28, 2], 
                 [20, 30, 26, 28]])

match = np.array([[30, 2, 2, 9], 
                 [20, 3, 20, 20]])

full_max = np.max(full, axis=1, keepdims=True)
match_max = np.max(match, axis=1, keepdims=True)
avg_importance = (full_max + match_max) / 2.0
total_importance = np.sum(avg_importance)
weighted_arr = similarity * avg_importance
total_weight = np.sum(weighted_arr)
weight = total_weight / total_importance
print(weight)