import librosa
import soundfile
from pathlib import Path
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import time

full = np.array([[20, 25, 28, 29], [20, 24, 26, 28]])

arr = np.where(full < 25, 0, full)

print(arr)