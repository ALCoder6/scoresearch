import numpy as np
from scipy.signal import correlate

    # Ensure they are numpy arrays
full = np.array([[1, 2, 3, 4, 5, 6], [0, 2, 3, 5, 3, 2]])#signal_to_db(signal_full))
match = np.array([[2, 3], [2, 3]])#signal_to_db(signal_match))

def cosine_similarity_1d(v, w):
    dot_prod = np.dot(v, w) / (np.linalg.norm(v) * np.linalg.norm(w))
    return dot_prod

def cosine_similarity(window, match):
    return np.sum(window * match, axis=1, keepdims=True) / (np.linalg.norm(window, axis=1, keepdims=True) * np.linalg.norm(match, axis=1, keepdims=True))
    # window norm : iterating over columns, along the row

    # |------ all one frequency over time
    # |
    # |
    # | for each window

    #array of arrays
    #outer array is for each frequency, each element is array of cosine similarities for each time stamp
    #frequency x cosine similarity
    

def correlate(full, match):
    full_w = full.shape[1]
    match_w = match.shape[1]
    
    max_col = full_w - match_w + 1
    
    all_windows = []
    
    for c in range(max_col):
        window = full[:, c : c + match_w]
        all_windows.append(cosine_similarity(window, match))
                
    return np.concat(all_windows, axis=1)

match_indices = cosine_similarity(full[ : , 1 : 3], match)



print(correlate(full, match))

