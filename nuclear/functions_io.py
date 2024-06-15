"""
I/O functions.

2024-06-07
"""
import numpy as np


def read_coords(N):
    if N <= 150:
        fpath = f'cluster_coords/LJ150/{N}.TXT'
    elif N <= 309:
        fpath = f'cluster_coords/LJ151-309/{N}.TXT'
    elif N <= 561:
        fpath = f'cluster_coords/LJ310-561/{N}.TXT'
    else:
        print(f'Warning: missing file for {N} particles')
        return -1

    X = np.loadtxt(fpath)
    return X