import numpy as np

def values_standardized(x):
    x = x.copy()
    x = np.unique(x)
    mmi = np.min(x)
    if mmi >= 0:
        raise ValueError('The input is not z-standardized!')

