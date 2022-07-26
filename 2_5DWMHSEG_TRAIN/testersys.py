import numpy as np

def values_standardized(x):
    x = x.copy()
    x = np.unique(x)
    mmi = np.min(x)
    if mmi >= 0:
        raise ValueError('The input is not z-standardized!')
    
def values_eigthbit(x):
    x = x.copy()
    x = np.unique(x)
    mmi = np.min(x)
    mmj = np.max(x)
    if mmi < 0 and mmj > 255:
        raise ValueError('The input is not 8 bit.')


def labels_binary(x):
    x = x.copy()
    x = np.unique(x)
    if len(x) > 2:
        raise ValueError('The input labels are not binary!')

def correct_fazekas_labels(x):
    x = x.copy()
    x = np.unique(x)
    if -999 in x:
        raise ValueError('The input labels are not correct!')