from skimage.transform import resize
import numpy as np


def normalise(data):
    data = np.array(data, dtype=np.float32)
    data -= np.median(data)
    data /= np.std(data)
    return data

def freq_decimate(data, channels=256):
    fdata = resize(data, (channels, np.max(data.shape), 1), anti_aliasing=None)
    return fdata

def decimate(data, factor=1):
    time_axis = np.argmax(data.shape)
    freq_axis = data.shape.index(256)
    dtdata = data.reshape(data.shape[freq_axis], (data.shape[time_axis] // factor), factor).mean(2)
    return dtdata

