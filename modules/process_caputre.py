import numpy as np
from scipy import signal
from scipy.fft import fftshift
import matplotlib.pyplot as plt


## Functions for IQ files recorded by gqrx or similar SDR recievers
def interleave(x_c):
    '''Converts array of complex numbers and converts to an interleaved array 
    of the form:

    [I[0], Q[0], I[1], Q[1], ..., I[n], Q[n]]

    '''

    I = np.real(x_c)
    Q = np.imag(x_c)
    x_iq = np.empty(I.size + Q.size, dtype=np.float32)
    x_iq[0::2] = I
    x_iq[1::2] = Q
    return x_iq


def _normalize(arr):
    '''Maps an array to interval [0, 1]'''

    return (arr - arr.min()) / (arr.max() - arr.min())


def _sigma_thresh(arr, thresh):
    '''thresholds a given array by a lower and upper threshold given in number
    of standard deviations
    inputs:
        arr         input array
        thresh      2 valued tuple or array type uppper and lower threshold
    returns:
        arr         array
    '''
    low_thresh, up_thresh = thresh
    sigma = np.std(arr)
    mx = np.mean(arr)
    arr = np.minimum(arr, mx + up_thresh*sigma)
    arr = np.maximum(mx - low_thresh*sigma, arr)
    return arr


def calc_spectrogram(x_c, fs, thresh, mode, nperseg, noverlap, nfft):
    f, t, sxx = signal.spectrogram(x_c, 
                                   fs, 
                                   nperseg=nperseg, 
                                   noverlap=noverlap, 
                                   nfft=nfft, 
                                   mode=mode, 
                                   return_onesided=False)

    sxx = _sigma_thresh(sxx, thresh)
    sxx = _normalize(sxx)

    return f, t, sxx


def plot_spectrogram(f, t, sxx, center_freq):
    offset = center_freq - f[-1]

    plt.pcolormesh(t, 
                   fftshift(f) + offset, 
                   fftshift((sxx), 
                   axes=0), 
                   cmap='jet')#,
                   # norm=LogNorm(.1, .5))

    plt.ylabel('Frequency [Hz]')
    plt.xlabel('Time [s]')
    plt.show()


def write_to_file(filename, data):
    np.save(filename, data, allow_pickle=False)