import pickle
import numpy as np
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
import cv2
import tensorflow as tf

from scipy import signal
from scipy.fft import fftshift


## Functions on dictionaries
def load_dataset(filename):
    # Loads 2016.10a dasaset which uses latin1 encoding
    data = pickle.load(open(filename, 'rb'), encoding='latin1')
    return data


def to_pickle(sample_dict, filename):
    # Save data to pickle format
    if filename[-4:] != '.pkl':
        filename = filename + '.pkl'

    with open(filename, 'wb') as handle:
        pickle.dump(sample_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)


def from_pickle(filename):
    # loads processed data from pickle format
    if filename[-4:] != '.pkl':
        filename = filename + '.pkl'
        
    with open(filename, 'rb') as handle:
        spec_dict = pickle.load(handle)
        
    return sample_dict


def get_mods_and_snrs(raw_dict):
    # creates lists of modulations and SNRs in dataset
    mods = sorted(set([key[0] for key in raw_dict.keys()]))
    snrs = sorted(set([key[1] for key in raw_dict.keys()]))
    return mods, snrs


def select_SNRs(raw_dict, snrs=None):
    ''' Makes dict of selected SNRs from initial dataset 
    
    Arguments:
       raw_dict: raw dataset dict, eg 2016.10a
       snrs:     list of SNRs to use for futher processing
    
    data output with the following keys:
       x:     complex valued samples
       y:     ground truth modulation type
       snr:   SNR level
    '''

    if snrs == None:
        mods, snrs = get_mods_and_snrs(raw_dict)
    else:
        mods, _ = get_mods_and_snrs(raw_dict)

    for mod in mods:
        for power in snrs:
            vals = raw_dict[(mod, power)]
            vals = vals[:,0] + 1j*vals[:,1]
            try:
                x = np.append(x, vals, axis=0)
                y = np.append(y, np.repeat(mod, vals.shape[0]))
                snr = np.append(snr, np.repeat(power, vals.shape[0]))
            except NameError:
                x = vals
                y = np.repeat(mod, vals.shape[0])
                snr = np.repeat(power, vals.shape[0])
    
    data_dict = {'x': x, 'y': y, 'snr': snr}
    return data_dict


def normalize_spectrogram(x_s):
    # normalizes spectrogram between 0 and 1
    # used in 2 channel iq_to_spec

    num_spec_type = x_s.shape[-1]
    for i in range(num_spec_type):
        x_s[:,:,:,i] =  (x_s[:,:,:,i]-np.min(x_s[:,:,:,i])) / (np.max(x_s[:,:,:,i])-np.min(x_s[:,:,:,i]))
    return x_s


def process_spec(data_dict, nperseg, noverlap, n_ex=None, nfft=None, 
                    I=False, Q=False, mag=True, ph=True, ph_unwrap=False):
    ''' Takes processed dict containing complex valued samples, ground truth
    modulation types, and SNR levels and converts examples to complex 
    spectrograms.

    Arguments:
        data_dict: {
            x: complex samples,
            y: modulations,
            snr: SNRs}
        nperseg: window length
        noverlap: window overlap
        n_ex: number of examples to output
        nfft: length of fft

    Returns:
        spec_dict: {
            x_s: complex spectrograms,
            y: modulations,
            snr: SNRs,
            t: time labels for spectrograms,
            f: freqency labels for spectrograms}
    '''

    # Determine number of channels for the spectrogram output
    nchan = int(I) + int(Q) + int(mag) + int(ph) + int(ph_unwrap)

    # number of examples in dataset used
    if n_ex == None:
        n_ex = data_dict['y'].size

    # nfft handler
    if nfft == None:
        n_f = nperseg
    else:
        n_f = nfft

    # time axis length
    n_t = int((128-nperseg)/(nperseg-noverlap) + 1)
    
    # initialize outputs
    x_s = np.zeros((n_f * n_ex, n_t, nchan))
    y = []
    snr = []
    spec_types = []

    # make spec_type list for plotting purposes
    if I:
        spec_types.append('I')
    if Q:
        spec_types.append('Q')
    if mag:
        spec_types.append('mag')
    if ph:
        spec_types.append('ph')
    if ph_unwrap:
        spec_types.append('ph_unwrap')

    # iterate over each example calculating spectrogram and output values
    for j in range(n_ex):
        x = data_dict['x'][j]
        f, t, sxx = signal.spectrogram(x,
                                       window='hann',
                                       nperseg=nperseg,
                                       noverlap=noverlap,
                                       nfft=nfft,
                                       mode='complex',
                                       return_onesided=False)

        # build spectrogram output (x_s) with selected spectrogram types
        # stacked into image-like channels
        ch = 0
        if I:
            x_s[j*n_f : n_f*(j + 1), :, ch] = np.real(sxx)
            ch += 1
        if Q:
            x_s[j*n_f : n_f*(j + 1), :, ch] = np.imag(sxx)
            ch += 1
        if mag:
            x_s[j*n_f : n_f*(j + 1), :, ch] = np.abs(sxx)
            ch += 1
        if ph:
            x_s[j*n_f : n_f*(j + 1), :, ch] = np.angle(sxx)
            ch += 1
        if ph_unwrap:
            x_s[j*n_f : n_f*(j + 1), :, ch] = np.unwrap(np.angle(sxx))
            ch += 1

        # save other output values for the example
        y.append(data_dict['y'][j])
        snr.append(data_dict['snr'][j])

    # modify outputs CNN implementation
    x_s = x_s.reshape(n_ex, n_f, n_t, nchan)
    x_s = normalize_spectrogram(x_s)
    y = np.array(y)
    snr = np.array(snr)
    
    # build spectrogram dictionary
    spec_dict = {'x_s': x_s, 
                 'y': y, 
                 'snr': snr, 
                 't': t, 
                 'f': f,
                 'types': spec_types}
                
    return spec_dict



def plot_spectrogram(spec_dict, ex_num):
    # plots spectrograms given example number and dictionary of spectrograms
    
    # spectrogram axes
    t = spec_dict['t']
    f = spec_dict['f']

    # modulation type
    y = spec_dict['y'][ex_num]

    # SNR value
    snr = spec_dict['snr'][ex_num]

    # list of specrogram types in dict
    spec_types = spec_dict['types']
    ntypes = len(spec_types)

    # fig, axs = plt.subplots(1, ntypes)

    # plot each spectrogram type for a given example number
    for j in range(ntypes):

        sxx = spec_dict['x_s'][ex_num,:,:,j]

        plt.subplot(1, ntypes, j+1)
        sxx = np.reshape(sxx, (sxx.shape[0], sxx.shape[1]))
        plt.title(str(spec_types[j]) + '\nWaveform: ' + y + '    SNR: ' + str(snr))
        plt.pcolormesh(t, 
                       fftshift(f), 
                       fftshift((sxx), 
                       axes=0), 
                       cmap='jet')
    
    plt.show()


def plot_iq(data_dict, ex_num):
    x = data_dict['x'][ex_num]
    y = data_dict['y'][ex_num]
    snr = data_dict['snr'][ex_num]

    plt.plot(np.real(x))
    plt.plot(np.imag(x))
    plt.title('Waveform: ' + y + '    SNR: ' + str(snr))
    plt.show()


## Functions for IQ files recorded by gqrx or similar SDR recievers
# def interleave(x_c):
#     I = np.real(x_c)
#     Q = np.imag(x_c)
#     x_iq = np.empty(I.size + Q.size, dtype=np.float32)
#     x_iq[0::2] = I
#     x_iq[1::2] = Q
#     return x_iq


# def _normalize(arr):
#     return (arr - arr.min()) / (arr.max() - arr.min())


# def _sigma_thresh(arr, thresh):
#     low_thresh, up_thresh = thresh
#     sigma = np.std(arr)
#     mx = np.mean(arr)
#     arr = np.minimum(arr, mx + up_thresh*sigma)
#     arr = np.maximum(mx - low_thresh*sigma, arr)
#     return arr


# def calc_spectrogram(x_c, fs, thresh, mode, nperseg, noverlap, nfft):
#     f, t, sxx = signal.spectrogram(x_c, 
#                                    fs, 
#                                    nperseg=nperseg, 
#                                    noverlap=noverlap, 
#                                    nfft=nfft, 
#                                    mode=mode, 
#                                    return_onesided=False)

# #     sxx = _sigma_thresh(sxx, thresh)
# #     sxx = _normalize(sxx)

#     return f, t, sxx


# def plot_spectrogram(f, t, sxx, center_freq):
#     offset = center_freq - f[-1]

#     plt.pcolormesh(t, 
#                    fftshift(f) + offset, 
#                    fftshift((sxx), 
#                    axes=0), 
#                    cmap='jet')#,
#                    # norm=LogNorm(.1, .5))

#     plt.ylabel('Frequency [Hz]')
#     plt.xlabel('Time [s]')
#     plt.show()


# def write_to_file(filename, data):
#     np.save(filename, data, allow_pickle=False)

