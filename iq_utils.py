import pickle
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal
from sklearn.utils import shuffle
import cv2
import tensorflow as tf

def load_data(filename):
    data = pickle.load(open(filename, 'rb'), encoding='latin1')
    return data


def to_pickle(spec_dict, filename):
    # Save data to pickle format
    if filename[-4:] != '.pkl':
        filename = filename + '.pkl'

    with open(filename, 'wb') as handle:
        pickle.dump(spec_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)


def from_pickle(filename):
    # Open capture data from pickle format
    if filename[-4:] != '.pkl':
        filename = filename + '.pkl'
        
    with open(filename, 'rb') as handle:
        spec_dict = pickle.load(handle)
        
    return spec_dict


def split_dict(tdt_dict):
    key = [key for key in tdt_dict.keys()]
    x = tdt_dict[key[0]]
    y = tdt_dict[key[1]]
    snr = tdt_dict[key[2]]

    return x, y, snr


def get_mods_and_snrs(raw_dict):
    mods = sorted(set([key[0] for key in raw_dict.keys()]))
    snrs = sorted(set([key[1] for key in raw_dict.keys()]))
    return mods, snrs


def make_data_dict(raw_dict, snrs=None):
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


def iq_to_spec(data_dict, nperseg, noverlap, n_ex=None, nfft=None):
    
    if n_ex == None:
        n_ex = data_dict['y'].size

    if nfft == None:
        n_f = nperseg
    else:
        n_f = nfft

    n_t = int((128-nperseg)/(nperseg-noverlap) + 1)
    
    x_s = np.zeros((n_f * n_ex, n_t))
    y = []
    snr = []

    for i in range(n_ex):
        x = data_dict['x'][i]
        f, t, sxx = scipy.signal.spectrogram(x,
                                             window='hann',
                                             nperseg=nperseg,
                                             noverlap=noverlap,
                                             nfft=nfft,
                                             mode='psd',
                                             return_onesided=True)
        
        x_s[i*n_f : n_f*(i + 1), :] = sxx
        y.append(data_dict['y'][i])
        snr.append(data_dict['snr'][i])

    x_s = x_s.reshape(n_ex, n_f, n_t, 1)
    y = np.array(y)
    snr = np.array(snr)

    x_s = normalize_spectrogram(x_s)

    spec_dict = {'x_s': x_s, 'y': y, 'snr': snr}
                
    return spec_dict


def train_test_split(spec_dict, test_split=0.1):
    # test_split is the proprtion of examples to test on

    shuffle_dict = shuffle_data(spec_dict)

    n_ex = shuffle_dict['y'].size

    itrain = int((1-test_split) * n_ex)

    x_train = shuffle_dict['x_s'][:itrain]
    y_train = shuffle_dict['y'][:itrain]
    snr_train = shuffle_dict['snr'][:itrain]

    x_test = shuffle_dict['x_s'][itrain:]
    y_test = shuffle_dict['y'][itrain:]
    snr_test = shuffle_dict['snr'][itrain:]

    train_dict = {'x_train': x_train, 'y_train': y_train, 'snr_train': snr_train}
    test_dict = {'x_test': x_test, 'y_test': y_test, 'snr_test': snr_test}

    return train_dict, test_dict


def process_data(spec_dict, test_split=0, blur=False):

    if blur:
        spec_dict = blur_spec(spec_dict, ksize=(7,7))

    train_dict, test_dict = train_test_split(spec_dict, test_split)
    X_train, y_train, snr_train = split_dict(train_dict)
    X_test, y_test, snr_test = split_dict(test_dict)
    y_train_labels, y_train = tf.unique(y_train)
    y_test_labels, y_test = tf.unique(y_test)

    return X_train, y_train, X_test, y_test


def process_data_3c(spec_dict, test_split=0, blur=False):

    if blur:
        spec_dict = blur_spec_3c(spec_dict, ksize=(7,7))

    train_dict, test_dict = train_test_split(spec_dict, test_split)
    X_train, y_train, snr_train = split_dict(train_dict)
    X_test, y_test, snr_test = split_dict(test_dict)
    y_train_labels, y_train = tf.unique(y_train)
    y_test_labels, y_test = tf.unique(y_test)

    return X_train, y_train, X_test, y_test


def plot_spectrogram(spec_dict, ex_num):
    x_s = spec_dict['x_s'][ex_num]
    y = spec_dict['y'][ex_num]
    snr = spec_dict['snr'][ex_num]

    x_s = np.reshape(x_s, (x_s.shape[0], x_s.shape[1]))
    plt.title('Waveform: ' + y + '    SNR: ' + str(snr))
    plt.pcolormesh(10 * np.log10(x_s))
    plt.show()


def plot_iq(data_dict, ex_num):
    x = data_dict['x'][ex_num]
    y = data_dict['y'][ex_num]
    snr = data_dict['snr'][ex_num]

    plt.plot(np.real(x))
    plt.plot(np.imag(x))
    plt.title('Waveform: ' + y + '    SNR: ' + str(snr))
    plt.show()


def normalize_spectrogram(x_s):
    return (x_s-np.min(x_s)) / (np.max(x_s)-np.min(x_s))


def shuffle_data(spec_dict):
    key = [key for key in spec_dict.keys()]

    x_s = spec_dict[key[0]]
    y = spec_dict[key[1]]
    snr = spec_dict[key[2]]

    x_s, y, snr = shuffle(x_s, y, snr)
    shuffle_dict = {'x_s': x_s, 'y': y, 'snr': snr}
    return shuffle_dict


def blur_spec(spec_dict, ksize=(7,7)):
    specs = spec_dict['x_s'][:,:,:,0]
    n_ex, n_f, n_t = specs.shape[:3]
    for i in range(n_ex):
        specs[i] = cv2.GaussianBlur(specs[i], ksize=ksize, sigmaX=0, sigmaY=0)

    blur_dict['x_s'] = specs.reshape(n_ex, n_f, n_t, 1)

    return blur_dict
