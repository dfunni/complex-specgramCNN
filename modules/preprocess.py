import numpy as np
from sklearn.utils import shuffle
from scipy import signal
import cv2
import tensorflow as tf


def _get_mods_and_snrs(raw_dict):
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

    if snrs is None:
        mods, snrs = _get_mods_and_snrs(raw_dict)
    else:
        mods, _ = _get_mods_and_snrs(raw_dict)

    for mod in mods:
        for power in snrs:
            vals = raw_dict[(mod, power)]
            vals = vals[:, 0] + 1j*vals[:, 1]
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
        x_s[:,:,:,i] = (x_s[:,:,:,i]-np.min(x_s[:,:,:,i])) / (np.max(x_s[:,:,:,i])-np.min(x_s[:,:,:,i]))
    return x_s


def process_spec(data_dict, nperseg, noverlap, n_ex=None, nfft=None,
                 inph=False, quad=False, mag=True, ph=True, ph_unwrap=False):
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
    nchan = int(inph) + int(quad) + int(mag) + int(ph) + int(ph_unwrap)

    # number of examples in dataset used
    if n_ex is None:
        n_ex = data_dict['y'].size

    # nfft handler
    if nfft is None:
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
    if inph:
        spec_types.append('inph')
    if quad:
        spec_types.append('quad')
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
        if inph:
            x_s[j*n_f: n_f*(j + 1), :, ch] = np.real(sxx)
            ch += 1
        if quad:
            x_s[j*n_f: n_f*(j + 1), :, ch] = np.imag(sxx)
            ch += 1
        if mag:
            x_s[j*n_f: n_f*(j + 1), :, ch] = np.abs(sxx)
            ch += 1
        if ph:
            x_s[j*n_f: n_f*(j + 1), :, ch] = np.angle(sxx)
            ch += 1
        if ph_unwrap:
            x_s[j*n_f: n_f*(j + 1), :, ch] = np.unwrap(np.angle(sxx))
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


def _shuffle_data(spec_dict):
    key = [key for key in spec_dict.keys()]

    x_s = spec_dict[key[0]]
    y = spec_dict[key[1]]
    snr = spec_dict[key[2]]

    x_s, y, snr = shuffle(x_s, y, snr)
    shuffle_dict = {'x_s': x_s, 'y': y, 'snr': snr}
    return shuffle_dict


def _train_test_split(spec_dict, test_split=0.1):
    # test_split is the proprtion of examples to test on

    shuffle_dict = _shuffle_data(spec_dict)

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


def _split_dict(tdt_dict):
    key = [key for key in tdt_dict.keys()]
    x = tdt_dict[key[0]]
    y = tdt_dict[key[1]]
    snr = tdt_dict[key[2]]

    return x, y, snr


def blur_spec(spec_dict, ksize=(7, 7)):
    specs = spec_dict['x_s'][:,:,:,0]
    n_ex, n_f, n_t = specs.shape[:3]
    for i in range(n_ex):
        specs[i] = cv2.GaussianBlur(specs[i], ksize=ksize, sigmaX=0, sigmaY=0)

    spec_dict['x_s'] = specs.reshape(n_ex, n_f, n_t, 1)

    return spec_dict


def process_data(spec_dict, test_split=0, blur=False):

    if blur:
        spec_dict = blur_spec(spec_dict, ksize=(7,7))

    train_dict, test_dict = _train_test_split(spec_dict, test_split)
    X_train, y_train, snr_train = _split_dict(train_dict)
    X_test, y_test, snr_test = _split_dict(test_dict)
    y_train_labels, y_train = tf.unique(y_train)
    y_test_labels, y_test = tf.unique(y_test)

    return X_train, y_train, X_test, y_test
