import numpy as np
from sklearn.utils import shuffle
from scipy import signal
import cv2
import tensorflow as tf


def _get_mods_and_snrs(raw_dict):
    """creates lists of modulations and SNRs in dataset"""

    mods = sorted(set([key[0] for key in raw_dict.keys()]))
    snrs = sorted(set([key[1] for key in raw_dict.keys()]))
    return mods, snrs


def select_SNRs(raw_dict, snrs=None):
    """Makes dict of selected SNRs from initial dataset
            
    Args:
        raw_dict (dict): raw dataset dict, eg 2016.10a
        snrs (list): SNRs to use for futher processing
            (default is None)

    Returns:
        data_dict (dict): data dict containing IQ data of type:
            {x: complex samples,
             y: modulations,
             snr: SNRs}
    """

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
    """maps spectrogram to [0,1], used in 2 channel iq_to_spec
    
    Args:
        x_s (np.array): spectrogram array (of type i, q, m, p, or u)

    Returns:
        x_s (np.array): normalized spectrogram array
    """

    num_spec_type = x_s.shape[-1]
    for i in range(num_spec_type):
        x_i = x_s[...,i]
        x_s[...,i] = (x_i-x_i.min()) / (x_i.max()-x_i.min())
    return x_s


def process_spec(data_dict, nperseg, noverlap, n_ex=None, nfft=None,
                 i=False, q=False, m=True, p=True, u=False):
    """Converts processed dict to complex spectrograms.

    Args:
        data_dict (dict): data dict containing IQ data of type:
            {x: complex samples,
             y: modulations,
             snr: SNRs}
        nperseg (int): window length
        noverlap (int): window overlap
        n_ex (int): number of examples to output
        nfft (int): length of fft
        i (bool): in-phase (default is False)
        q (bool): quadrature (default is False)
        m (bool): magnitude (default is True)
        p (bool): phase (default is True)
        u (bool): unwrapped phase (aka angle) (default is False)

    Returns:
        spec_dict (dict): dict with spectrogram data of form:
            {x_s: complex spectrograms,
             y: modulations,
             snr: SNRs,
             t: time labels for spectrograms,
             f: freqency labels for spectrograms}
    """
    
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
    
    n_t = int((128-nperseg)/(nperseg-noverlap) + 1)  # time axis length

    # initialize outputs
    x_s = np.zeros((n_f * n_ex, n_t, nchan))
    y = []
    snr = []
    spec_types = []

    # make spec_type list for plotting purposes
    if inph:
        spec_types.append('i')
    if quad:
        spec_types.append('q')
    if mag:
        spec_types.append('m')
    if ph:
        spec_types.append('p')
    if ph_unwrap:
        spec_types.append('u')

    # iterate over each example calculating spect and output values
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
    """shuffles data in spectrogram dictionary
    
    Args:
        spec_dict (dict): dict with spectrogram data of form:
            {x_s: complex spectrograms,
             y: modulations,
             snr: SNRs,
             t: time labels for spectrograms,
             f: freqency labels for spectrograms}

    Returns:
        shuffle_dict (dict): shuffled spectrogram data of form:
            {x_s: complex spectrograms,
             y: modulations,
             snr: SNRs}

    NOTE: time and frequency labels are removed from output dictionary
    NOTE: this function is handled by tensorflow and is unessary
    """

    key = [key for key in spec_dict.keys()]

    x_s = spec_dict[key[0]]
    y = spec_dict[key[1]]
    snr = spec_dict[key[2]]

    x_s, y, snr = shuffle(x_s, y, snr)
    shuffle_dict = {'x_s': x_s, 'y': y, 'snr': snr}
    return shuffle_dict


def _train_test_split(spec_dict, test_split=0.1):
    """splits training and testing set
    
    Args:
        spec_dict (dict): dict with spectrogram data of form:
            {x_s: complex spectrograms,
             y: modulations,
             snr: SNRs,
             t: time labels for spectrograms,
             f: freqency labels for spectrograms}
        test_split (float): portion of data to be used for testing [0,1]
            (default is 0.1)

    Returns:
        train_dict (dict): training data in dictionary of form:
            {x_s: complex spectrograms,
             y: modulations,
             snr: SNRs}
        test_dict (dict): testing data in dictionary of form:
            {x_s: complex spectrograms,
             y: modulations,
             snr: SNRs}
             

    NOTE: time and frequency labels are removed from output dictionary
    NOTE: this function is handled by tensorflow and is unessary
    """

    shuffle_dict = _shuffle_data(spec_dict)

    n_ex = shuffle_dict['y'].size

    itrain = int((1-test_split) * n_ex)

    x_train = shuffle_dict['x_s'][:itrain]
    y_train = shuffle_dict['y'][:itrain]
    snr_train = shuffle_dict['snr'][:itrain]

    x_test = shuffle_dict['x_s'][itrain:]
    y_test = shuffle_dict['y'][itrain:]
    snr_test = shuffle_dict['snr'][itrain:]

    train_dict = {'x_train': x_train,
                  'y_train': y_train,
                  'snr_train': snr_train}
    test_dict = {'x_test': x_test,
                 'y_test': y_test,
                  'snr_test': snr_test}

    return train_dict, test_dict


def _split_dict(tdt_dict):
    """extracts data from dictionary form
    
    Args:
        tdt_dict (dict): dict with data of form:
            {x_s: complex spectrograms,
             y: modulations,
             snr: SNRs,
             t: time labels for spectrograms,
             f: freqency labels for spectrograms}
        test_split (float): portion of data to be used for testing [0,1]

    Returns:
        x (np.array): data to be classified
        y (np.array): ground truth values from dataset
        snr (np.array): SNR lables of the data
    """

    key = [key for key in tdt_dict.keys()]
    x = tdt_dict[key[0]]
    y = tdt_dict[key[1]]
    snr = tdt_dict[key[2]]

    return x, y, snr


def blur_spec(spec_dict, ksize=(7, 7)):
    """blurs spectrogram as discussed by Zeng et al 2019
    
    Args:
        spec_dict (dict): dict with spectrogram data of form:
            {x_s: complex spectrograms,
             y: modulations,
             snr: SNRs,
             t: time labels for spectrograms,
             f: freqency labels for spectrograms}
        ksize (int, int): kernel size of Gaussian blur, must be tuple of 
            odd numbers
            (default is (7, 7))

    Returns:
        spec_dict (dict): dict with spectrogram data of form:
            {x_s: complex spectrograms,
             y: modulations,
             snr: SNRs,
             t: time labels for spectrograms,
             f: freqency labels for spectrograms}
    """

    specs = spec_dict['x_s'][:,:,:,0]
    n_ex, n_f, n_t = specs.shape[:3]
    for i in range(n_ex):
        specs[i] = cv2.GaussianBlur(specs[i], ksize=ksize, sigmaX=0, sigmaY=0)

    spec_dict['x_s'] = specs.reshape(n_ex, n_f, n_t, 1)

    return spec_dict


def process_data(spec_dict, test_split=0, blur=False):
    """blurs spectrogram as discussed by Zeng et al 2019
    
    Args:
        spec_dict (dict): dict with spectrogram data of form:
            {x_s: complex spectrograms,
             y: modulations,
             snr: SNRs,
             t: time labels for spectrograms,
             f: freqency labels for spectrograms}
        test_split (float): portion of data to be used for testing [0,1]
            (default is 0)
        blur (bool): flag for Gaussian blur
            (default is False)

    Returns:
        spec_dict (dict): dict with spectrogram data of form:
            {x_s: complex spectrograms,
             y: modulations,
             snr: SNRs,
             t: time labels for spectrograms,
             f: freqency labels for spectrograms}
    """

    if blur:
        spec_dict = blur_spec(spec_dict, ksize=(7,7))

    train_dict, test_dict = _train_test_split(spec_dict, test_split)
    X_train, y_train, snr_train = _split_dict(train_dict)
    X_test, y_test, snr_test = _split_dict(test_dict)
    y_train_labels, y_train = tf.unique(y_train)
    y_test_labels, y_test = tf.unique(y_test)

    return X_train, y_train, X_test, y_test
