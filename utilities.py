import pickle
import numpy as np
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
import cv2
import tensorflow as tf

from scipy import signal
from scipy.fft import fftshift


ef load_data(filename):
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


def _get_mods_and_snrs(raw_dict):
    mods = sorted(set([key[0] for key in raw_dict.keys()]))
    snrs = sorted(set([key[1] for key in raw_dict.keys()]))
    return mods, snrs


def make_data_dict(raw_dict, snrs=None):
    if snrs == None:
        mods, snrs = _get_mods_and_snrs(raw_dict)
    else:
        mods, _ = _get_mods_and_snrs(raw_dict)

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