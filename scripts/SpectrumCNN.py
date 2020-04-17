#!/usr/bin/env python3
# coding: utf-8

"""
Spectrum CNN

Automatic modulation recognition with CNN using spectrogram data based on 
technique from Zeng et al's paper "Spectrum Analysis and Convolutiopnal 
Neural Netowrk for Automatic Modulation Recognition" published in IEEE 
Wireless Communications Letters, Vol 8, No 3, of June 2019.

This example deviates from Zeng et al's in tje following ways:

  - Spectrogram data (CNN input) is provided as a 2D (time and freqnency) 
    array of intensity values as apposed to a 3D (time, frequency, and three
    color channels) array. This allows Conv2D layers to be used vs Conv3D. This
    change reduces the number of learned parameters from 199k to 172k.


  - The dataset combines spectrograms from multiple SNRs (0 to 18 dB) vs only
    using a single SNR per model. This is inteded to make the trained model
    more robust to noise by 1) providing more training data, and 2) providing a 
    wider variety of data due to various ammounts of noise present.
"""

import os
import sys
sys.path.append("../modules")

import tensorflow as tf
import tensorflow.keras as keras
import numpy as np
import matplotlib.pyplot as plt
import pickle
import pandas as pd

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.layers import BatchNormalization, Dropout, Activation

import utilities as ut
import preprocess
import visualize as viz


gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        # Currently, memory growth needs to be the same across GPUs
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        # Memory growth must be set before GPUs have been initialized
        print(e)


def build_SCNN(X_train, y_train, fit_params, filename):

    csv_logger = keras.callbacks.CSVLogger(f'../logs/{filename}.log')
    callbacks = [csv_logger]

    model = Sequential()
    model.add(Conv2D(64, kernel_size=(3,3), activation='relu',
                     input_shape=X_train.shape[1:], padding='same'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Conv2D(32, kernel_size=(3,3), activation='relu',
                     padding='same'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Conv2D(12, kernel_size=(3,3), activation='relu',
                     padding='same'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Conv2D(8, kernel_size=(3,3), activation='relu',
                     padding='same'))

    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dense(11, activation='softmax'))

    model.compile(loss='sparse_categorical_crossentropy',
                  optimizer='adam',
                  metrics=['acc'])

    history = model.fit(X_train, y_train, callbacks=callbacks, **fit_params)

    return model, history


def get_data(data_dict, data_params):
    spec_dict = preprocess.process_spec(data_dict, **data_params)

    X_train, y_train, X_test, y_test = preprocess.process_data(spec_dict,
                                                               test_split=0.0,
                                                               blur=False)

    return X_train, y_train, X_test, y_test


###############################################################################
##                                  MAIN                                     ##
###############################################################################
def main(fit_params):
    desc = ['m', 'mp', 'mu', 'iq']

    df_hist = pd.DataFrame()

    ## Generate Dataset
    filename = "../datasets/RML2016_10a_dict.pkl"
    raw_data = ut.load_dataset(filename)

    snrs = range(-20, 20, 2)

    # print(f"\nThere are {data_dict['x'].shape[0]} examples in dataset.")
    # print(f"Examples are complex vectors of length {data_dict['x'].shape[1]}.")
    # print(f"There were {len(snrs)} SNRs used.\n")
    for snr in snrs:

        data_dict = preprocess.select_SNRs(raw_data, [snr])

    ###########################################################################
        ## Get dataset 1
        data_params = {"nperseg": 29,
                       "noverlap": 28,
                       "n_ex": None,
                       "nfft": 100,
                       "inph": 0,
                       "quad": 0,
                       "mag": 1,
                       "ph": 0,
                       "ph_unwrap": 0}

        X_train, y_train, X_test, y_test = get_data(data_dict, data_params)

        print(f"\nRunning model for SNR: {snr} dB.")
        print(f"Data is of type: {desc[0]}")
        print(f"The training dataset of shape: {X_train.shape}\n")

        fn = f'{desc[0]}_{snr}'
        ## Train model 1
        model, history_mag = build_SCNN(X_train, y_train, fit_params, fn)
        #######################################################################
        ## Get dataset 1
        data_params = {"nperseg": 29,
                       "noverlap": 28,
                       "n_ex": None,
                       "nfft": 100,
                       "inph": 0,
                       "quad": 0,
                       "mag": 1,
                       "ph": 1,
                       "ph_unwrap": 0}

        X_train, y_train, X_test, y_test = get_data(data_dict, data_params)

        print(f"\nRunning model for SNR: {snr} dB.")
        print(f"Data is of type: {desc[1]}")
        print(f"The training dataset of shape: {X_train.shape}\n")

        fn = f'{desc[1]}_{snr}'

        ## Train model 2
        model, history_mp = build_SCNN(X_train, y_train, fit_params, fn)      
        #######################################################################
        ## Get dataset 2
        data_params = {"nperseg": 29,
                       "noverlap": 28,
                      "n_ex": None,
                       "nfft": 100,
                       "inph": 0,
                       "quad": 0,
                       "mag": 1,
                       "ph": 0,
                       "ph_unwrap": 1}

        X_train, y_train, X_test, y_test = get_data(data_dict, data_params)

        print(f"\nRunning model for SNR: {snr} dB.")
        print(f"Data is of type: {desc[2]}")
        print(f"The training dataset of shape: {X_train.shape}\n")

        fn = f'{desc[2]}_{snr}'

        ## Train model 2
        model, history_mpu = build_SCNN(X_train, y_train, fit_params, fn)
        #######################################################################
        ## Get dataset 3
        data_params = {"nperseg": 29,
                       "noverlap": 28,
                       "n_ex": None,
                       "nfft": 100,
                       "inph": 1,
                       "quad": 1,
                       "mag": 0,
                       "ph": 0,
                       "ph_unwrap": 0}

        X_train, y_train, X_test, y_test = get_data(data_dict, data_params)

        print(f"\nRunning model for SNR: {snr} dB.")
        print(f"Data is of type: {desc[3]}")
        print(f"The training dataset of shape: {X_train.shape}\n")

        fn = f'{desc[3]}_{snr}'

        ## Train model 2
        model, history_iq = build_SCNN(X_train, y_train, fit_params, fn)
        #######################################################################


if __name__ == "__main__":

    fit_params =  {"batch_size": 32,
                   "epochs": 25,
                   "validation_split": 0.2,
                   "verbose": 1}
    
    if not os.path.isdir('../logs'):
        os.mkdir('../logs')

    main(fit_params)
