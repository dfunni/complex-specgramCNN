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


import sys
sys.path.append("../modules")

import tensorflow as tf
import tensorflow.keras as K
import numpy as np
import matplotlib.pyplot as plt
import pickle

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.layers import BatchNormalization, Dropout, Activation

import utilities as ut
import preprocess
import visualize as viz

# Constants
EPOCHS = 20
BATCH_SIZE = 32


def build_SCNN(X_train):
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

    return model


###############################################################################
##                                  MAIN                                     ##
###############################################################################
def main():
    ## Generate Dataset
    filename = "../datasets/RML2016_10a_dict.pkl"
    raw_data = ut.load_dataset(filename)

    snrs = [2, 6, 10, 14, 18]

    # print(f"\nThere are {data_dict['x'].shape[0]} examples in dataset.")
    # print(f"Examples are complex vectors of length {data_dict['x'].shape[1]}.")
    # print(f"There were {len(snrs)} SNRs used.\n")
    for snr in snrs:

        data_dict = preprocess.select_SNRs(raw_data, [snr])

    ###########################################################################
        ## Get dataset 1
        desc = 'mag'

        spec_dict = preprocess.process_spec(data_dict, nperseg=29, noverlap=28,
                                            n_ex=None, nfft=100,
                                            inph=0, quad=0, 
                                            mag=1, ph=0, ph_unwrap=0)

        X_train, y_train, X_test, y_test = preprocess.process_data(spec_dict,
                                                                   test_split=0.0,
                                                                   blur=False)
        print(f"\nRunning model for SNR: {snr} dB.")
        print(f"The training dataset is of shape: {X_train.shape}\n")

        ## Train model 1
        model = build_SCNN(X_train)
        model.summary()
        history_mag = model.fit(X_train, y_train, batch_size=BATCH_SIZE, 
                                epochs=EPOCHS, validation_split=0.2, verbose=1)

        

        plt.plot(history_mag.history['acc'], ls=':', label=f'acc {desc}')
        plt.plot(history_mag.history['val_acc'],
                 label=f'val_acc {desc} {snr}')
        plt.title('Model accuracy')
        plt.ylabel('Accuracy')
        plt.xlabel('Epoch')
        plt.ylim(0,1)
        plt.legend()

        ###########################################################################
        ## Get dataset 2
        desc = 'mag + phase'

        spec_dict = preprocess.process_spec(data_dict, nperseg=29, noverlap=28,
                                            n_ex=None, nfft=100,
                                            inph=0, quad=0,
                                            mag=1, ph=1, ph_unwrap=0)

        X_train, y_train, X_test, y_test = preprocess.process_data(spec_dict,
                                                                   test_split=0.0,
                                                                   blur=False)

        print(f"\nThe training dataset is of shape: {X_train.shape}")

        ## Train model 2
        model = build_SCNN(X_train)
        model.summary()
        history_magph = model.fit(X_train, y_train, batch_size=BATCH_SIZE,
                                  epochs=EPOCHS, validation_split=0.2, verbose=1)

        

        # plt.plot(history_magph.history['acc'], ls=':', label=f'acc {desc}')
        plt.plot(history_magph.history['val_acc'],
                 label=f'val_acc {desc} {snr}')
        plt.title('Model accuracy')
        plt.ylabel('Accuracy')
        plt.xlabel('Epoch')
        plt.ylim(0,1)
        plt.legend()

        ###########################################################################
    
    plt.show()


if __name__ == "__main__":
    main()
