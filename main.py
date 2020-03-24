#!/usr/bin/env python3

''' Basic Packages'''
import argparse
# import os
import sys
import re

''' Packages for pre-processing'''
import pickle
# import numpy as np
import matplotlib.pyplot as plt
# from scipy import signal
# from scipy.fft import fftshift

''' Packages for deep learning'''
# import tensorflow as tf
# import tensorflow.keras as K
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

''' Modules: first update path to include all relevant files'''
sys.path.append("modules")
sys.path.append("datasets")
sys.path.append("processed_data")
import utilities as ut
import preprocess


def view_data(plot_dict, step):
    n_ex = plot_dict['y'].size
    for i in range(0, n_ex, step):
        ut.plot_spectrogram(plot_dict, ex_num=i)


def main(dataset, snrs, model_name, epochs, I, Q, mag, ph, ph_unwrap):
    raw_data = ut.load_dataset(dataset)

    ''' Pre-process dataset'''
    data_dict = preprocess.select_SNRs(raw_data, snrs)
    n_ex = data_dict['x'].shape[0]
    l_ex = data_dict['x'].shape[1]
    print(f'There are {n_ex} examples in dataset.')
    print(f'Each example is a complex vector of length of {l_ex}.')
    print(f'There were {len(snrs)} SNRs used.')
    spec_dict = preprocess.process_spec(data_dict,
                                        nperseg=29,
                                        noverlap=28,
                                        n_ex=None,
                                        nfft=100,
                                        inph=I, quad=Q,
                                        mag=mag, ph=ph,
                                        ph_unwrap=ph_unwrap)

    # view_data(spec_dict, step=1000)

    ''' Implement deep learning'''
    X_train, y_train, X_test, y_test = preprocess.process_data(spec_dict,
                                                               test_split=0.0,
                                                               blur=False)
    print("Input shape: {}".format(X_train.shape))

    model = Sequential()
    model.add(Conv2D(64, kernel_size=(3, 3),
                     activation='relu',
                     input_shape=X_train.shape[1:],
                     padding='same'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(32, kernel_size=(3, 3),
                     activation='relu', padding='same'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(12, kernel_size=(3, 3),
                     activation='relu', padding='same'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(8, kernel_size=(3, 3),
                     activation='relu', padding='same'))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dense(11, activation='softmax'))

    model.compile(loss='sparse_categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

    model.summary()

    hist = model.fit(X_train, y_train, batch_size=128,
                     epochs=epochs, validation_split=0.1, verbose=1)

    plt.plot(hist.history['accuracy'])
    plt.plot(hist.history['val_accuracy'])
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.ylim(0,1)
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.show()

    model.save(model_name)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset', type=str,
                        help='str, default = "RML2016_10a_dict.pkl"',
                        default="RML2016_10a_dict.pkl")

    parser.add_argument('--snrs', nargs='+', type=int,
                        help='''list of desired SNRs, must be even integers between
                        -20 and 18''',
                        default='18')

    parser.add_argument('--model_filename', type=str,
                        help='str, eg. "filename.model"',
                        default='cSCNN.model')

    parser.add_argument('--epochs', type=int,
                        help='number of epochs to train.',
                        default=10)

    parser.add_argument('-o', '--channel_options', type=str,
                        help='''options for channels:
                        i = in-phase
                        q = quadrature
                        m = magnitude
                        p = phase angle
                        u = unwrapped phase angle
                        ex: -o iqmpu    selecs all types of channels.''',
                        default='mp')

    args = parser.parse_args()

    ch_opts = list(args.channel_options)
    ch_i = any([bool(re.search('i', c, re.IGNORECASE)) for c in ch_opts])
    ch_q = any([bool(re.search('q', c, re.IGNORECASE)) for c in ch_opts])
    ch_m = any([bool(re.search('m', c, re.IGNORECASE)) for c in ch_opts])
    ch_p = any([bool(re.search('p', c, re.IGNORECASE)) for c in ch_opts])
    ch_u = any([bool(re.search('u', c, re.IGNORECASE)) for c in ch_opts])

    dataset = 'datasets/' + args.dataset
    main(dataset, list(args.snrs), args.model_filename, args.epochs,
         inph=ch_i, quad=ch_q, mag=ch_m, ph=ch_p, ph_unwrap=ch_u)
