#!/usr/bin/env python3

## Basic Packages
import argparse
import os
import sys

## update path to include all relevant files
sys.path.append("modules")
sys.path.append("datasets")
sys.path.append("processed_data")

## packages for pre-processing
import pickle
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from scipy.fft import fftshift

## Packages for deep learning
import tensorflow as tf
import tensorflow.keras as K
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

## Modules
import utilities as ut


def main(dataset, snrs, model_name, epochs):
    ## Pre-process dataset
    raw_data = ut.load_dataset(dataset)
    data_dict = ut.select_SNRs(raw_data, snrs)

    n_ex = data_dict['x'].shape[0]

    print('There are {} examples in dataset.'.format(data_dict['x'].shape[0]))
    print('Each example is a complex vector of length of {}.'.format(data_dict['x'].shape[1]))
    print('There were {} SNRs used.'.format(len(snrs)))

    spec_dict = ut.process_spec(data_dict, nperseg=29, noverlap=28, n_ex=None, nfft=100, 
                                I=True, Q=True, mag=False, ph=False, ph_unwrap=False)


    ## View data
    # n_ex = spec_dict['y'].size
    # for i in range(0, n_ex, 1000):
    #     ut.plot_spectrogram(spec_dict, ex_num=i)

    # for i in range(0, n_ex, 1000):
        # ut.plot_iq(data_dict, ex_num=i)


    ## Implement deep learning
    X = spec_dict['x_s']
    y_labels, y = tf.unique(spec_dict['y'])
    # X = tf.convert_to_tensor(X)
    # y = tf.convert_to_tensor(y)

    model = Sequential()
    model.add(Conv2D(64, kernel_size=(3,3), activation='relu', input_shape=X.shape[1:], padding='same'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Conv2D(32, kernel_size=(3,3), activation='relu', padding='same'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Conv2D(12, kernel_size=(3,3), activation='relu', padding='same'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Conv2D(8, kernel_size=(3,3), activation='relu', padding='same'))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dense(11, activation='softmax'))

    model.compile(loss='sparse_categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

    model.summary()

    hist = model.fit(X, y, batch_size=128, epochs=epochs, validation_split=0.1, verbose=1)

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

    parser.add_argument('--epochs', type=int, help='number of epochs to train.',
                        default=10)

    args = parser.parse_args()

    dataset = 'datasets/' + args.dataset
    main(dataset, [args.snrs], args.model_filename, args.epochs)



