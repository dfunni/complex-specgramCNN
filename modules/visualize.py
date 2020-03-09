import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fftshift


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