# complex-specgramCNN

Automatic modulation recognition is an area of increasing interest for applications such as cognitive radio and interference detection and classification. Past efforts have included numerous deep learning methodologies such as deep neural networks [1], long short term memory [1-2],  and convolutional neural networks (CNN) [1-3]. Of these methods, CNNs have shown great promise with relatively low complexity. Specifically, the approach of Zeng et al [3] reports high accuracy using the in-phase and quadrature (IQ) signal components to compute magnitude spectrograms as input to a convolutional neural network. 

This project builds on the work of [3] by investigating complex representations of spectrograms. Examples of in-phase, quadrature, magnitude, and phase angle spectrograms can be viewd in the signal_view_testbed notebook. We use the 2016.10a dataset created by O’Shea and West [4] which contains 1000 examples of each of 11 different modulation techniques and 20 signal to noise ratios (SNR) from -20 to 18 dBm. Each example consists of 128 IQ values of the modulated signal. After calculating the complex spectrogram representations of each signal, we stack the two-dimensional spectrograms along a third axis creating a dual-channel image allowing for simple implementation in popular deep learning frameworks. The SCNN architecture developed by Zeng et al is then implemented with our complex spectrogram representations as input. For comparison, we present results obtained using dual-channel spectrograms of in-phase/quadrature and magnitude/phase angle in addition to each single-channel spectrogram type. Additionally, the effect of unwrapping phase angle is considered. Using our methodology, preliminary results of dual-channel complex spectrograms show significantly higher validation accuracy with 15 percent fewer parameters than [3].


## Setup
Datasets used can be found on https://www.deepsig.io/datasets. These were originally created by O'Shea and West as discussed in [4]. Download the 2016.10a dataset and move into a datasets/ directory in the main directory for compatability with the process_data notebook.

## References

1. N. E. West and T. O’Shea, “Deep architectures for modulation recognition,” in 2017 IEEE International Symposium on Dynamic Spectrum Access Networks (DySPAN), 2017, pp. 1–6, doi: 10.1109/DySPAN.2017.7920754.

2. M. Zhang, Y. Zeng, Z. Han, and Y. Gong, “Automatic Modulation Recognition Using Deep Learning Architectures,” in 2018 IEEE 19th International Workshop on Signal Processing Advances in Wireless Communications (SPAWC), 2018, pp. 1–5, doi: 10.1109/SPAWC.2018.8446021.

3. Y. Zeng, M. Zhang, F. Han, Y. Gong, and J. Zhang, “Spectrum Analysis and Convolutional Neural Network for Automatic Modulation Recognition,” IEEE Wireless Communications Letters, vol. 8, no. 3, pp. 929–932, Jun. 2019, doi: 10.1109/LWC.2019.2900247.

4. T. J. O’Shea and N. West, “Radio Machine Learning Dataset Generation with GNU Radio,” Proceedings of the GNU Radio Conference; Vol 1 No 1 (2016): Proceedings of the 6th GNU Radio Conference, Sep. 2016.
