import pickle


def load_dataset(filename):
    # Loads 2016.10a dasaset which uses latin1 encoding
    data = pickle.load(open(filename, 'rb'), encoding='latin1')
    return data


def to_pickle(sample_dict, filename):
    # Save data to pickle format
    if filename[-4:] != '.pkl':
        filename = filename + '.pkl'

    with open(filename, 'wb') as handle:
        pickle.dump(sample_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)


def from_pickle(filename):
    # loads processed data from pickle format
    if filename[-4:] != '.pkl':
        filename = filename + '.pkl'

    with open(filename, 'rb') as handle:
        sample_dict = pickle.load(handle)

    return sample_dict
