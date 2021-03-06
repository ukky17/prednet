import os
import argparse

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"]="3"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import numpy as np
from six.moves import cPickle
import hickle
from tqdm import tqdm
import gc

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from keras import backend as K
from keras.models import Model, model_from_json
from keras.layers import Input, Dense, Flatten

from prednet import PredNet
from data_utils import SequenceGenerator

import matplotlib as mpl
mpl.rcParams['figure.figsize'] = [8.0, 6.0]
mpl.rcParams['font.size'] = 20
mpl.rcParams['xtick.labelsize'] = 16
mpl.rcParams['ytick.labelsize'] = 16
mpl.rcParams['font.sans-serif'] = 'Arial'
mpl.rcParams['axes.spines.top'] = False
mpl.rcParams['axes.spines.right'] = False
plt.rcParams['ps.useafm'] = True
plt.rcParams['pdf.use14corefonts'] = True

def create_test_model(json_file, weights_file, target):
    # Load trained model
    f = open(json_file, 'r')
    json_string = f.read()
    f.close()
    train_model = model_from_json(json_string, custom_objects={'PredNet': PredNet})
    train_model.load_weights(weights_file)

    # Create testing model (to output predictions)
    layer_config = train_model.layers[1].get_config()
    layer_config['output_mode'] = target
    data_format = layer_config['data_format'] if 'data_format' \
                               in layer_config else layer_config['dim_ordering']
    test_prednet = PredNet(weights=train_model.layers[1].get_weights(),
                           **layer_config)
    input_shape = list(train_model.layers[0].batch_input_shape[1:])
    input_shape[0] = nt
    inputs = Input(shape=tuple(input_shape))
    predictions = test_prednet(inputs)
    test_model = Model(inputs=inputs, outputs=predictions)
    return input_shape, data_format, test_model

def convert_to_ratio(_X_hat):
    _X_hat_ratio = np.zeros_like(_X_hat)
    for i in range(_X_hat.shape[1]):
        _X_hat_ratio[:, i, ::] = _X_hat[:, i, ::] / _X_hat[:, 0, ::]
    return _X_hat_ratio

if __name__ == '__main__':
    # params
    parser = argparse.ArgumentParser()
    parser.add_argument('--nt', type=int, default=50)
    parser.add_argument('--target', type=str)
    parser.add_argument('--stim', type=str)
    parser.add_argument('--SAVE_DIR', type=str)
    parser.add_argument('--WEIGHTS_DIR', type=str)

    args = parser.parse_args()
    nt = args.nt
    target = args.target
    stim = args.stim
    SAVE_DIR = args.SAVE_DIR
    WEIGHTS_DIR = args.WEIGHTS_DIR
    print(args)

    if not os.path.exists(SAVE_DIR):
        os.makedirs(SAVE_DIR)

    # get the model path
    # K.set_learning_phase(0) # for bn
    json_file = os.path.join(WEIGHTS_DIR, 'prednet_kitti_model.json')
    weights_file = WEIGHTS_DIR + 'prednet_kitti_weights.hdf5'

    # model
    input_shape, data_format, test_model = create_test_model(json_file, weights_file, target)

    # get the data path
    DATA_DIR = 'stim/' + str(nt) + 'frames'
    if input_shape[1] == 128 and input_shape[2] == 160:
        DATA_DIR += '/'
    else:
        DATA_DIR += str(input_shape[1]) + 'x' + str(input_shape[2]) + '_1000trials/'

    # data
    n_movies = 1000
    batch_size = 1

    for n in tqdm(range(n_movies // batch_size)):
        test_file =    DATA_DIR + stim + '_' + str(n) + '.hkl'
        test_sources = DATA_DIR + stim + '_' + str(n) + '_source.hkl'

        # load the movie
        test_generator = SequenceGenerator(test_file, test_sources, nt,
                                           sequence_start_mode='unique',
                                           data_format=data_format,
                                           batch_size=batch_size)
        X_test = test_generator.create_all()

        # feed
        X_hat = test_model.predict(X_test)
        if data_format == 'channels_first':
            X_hat = np.transpose(X_hat, (0, 1, 3, 4, 2))

        """
        # create the permuted movie
        X_test_permuted = np.zeros_like(X_test)
        p = np.random.permutation(X_test.shape[0] * X_test.shape[1])
        p = p.reshape(X_test.shape[:2])
        for i in range(X_test.shape[0]):
            for j in range(X_test.shape[1]):
                _p = p[i, j]
                X_test_permuted[i, j, ::] = X_test[_p // X_test.shape[1],
                                                   _p % X_test.shape[1], ::]

        X_hat_permuted = test_model.predict(X_test_permuted)
        if data_format == 'channels_first':
            X_hat_permuted = np.transpose(X_hat_permuted, (0, 1, 3, 4, 2))
        """

        del X_test
        # del X_test_permuted
        gc.collect()

        hickle.dump(X_hat, SAVE_DIR + stim + '_' + target + '_' + str(n) + '.hkl',
                    mode='w')
        # hickle.dump(X_hat_permuted, SAVE_DIR + stim + '_' + target + '_' + \
        #             str(n) + '_permuted.hkl', mode='w')
