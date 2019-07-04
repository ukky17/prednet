import os
import argparse

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"]="2"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import numpy as np
from six.moves import cPickle
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
    return data_format, test_model

def convert_to_ratio(_X_hat):
    _X_hat_ratio = np.zeros_like(_X_hat)
    for i in range(_X_hat.shape[1]):
        _X_hat_ratio[:, i, ::] = _X_hat[:, i, ::] / _X_hat[:, 0, ::]
    return _X_hat_ratio

if __name__ == '__main__':
    # params
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_movies', type=int, default=10)
    parser.add_argument('--pre_frames', type=int, default=10)
    parser.add_argument('--stim_frames', type=int, default=20)
    parser.add_argument('--post_frames', type=int, default=20)

    parser.add_argument('--model_type', type=str, default='Lall',
                        help='L0, or Lall')
    parser.add_argument('--target', type=str, default='E0')
    parser.add_argument('--stim', type=str, default='MAE_P_deg0',
                        help='MAE_P_deg0, MAE_P_deg180 or OF_R_out, OF_R_in')

    args = parser.parse_args()
    n_movies = args.n_movies
    pre_frames = args.pre_frames
    stim_frames = args.stim_frames
    post_frames = args.post_frames
    model_type = args.model_type
    target = args.target
    stim = args.stim
    print(args)

    nt = pre_frames + stim_frames + post_frames

    # get the result path
    RESULTS_SAVE_DIR = './response_' + str(nt) + 'frames_' + model_type + '/'

    # get the model path
    WEIGHTS_DIR = './model_' + str(nt) + 'frames/'
    json_file = os.path.join(WEIGHTS_DIR, 'prednet_kitti_model.json')
    weights_file = WEIGHTS_DIR + 'prednet_kitti_weights.hdf5'

    # get the data path
    DATA_DIR = 'stim/stims_' + str(nt) + 'frames/'
    test_file = DATA_DIR + stim + '.hkl'
    test_sources = DATA_DIR + stim + '_source.hkl'

    # model
    data_format, test_model = create_test_model(json_file, weights_file, target)

    # data
    test_generator = SequenceGenerator(test_file, test_sources, nt,
                                       sequence_start_mode='unique',
                                       data_format=data_format)
    X_test = test_generator.create_all()

    # create permuted data
    X_test_permuted = np.zeros_like(X_test)
    p = np.random.permutation(X_test.shape[0] * X_test.shape[1])
    p = p.reshape(X_test.shape[:2])
    for i in range(X_test.shape[0]):
        for j in range(X_test.shape[1]):
            _p = p[i, j]
            X_test_permuted[i, j, ::] = X_test[_p // X_test.shape[1],
                                               _p % X_test.shape[1], ::]

    # predict
    X_hat = test_model.predict(X_test, n_movies)
    X_hat_permuted = test_model.predict(X_test_permuted, n_movies)
    if data_format == 'channels_first':
        X_test = np.transpose(X_test, (0, 1, 3, 4, 2))
        X_test_permuted = np.transpose(X_test_permuted, (0, 1, 3, 4, 2))
        X_hat = np.transpose(X_hat, (0, 1, 3, 4, 2))
        X_hat_permuted = np.transpose(X_hat_permuted, (0, 1, 3, 4, 2))

    tmp1 = np.mean(X_hat, axis=(2, 3))[:, 1:, :] # (n_movies, nt-1, n_neurons)
    tmp2 = np.mean(X_hat_permuted, axis=(2, 3))[:, 1:, :]

    # tmp1 = convert_to_ratio(tmp1)
    # tmp2 = convert_to_ratio(tmp2)

    if target[0] == 'R':
        n_neurons = tmp1.shape[-1]
        n_plot = int(np.ceil(n_neurons / 100))
        for n in range(n_plot):
            fig = plt.figure(figsize=(20, 20))

            for i in range(min(100, n_neurons - n * 100)):
                idx = 100 * n + i
                ax = plt.subplot(10, 10, i+1)
                ax.errorbar(range(1, nt), np.mean(tmp1[:, :, idx], axis=0),
                            yerr=np.std(tmp1[:, :, idx], axis=0) / np.sqrt(n_movies))
                ax.errorbar(range(1, nt), np.mean(tmp2[:, :, idx], axis=0),
                            yerr=np.std(tmp2[:, :, idx], axis=0) / np.sqrt(n_movies))
                # ax.set_xlabel('Frames')
                # ax.set_ylabel('Response')
                ax.axvspan(pre_frames, pre_frames + stim_frames - 0.5,
                            facecolor='r', alpha=0.3)
            plt.tight_layout()
            plt.savefig(RESULTS_SAVE_DIR + stim + '_' + target + '_' + str(n) + '.png')
            plt.savefig(RESULTS_SAVE_DIR + stim + '_' + target + '_' + str(n) + '.pdf')

    elif target[0] == 'E':
        fig = plt.figure()
        plt.errorbar(range(1, nt),
                     np.mean(tmp1, axis=(0, 2)),
                     yerr=np.std(tmp1, axis=(0, 2)) / \
                          np.sqrt(tmp1.shape[0] * tmp1.shape[2]))
        plt.errorbar(range(1, nt),
                     np.mean(tmp2, axis=(0, 2)),
                     yerr=np.std(tmp2, axis=(0, 2)) / \
                          np.sqrt(tmp2.shape[0] * tmp2.shape[2]))
        plt.xlabel('Frames')
        plt.ylabel('Response')
        plt.legend(['Stim', 'Permuted'])
        plt.axvspan(pre_frames, pre_frames + stim_frames - 0.5,
                    facecolor='r', alpha=0.3)
        plt.tight_layout()
        plt.savefig(RESULTS_SAVE_DIR + stim + '_' + target + '.png')
        plt.savefig(RESULTS_SAVE_DIR + stim + '_' + target + '.pdf')
