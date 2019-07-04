import os
import argparse

import numpy as np
from six.moves import cPickle
import hickle

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

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

pre_frames = 10
stim_frames = 20
post_frames = 20
model_type = 'Lall'
_stim = 'MAE_P_'

nt = pre_frames + stim_frames + post_frames

if __name__ == '__main__':
    SAVE_DIR = './response/' + str(nt) + 'frames_' + model_type + '/'

    for target in ['E0', 'E1', 'E2', 'E3', 'R0', 'R1', 'R2', 'R3']:
        resp_deg0 = hickle.load(SAVE_DIR + 'MAE_P_deg0_' + target + '.hkl')
        resp_deg0_p = hickle.load(SAVE_DIR + 'MAE_P_deg0_' + target + '_permuted.hkl')
        resp_deg180 = hickle.load(SAVE_DIR + 'MAE_P_deg180_' + target + '.hkl')
        resp_deg180_p = hickle.load(SAVE_DIR + 'MAE_P_deg180_' + target + '_permuted.hkl')

        resp_deg0 = np.mean(resp_deg0, axis=(2, 3))[:, 1:, :] # (n_movies, nt-1, n_neurons)
        resp_deg0_p = np.mean(resp_deg0_p, axis=(2, 3))[:, 1:, :]
        resp_deg180 = np.mean(resp_deg180, axis=(2, 3))[:, 1:, :] # (n_movies, nt-1, n_neurons)
        resp_deg180_p = np.mean(resp_deg180_p, axis=(2, 3))[:, 1:, :]

        if target[0] == 'R':
            mpl.rcParams['ytick.labelsize'] = 9

            n_neurons = resp_deg0.shape[-1]
            n_plot = int(np.ceil(n_neurons / 100))
            for n in range(n_plot):
                fig = plt.figure(figsize=(20, 20))

                for i in range(min(100, n_neurons - n * 100)):
                    s = np.sqrt(len(resp_deg0))
                    idx = 100 * n + i
                    ax = plt.subplot(10, 10, i+1)
                    ax.errorbar(range(1, nt), np.mean(resp_deg0[:, :, idx], axis=0),
                                yerr=np.std(resp_deg0[:, :, idx], axis=0) / s,
                                c='m')
                    ax.errorbar(range(1, nt), np.mean(resp_deg180[:, :, idx], axis=0),
                                yerr=np.std(resp_deg180[:, :, idx], axis=0) / s,
                                c='c')
                    # ax.set_xlabel('Frames')
                    # ax.set_ylabel('Response')
                    plt.setp(ax.get_xticklabels(), visible=False)
                    ax.set_xticks(np.arange(0, 60, 10))
                    ax.axvspan(pre_frames, pre_frames + stim_frames - 0.5,
                                facecolor='r', alpha=0.3)
                plt.tight_layout()
                plt.savefig(SAVE_DIR + _stim + target + '_' + str(n) + '.png')
                plt.savefig(SAVE_DIR + _stim + target + '_' + str(n) + '.pdf')

        elif target[0] == 'E':
            s = np.sqrt(resp_deg0.shape[0] * resp_deg0.shape[2])
            fig = plt.figure()
            plt.errorbar(range(1, nt), np.mean(resp_deg0, axis=(0, 2)),
                         yerr=np.std(resp_deg0, axis=(0, 2)) / s,
                         c='m')
            plt.errorbar(range(1, nt), np.mean(resp_deg0_p, axis=(0, 2)),
                         yerr=np.std(resp_deg0_p, axis=(0, 2)) / s,
                         c='lightpink')
            plt.errorbar(range(1, nt), np.mean(resp_deg180, axis=(0, 2)),
                         yerr=np.std(resp_deg180, axis=(0, 2)) / s,
                         c='c')
            plt.errorbar(range(1, nt), np.mean(resp_deg180_p, axis=(0, 2)),
                         yerr=np.std(resp_deg180_p, axis=(0, 2)) / s,
                         c='lightskyblue')
            plt.xlabel('Frames')
            plt.ylabel('Response')
            plt.legend(['Stim 0deg', 'Permuted 0deg', 'Stim 180deg', 'Permuted 180deg'], fontsize=10)
            plt.axvspan(pre_frames, pre_frames + stim_frames - 0.5,
                        facecolor='r', alpha=0.3)
            plt.tight_layout()
            plt.savefig(SAVE_DIR + _stim + target + '.png')
            plt.savefig(SAVE_DIR + _stim + target + '.pdf')
