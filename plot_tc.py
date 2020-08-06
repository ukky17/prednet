import os

import numpy as np
from scipy import stats
import hickle
from tqdm import tqdm

import matplotlib
import matplotlib.pyplot as plt

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

np.random.seed(0)

def classify(resp):
    """
    resp: (2, n_movies, nt, n_units)
    """

    nt2 = stim_frames + post_frames

    # responsiveness
    resp_mean = np.mean(resp, axis=1) # (2, nt, n_units)
    resp_std = np.std(resp, axis=1) # (2, nt, n_units)

    base_mean = np.mean(resp_mean[:, pre_frames // 2: pre_frames, :], axis=1, keepdims=True) # (2, 1, n_units)
    base_std =  np.mean(resp_std[:,  pre_frames // 2: pre_frames, :], axis=1, keepdims=True) # (2, 1, n_units)

    d_resp = np.abs(resp_mean[:, pre_frames:, :] - base_mean)
    d_resp = np.where(d_resp > (resp_std[:, pre_frames:, :] + base_std) * 0.5, d_resp, 0) # (2, 180, n_units)

    # non-responsive cell
    c = np.sum(d_resp != 0, axis=1) < nt2 * 0.1 # (2, n_units)
    flat_idx = np.where(np.sum(c, axis=0) == n_stims)[0]
    return set(flat_idx.tolist())

def is_flat_old(_resp):
    """
    _resp: (2, n_movies, nt)
    """

    nt2 = stim_frames + post_frames

    # responsiveness
    resp_mean = np.mean(_resp, axis=1) # (2, nt)
    resp_std = np.std(_resp, axis=1) # (2, nt)

    base_mean = np.mean(resp_mean[:, pre_frames // 2: pre_frames], axis=1, keepdims=True) # (2, 1)
    base_std =  np.mean(resp_std[:,  pre_frames // 2: pre_frames], axis=1, keepdims=True) # (2, 1)

    d_resp = np.abs(resp_mean[:, pre_frames:] - base_mean)
    d_resp = np.where(d_resp > (resp_std[:, pre_frames:] + base_std) * 0.5, d_resp, 0) # (2, 180)

    # non-responsive cell
    c = np.sum(d_resp != 0, axis=1) < nt2 * 0.1 # (2, n_units)
    return np.sum(c, axis=0) == n_stims

def is_flat(_resp):
    """
    _resp: (2, n_movies, nt)
    check flatness of the timecourse by `mean > SD`
    """

    # responsiveness
    resp_mean = np.mean(_resp[:, :, pre_frames:], axis=1) # (2, nt2)
    resp_std = np.std(_resp[:, :, pre_frames:], axis=1) # (2, nt2)
    return np.sum(np.abs(resp_mean) > resp_std) < (stim_frames + post_frames) * n_stims * 0.1

def is_flat2(_resp, corr_n=2128896):
    """
    _resp: (2, n_movies, nt)
    check flatness of the timecourse by `p-value < 0.05`

    total units:
    (E1, E2, A1, A2, Ahat1, Ahat2)
    1032192 + 32256 + 516096 + 16128 + 516096 + 16128 = 2128896
    """

    s, p = stats.ttest_1samp(_resp[:, :, pre_frames:], 0, axis=1) # p: (2, nt2)

    _resp_mean = np.mean(_resp[:, :, pre_frames:], axis=1) # (2, nt2)

    # correct p: one-way and Bonferroni
    p_corrected = p * 0.5 * corr_n # (2, nt2)

    for i in range(n_stims):
        for j in range(stim_frames + post_frames):
            if np.isnan(p_corrected[i, j]):
                if _resp_mean[i, j] == 0:
                    p_corrected[i, j] = 1
                else:
                    p_corrected[i, j] = 0

    return np.sum(p_corrected < 0.05) < 1


############################### parameters #####################################

pre_frames = 10 # 50
stim_frames = 20 # 80
post_frames = 20 # 100

n_movies = 1000
batch_size = 1
n_stims = 2
n_cells_to_plot = 100

SAVE_DIR = './response/200806_1/'

# targets = ['E0', 'E1', 'E2', 'R0', 'R1', 'R2', 'A0', 'A1', 'A2', 'Ahat0', 'Ahat1', 'Ahat2']
# targets = ['E0', 'E1', 'E2', 'A0', 'A1', 'A2', 'Ahat0', 'Ahat1', 'Ahat2']
targets = ['E1', 'E2', 'A1', 'A2', 'Ahat1', 'Ahat2']
# targets = ['E2', 'A2', 'Ahat2', 'R2']
# targets = ['E0', 'A0', 'Ahat0']
# taregets = ['E1', 'A1', 'Ahat1']
# targets = ['E2']

for target in targets:
    # load all data
    for n in tqdm(range(n_movies // batch_size)):
        resp_deg0 = hickle.load(SAVE_DIR + 'MAE_P_deg0_' + target + '_' + str(n) + '.hkl') # (batch_size, 230, 12, 14, 192)
        resp_deg180 = hickle.load(SAVE_DIR + 'MAE_P_deg180_' + target + '_' + str(n) + '.hkl')

        if n == 0:
            n_units = resp_deg0.shape[2] * resp_deg0.shape[3] * resp_deg0.shape[4]
            nt = resp_deg0.shape[1]

            # randomly sample 10%
            n_units2 = n_units // 10
            idxs = np.random.permutation(list(range(n_units)))[:n_units2]
            resp = np.zeros((n_stims, n_movies, nt, n_units2))

        resp_deg0 = resp_deg0.reshape((batch_size, nt, n_units))
        resp_deg180 = resp_deg180.reshape((batch_size, nt, n_units))

        resp_deg0 = resp_deg0[:, :, idxs] # (batch_size, nt, n_units2)
        resp_deg180 = resp_deg180[:, :, idxs]

        resp[0, n * batch_size: (n+1) * batch_size] = resp_deg0
        resp[1, n * batch_size: (n+1) * batch_size] = resp_deg180

    # reshape
    print(target)
    print(resp.shape) # (2, 100, 230, n_units2)

    # flatness
    i = 0
    count = 0
    responsive_idxs = []
    while count < n_cells_to_plot and i < n_units2:
        if not is_flat2(resp[:, :, :, i]):
            responsive_idxs.append(i)
        i += 1

    # plot
    x = np.arange(0, nt)
    n_plot = int(np.ceil(n_cells_to_plot / 18))
    idx = 0
    for n in range(n_plot):
        fig = plt.figure(figsize=(20, 20))

        for i in range(min(18, n_cells_to_plot - n * 18)):
            ymax = np.max(np.mean(resp[:, :, :, responsive_idxs[idx]], axis=1))

            for j, col, col_shade in zip(range(2), ['m', 'c'], [(1, 0, 1, 0.3), (0, 1, 1, 0.3)]):
                ax = plt.subplot(6, 6, i * 2 + j + 1)
                _mean = np.mean(resp[j, :, :, responsive_idxs[idx]], axis=0)
                # _error = np.std(resp[j, :, :, responsive_idxs[idx]], axis=0) / 2

                ax.plot(x, _mean, col + '-')
                # ax.fill_between(x, _mean - _error, _mean + _error, facecolor = col_shade, edgecolor = col_shade)
                ax.axvspan(pre_frames, pre_frames + stim_frames, facecolor='k', alpha=0.1)
                ax.set_ylim(0, ymax)
                plt.setp(ax.get_xticklabels(), visible=False)
                # plt.setp(ax.get_yticklabels(), visible=False)
                ax.set_xticks(np.arange(0, nt, 10))
                ax.set_title('#' + str(responsive_idxs[idx]), fontsize=15)

            idx += 1

        plt.tight_layout()
        plt.savefig(SAVE_DIR + 'tc/' + target + '_' + str(n) + '.png')
        plt.savefig(SAVE_DIR + 'tc/' + target + '_' + str(n) + '.pdf')
        plt.close()
