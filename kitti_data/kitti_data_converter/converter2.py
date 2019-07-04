import os
import hickle
import numpy as np

parent = '/export/public/ukita/prednet/kitti_data/'

filenames = ['X_train', 'X_val', 'X_test', 'sources_train', 'sources_val', 'sources_test']

for filename in filenames:
    tmp = np.load(parent + filename + '.npy')
    hickle.dump(tmp, parent + filename + '.hkl', mode='w')
