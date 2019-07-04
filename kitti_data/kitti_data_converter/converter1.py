import os
import hickle
import numpy as np

parent_in  = '/export/public/ukita/prednet/kitti_data/kitti_data_original/'
parent_out = '/export/public/ukita/prednet/kitti_data/'

filenames = ['X_train', 'X_val', 'X_test', 'sources_train', 'sources_val', 'sources_test']

for filename in filenames:
    tmp = hickle.load(parent_in + filename + '.hkl')
    np.save(parent_out + filename, tmp)
