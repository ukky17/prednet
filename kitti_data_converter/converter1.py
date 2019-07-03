import os
import hickle
import pickle

parent_in = '/export/public/ukita/PredNet/prednet/kitti_data_original/'
parent_out = '/export/public/ukita/PredNet/prednet/kitti_data/'

filenames = ['X_val', 'X_test', 'sources_val', 'sources_test']

for filename in filenames:
    tmp = hickle.load(open(parent_in + filename + '.hkl'))
    pickle.dump(tmp, open(parent_out + filename + '.pkl', 'wb'), 2)
