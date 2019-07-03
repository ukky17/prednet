import os
import hickle
import pickle

parent = '/export/public/ukita/PredNet/prednet/kitti_data/'

filenames = ['X_val', 'X_test', 'sources_val', 'sources_test']

for filename in filenames:
    tmp = pickle.load(open(parent + filename + '.pkl', 'rb'), encoding='latin1')
    hickle.dump(tmp, parent + filename + '.hkl', mode='w')
