import os
import hickle
import numpy as np
import cv2

target_size = (192, 224)

parent = '/export/public/ukita/prednet/kitti_data/'

filenames = ['X_train', 'X_val', 'X_test', 'sources_train', 'sources_val', 'sources_test']

for filename in filenames:
    tmp = hickle.load(parent + filename + '.hkl')

    if 'sources' in filename:
        tmp_converted = tmp
    else:
        tmp_converted = np.zeros((len(tmp), ) + target_size + (3, ), dtype=tmp.dtype)
        for i in range(len(tmp)):
            tmp_converted[i] = cv2.resize(tmp[i], (target_size[1], target_size[0]))

    hickle.dump(tmp_converted, parent + filename + str(target_size[0]) + 'x' +
                               str(target_size[1]) + '.hkl', mode='w')
