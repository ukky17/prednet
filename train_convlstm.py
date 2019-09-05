import os
import shutil
import numpy as np
np.random.seed(123)
from six.moves import cPickle

from keras import backend as K
from keras.models import Model
from keras.layers import Input, Dense, Flatten
from keras.layers import ConvLSTM2D, Conv2D, MaxPooling2D
from keras.layers import TimeDistributed
from keras.callbacks import LearningRateScheduler, ModelCheckpoint
from keras.optimizers import Adam

from prednet import PredNet
from data_utils import SequenceGenerator

# path
DATA_DIR = './kitti_data/'
WEIGHTS_DIR = './model/190724_14/'

size = (192, 224)

save_model = True  # if weights will be saved
weights_file = os.path.join(WEIGHTS_DIR, 'prednet_kitti_weights.hdf5')  # where weights will be saved
json_file = os.path.join(WEIGHTS_DIR, 'prednet_kitti_model.json')

os.makedirs(WEIGHTS_DIR)
shutil.copy('train_convlstm.py', WEIGHTS_DIR)

# Data files
if size == (128, 160):
    train_file = os.path.join(DATA_DIR, 'X_train.hkl')
    train_sources = os.path.join(DATA_DIR, 'sources_train.hkl')
    val_file = os.path.join(DATA_DIR, 'X_val.hkl')
    val_sources = os.path.join(DATA_DIR, 'sources_val.hkl')
else:
    s = str(size[0]) + 'x' + str(size[1])
    train_file = os.path.join(DATA_DIR, 'X_train' + s + '.hkl')
    train_sources = os.path.join(DATA_DIR, 'sources_train' + s + '.hkl')
    val_file = os.path.join(DATA_DIR, 'X_val' + s + '.hkl')
    val_sources = os.path.join(DATA_DIR, 'sources_val' + s + '.hkl')

# Training parameters
nb_epoch = 150
batch_size = 4
samples_per_epoch = 500
N_seq_val = 100  # number of sequences to use for validation

# Model parameters
input_shape = (None, ) + size + (3, )
stack_sizes = (3, 48, 96)
nt = 5  # number of timesteps used for sequences in training

inputs = Input(shape=input_shape)
x = ConvLSTM2D(filters=stack_sizes[0], kernel_size=(3, 3), strides=(1, 1),
                   input_shape=input_shape, padding='same', activation='relu',
                   return_sequences=True)(inputs)
x = TimeDistributed(Conv2D(filters=stack_sizes[0], kernel_size=(3, 3), strides=(1, 1),
                            padding='same', activation='relu'))(x)

x = TimeDistributed(Conv2D(filters=stack_sizes[1], kernel_size=(3, 3), strides=(1, 1),
                            padding='same', activation='relu'))(x)
# x = TimeDistributed(MaxPooling2D((2, 2)))(x)

x = ConvLSTM2D(filters=stack_sizes[1], kernel_size=(3, 3), strides=(1, 1),
                   padding='same', activation='relu',
                   return_sequences=True)(x)
x = TimeDistributed(Conv2D(filters=stack_sizes[1], kernel_size=(3, 3), strides=(1, 1),
                            padding='same', activation='relu'))(x)

x = TimeDistributed(Conv2D(filters=stack_sizes[2], kernel_size=(3, 3), strides=(1, 1),
                            padding='same', activation='relu'))(x)
# x = TimeDistributed(MaxPooling2D((4, 4)))(x)

x = ConvLSTM2D(filters=stack_sizes[2], kernel_size=(4, 4), strides=(1, 1),
                   padding='same', activation='relu',
                   return_sequences=True)(x)
x = TimeDistributed(Conv2D(filters=3, kernel_size=(4, 4), strides=(1, 1),
                               padding='same', activation='relu'))(x)
model = Model(inputs=inputs, outputs=x)
model.compile(loss='mean_absolute_error', optimizer='adam')
print(model.summary())

train_generator = SequenceGenerator(train_file, train_sources, nt,
                    batch_size=batch_size, shuffle=True, output_mode='convlstm')
val_generator = SequenceGenerator(val_file, val_sources, nt,
                    batch_size=batch_size, N_seq=N_seq_val, output_mode='convlstm')

# start with lr of 0.001 and then drop to 0.0001 after 75 epochs
lr_schedule = lambda epoch: 0.001 if epoch < nb_epoch/2 else 0.0001
callbacks = [LearningRateScheduler(lr_schedule)]
if save_model:
    if not os.path.exists(WEIGHTS_DIR): os.mkdir(WEIGHTS_DIR)
    callbacks.append(ModelCheckpoint(filepath=weights_file, monitor='val_loss', save_best_only=True))

history = model.fit_generator(train_generator, samples_per_epoch / batch_size,
                epochs=nb_epoch, callbacks=callbacks,
                validation_data=val_generator, validation_steps=N_seq_val / batch_size)

if save_model:
    json_string = model.to_json()
    with open(json_file, "w") as f:
        f.write(json_string)
