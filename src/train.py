
import os
import pickle

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import keras.backend as K #to define custom loss function
import tensorflow as tf #We'll use tensorflow backend here

from collections import OrderedDict
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dropout, Dense, Flatten, Input, AveragePooling1D, Reshape, DepthwiseConv2D, SeparableConv2D
from keras.optimizers import Adam, SGD
from keras.callbacks import ReduceLROnPlateau, ModelCheckpoint, EarlyStopping
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime

meta_file = '../input/training_set_metadata.csv'
img_dir = '../input/train/train_dmdt'
df_meta = pd.read_csv(meta_file)

objects = df_meta['object_id'].drop_duplicates().values

# In[9]:
def load_dmdt_images(objects, base_dir='train'):
    dmdt_img_dict = OrderedDict()
    for obj in objects:
        key = '{}/{}_dmdt.pkl'.format(base_dir, obj)
        if os.path.isfile(key):
            with(open(key, 'rb')) as f:
                dmdt_img_dict[obj] = pickle.load(f)
    return dmdt_img_dict


dmdt_img_dict = load_dmdt_images(objects, img_dir)
X = np.array(list(dmdt_img_dict.values()), dtype='int')

labels = pd.get_dummies(df_meta.loc[df_meta['object_id'].isin(dmdt_img_dict.keys()) , 'target'])
y = labels.values


splitter = StratifiedShuffleSplit(n_splits=1, test_size=0.2)
splits = list(splitter.split(X, y))[0]
train_ind, test_ind = splits

X_train = X[train_ind]
X_test  = X[test_ind]

y_train = y[train_ind]
y_test  = y[test_ind]

print('Training set size: {}'.format(y_train.shape[0]))
print('Test set size: {}'.format(y_test.shape[0]))

class ReduceLRWithEarlyStopping(ReduceLROnPlateau):
    def __init__(self, *args, **kwargs):
        super(ReduceLRWithEarlyStopping, self).__init__(*args, **kwargs)
        self.stopped_epoch = 0

    def on_epoch_end(self, epoch, logs=None):
        super(ReduceLRWithEarlyStopping, self).on_epoch_end(epoch, logs)
        old_lr = float(K.get_value(self.model.optimizer.lr))
        if self.wait >= self.patience and old_lr <= self.min_lr:
            # Stop training early
            self.stopped_epoch = epoch
            self.model.stop_training = True

    def on_train_end(self, logs=None):
        super(ReduceLRWithEarlyStopping, self).on_epoch_end(logs)
        if self.stopped_epoch > 0 and self.verbose > 0:
            print('Epoch %05d: early stopping' % (self.stopped_epoch + 1))


def build_model(n_dm_bins=23, n_dt_bins=24, n_passbands=6, n_classes=14):
    model = Sequential()
    model.add(DepthwiseConv2D(kernel_size=2, strides=1, depth_multiplier=8,
                     padding="valid", activation="elu",
                     input_shape=(n_dm_bins, n_dt_bins, n_passbands)))
    model.add(Dropout(0.1))
    model.add(SeparableConv2D(48, kernel_size=2, strides=1, depth_multiplier=2,
                    padding="valid", activation="elu"))
    model.add(MaxPooling2D(pool_size=2))
    model.add(Dropout(0.1))
    model.add(Flatten())
    model.add(Dense(32, activation="elu"))
    model.add(Dropout(0.25))
    model.add(Dense(n_classes, activation="softmax"))
    print(model.summary())
    return model

n_classes=14
#assumes weights to be all ones as actual weights are hidden
#UPDATE - settings weights for classes 15 (idx=1) and 64(idx=7) to 2 based on LB probing post 
#https://www.kaggle.com/c/PLAsTiCC-2018/discussion/67194#397153
weights = np.ones(n_classes, dtype='float32') 
weights[1], weights[7] = 2, 2
epsilon = 1e-7
#number of objects per class
class_counts = df_meta.groupby('target')['object_id'].count().values 
#proportion of objects per class
class_proportions = class_counts/np.max(class_counts)
#set backend to float 32
K.set_floatx('float32')

#weighted multi-class log loss
def weighted_mc_log_loss(y_true, y_pred):
    y_pred_clipped = K.clip(y_pred, epsilon, 1-epsilon)
    #true labels weighted by weights and percent elements per class
    y_true_weighted = (y_true * weights)/class_proportions
    #multiply tensors element-wise and then sum
    loss_num = (y_true_weighted * K.log(y_pred_clipped))
    loss = -1*K.sum(loss_num)/K.sum(weights)
    
    return loss


model = build_model()
model.compile(loss=weighted_mc_log_loss, optimizer=Adam(lr=0.002), metrics=['accuracy'])

checkPoint = ModelCheckpoint("../model/keras.model",monitor='val_loss',mode = 'min', save_best_only=True, verbose=1)
reduce_lr = ReduceLRWithEarlyStopping(monitor='val_loss', factor=0.2, patience=5, min_lr=0.00001, verbose=1)
history = model.fit(X_train, y_train, batch_size=32, epochs=100, validation_data=[X_test, y_test],shuffle=True,verbose=1,callbacks=[checkPoint, reduce_lr])

loss_acc = model.evaluate(X_test, y_test, batch_size=32)
print('Validation loss: {}'.format(loss_acc[0]))
print('Validation accuracy: {}'.format(loss_acc[1]))


time_stamp = datetime.now().strftime('%Y_%m_%d_%H_%M_%S')
save_file = '../model/model_{}.h5'.format(time_stamp)
model.save(save_file)