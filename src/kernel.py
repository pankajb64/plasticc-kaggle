
# coding: utf-8

# In[1]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
import pickle
import multiprocessing
import csv

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import tensorflow as tf #We'll use tensorflow backend here
import dask.dataframe as dd #To handle gigantic csv files
import tensorflow as tf

from tqdm import tnrange, tqdm_notebook, tqdm
from collections import OrderedDict
from keras.backend import tensorflow_backend as K #to define custom loss function
from keras.models import Sequential, load_model
from keras.layers import Conv2D, MaxPooling2D, Dropout, Dense, Flatten
from keras.optimizers import Adam, SGD
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime
from sqlalchemy import create_engine
from sqlalchemy.pool import QueuePool

print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# #Ref - https://arxiv.org/pdf/1709.06257.pdf
# 
# CNN based classification of light curves
# 
# Do an EDA and find about the classification and decide whether or not to deal with it. If yes, then how. (This might be helpful - https://www.kaggle.com/danilodiogo/the-astronomical-complete-eda-plasticc-dataset)
# 
# Convert light curves in dmdt images (different bands can act as different channels of the image). Consider, the binning size for dm and dt in the image.
# 
# 
# CNN model to train the images (consider interfacing this multi channel input, and also what the output would look like, how would you classify "other" class).
# 
# Train it for training set with a validation set, and then test it for testing set. Report the Precision, Recall, F1-Score, confusion matrix. Poissibly evaluate examples for which it doesn't do a good job.
# 
# Consider imoprovements - varying # of bins, tuning CNN hyper-parameters, using a different architecture, subtracting image background based on cadence. These are suggested in no particular order, you may pick the low hanging fruit first. Repeat the previous step of reporting for each case.
# 
# Consider comparing with Random Forests or Gradient Boosting methods using features calculated from cesium.
# 
# Some more useful/fun references
# 
# https://www.kaggle.com/mithrillion/all-classes-light-curve-characteristics
# 
# https://www.kaggle.com/hrmello/using-cnns-in-time-series-analysis
# 
# Many more in the "kernels" section.
# 

# In[2]:


#df = pd.read_csv('../input/training_set.csv')


# In[3]:

#with open('hello.txt', 'w') as hf:
#with tf.Session(config=tf.ConfigProto(inter_op_parallelism_threads=16)) as sess:
#    K.set_session(sess)

def dmdtize_single_band(df, dm_bins, dt_bins, col='flux'):
    n_points = df.shape[0]
    dmdt_img = np.zeros((len(dm_bins), len(dt_bins)), dtype='int')
    for i in range(n_points):
        for j in range(i+1, n_points):
            dmi = float(df.iloc[i][col])
            dmj = float(df.iloc[j][col])
            dti = float(df.iloc[i]['mjd'])
            dtj = float(df.iloc[j]['mjd'])
            
            dm = dmj - dmi if dtj > dti else dmi - dmj
            dt = abs(dtj - dti)
            
            dm_idx = min(np.searchsorted(dm_bins, dm), len(dm_bins)-1)
            dt_idx = min(np.searchsorted(dt_bins, dt), len(dt_bins)-1)
            
            dmdt_img[dm_idx, dt_idx] += 1
    return dmdt_img
            


# In[4]:


def dmdtize_single_object(args):
    (df, object_id, base_dir) = args
    key = '{}/{}_dmdt.pkl'.format(base_dir, object_id)
    if os.path.isfile(key):
        return
    num_bands = 6
    dm_bins = [-8, -5, -3, -2.5, -2, -1.5, -1, -0.5, -0.3, -0.2, -0.1, 0, 0.1, 0.2, 0.3, 0.5, 1, 1.5, 2, 2.5, 3, 5, 8]
    dt_bins = [1/145, 2/145, 3/145, 4/145, 1/25, 2/25, 3/25, 1.5, 2.5, 3.5, 4.5, 5.5, 7, 10, 20, 30, 60, 90, 120, 240, 600, 960, 2000, 4000]
    dmdt_img = np.zeros((len(dm_bins), len(dt_bins), num_bands), dtype='int')
    
    max_points = 0
    for band_idx in range(num_bands):
        df_band = df.loc[df['passband'] == band_idx]
        dmdt_img[:, :, band_idx] = dmdtize_single_band(df_band, dm_bins, dt_bins)
        if band_idx == 0 or df_band.shape[0] > max_points:
            max_points = df_band.shape[0] #store max points to scale the image later
    
    max_pairs = (max_points*(max_points-1))//2
    if max_pairs == 0:
        print('max_pairs = 0 for object_id = {},  max_points = {}'.format(object_id, max_points))
    dmdt_img = np.floor(255*dmdt_img/max_pairs + 0.99999).astype('int')
    with open(key, 'wb') as f:
        pickle.dump(dmdt_img, f)
    #return dmdt_img
        


# In[5]:


def dmdtize(df, base_dir='train'):
    objects = df['object_id'].drop_duplicates().values
    #print(objects)
    #nobjects = len(objects)
    dmdt_img_dict = {}
    #with tqdm_notebook(total=nobjects, desc="Computing Features") as pbar:
    for i in [0]:
	    pool = multiprocessing.Pool()
	    df_args = []
	    for obj in objects:
	        df_obj = df.loc[df['object_id'] == obj]
	        df_args.append((df_obj, obj, base_dir))
	    pool.map(dmdtize_single_object, df_args[0])
	    #for idx, obj in enumerate(objects):
	    #    dmdt_img_dict[objects[idx]] = dmdt_imgs[idx]
	    #    pbar.update()
    #return dmdt_img_dict

def dmdtize_test(objects_file, read_dir, base_dir='test' ):
    objects = []
    with open(objects_file, 'r') as of:
        objects = of.read().split(',')
    #objects = df['object_id'].drop_duplicates().values.compute()
    #print(objects)
    #nobjects = len(objects)
    #dmdt_img_dict = {}

    col = 'tr_flux'
    print(col)
    
    with multiprocessing.Pool() as pool:
        with tqdm(total=len(objects), desc="Computing Features") as pbar:
            df_args = []
            for obj in objects:
                df_args.append((obj, read_dir, base_dir, col))
            for i, _ in tqdm(enumerate(pool.imap_unordered(dmdtize_single_object_test, df_args))):
                pbar.update()

    #with tqdm_notebook(total=3500000, desc="Computing Features") as pbar:
    #    df.groupby('object_id').apply(lambda x : dmdtize_single_object_test(x, base_dir, pbar), meta=pd.Series(dtype='int')).compute()
    #with tqdm_notebook(total=nobjects, desc="Computing Features") as pbar:
        #pool = multiprocessing.Pool()
        #df_args = []
        #for obj in objects:
        #    df_args.append((obj, base_dir))
        #pool.map(dmdtize_single_object_test, df_args)
        #for idx, obj in enumerate(objects):
        #    dmdt_img_dict[objects[idx]] = dmdt_imgs[idx]
        #    pbar.update()
    #return dmdt_img_dict

def dmdtize_single_object_test(args):
    
    (object_id, read_dir, base_dir, col) = args
    key = '{}/{}_dmdt.pkl'.format(base_dir, object_id)
    if os.path.isfile(key):
        return
    df_object = pd.read_csv('{}/{}.csv'.format(read_dir, object_id), header=None, names=['object_id', 'mjd', 'passband', 'flux', 'flux_err', 'detected'])

    mms = MinMaxScaler(feature_range=(-8,8))
    df_object['local_tr_flux'] = mms.fit_transform(df_object['flux'].values.reshape(-1,1))
    #print(object_id, base_dir)
    #df_object = pd.read_sql_query("select passband, mjd, flux from test_ts where object_id = {}".format(object_id), plasticc_db)
    num_bands=6
    dm_bins = [-8, -5, -3, -2.5, -2, -1.5, -1, -0.5, -0.3, -0.2, -0.1, 0, 0.1, 0.2, 0.3, 0.5, 1, 1.5, 2, 2.5, 3, 5, 8]
    dt_bins = [1/145, 2/145, 3/145, 4/145, 1/25, 2/25, 3/25, 1.5, 2.5, 3.5, 4.5, 5.5, 7, 10, 20, 30, 60, 90, 120, 240, 600, 960, 2000, 4000]
    result = df_object.groupby('passband').apply(lambda x : dmdtize_single_band(x, dm_bins, dt_bins, 'local_tr_flux' ))
    dmdt_img = np.stack(result.values, axis=-1)
    #print(dmdt_img.shape)
    max_pairs = np.max(dmdt_img)
    #print(max_pairs)
    if max_pairs == 0:
        print('max_pairs = 0 for object_id = {}'.format(object_id))
    dmdt_img = np.floor(255*dmdt_img/max_pairs + 0.99999).astype('int')
    with open(key, 'wb') as f:
        pickle.dump(dmdt_img, f)
    
'''    
def dmdtize_test(df, base_dir='train'):
    grouped = df.groupby('object_id')
    for object_id, group in grouped:
        dmdtize_single_object_test((group, object_id, base_dir))

def dmdtize_single_object_test(args):
    (df, object_id, base_dir) = args
    key = '{}/{}_dmdt.pkl'.format(base_dir, object_id)
    if os.path.isfile(key):
        return
    num_bands = 6
    dm_bins = [-8, -5, -3, -2.5, -2, -1.5, -1, -0.5, -0.3, -0.2, -0.1, 0, 0.1, 0.2, 0.3, 0.5, 1, 1.5, 2, 2.5, 3, 5, 8]
    dt_bins = [1/145, 2/145, 3/145, 4/145, 1/25, 2/25, 3/25, 1.5, 2.5, 3.5, 4.5, 5.5, 7, 10, 20, 30, 60, 90, 120, 240, 600, 960, 2000, 4000]
    dmdt_img = np.zeros((len(dm_bins), len(dt_bins), num_bands), dtype='int')
    
    max_points = 0
    grouped = df.groupby('passband')
    for band_idx, group in grouped:
        dmdt_img[:, :, band_idx] = dmdtize_single_band(group.compute(), dm_bins, dt_bins)

    print(dmdt_img.shape)
    max_pairs = np.max(np.sum(dmdt_img, axis=(0,1))) #find max number of pairs for a given passband    
    dmdt_img = np.floor(255*dmdt_img/max_pairs + 0.99999).astype('int')
    with open(key, 'wb') as f:
        pickle.dump(dmdt_img, f)
    #return dmdt_img
'''

df_meta = pd.read_csv('../input/training_set_metadata.csv')
n_classes=14
#UPDATE - settings weights for classes 65 (idx=8) and 92(idx=12) to 2 based on LB probing post 
weights = np.ones(n_classes, dtype='float32') 
weights[8], weights[12] = 2, 2
epsilon = 1e-7
class_counts = df_meta.groupby('target')['object_id'].count().values #number of objects per class, calcualte this
#set backend to float 32
#K.set_floatx('float32')

def weighted_mc_log_loss(y_true, y_pred):
    y_pred_clipped = K.clip(y_pred, epsilon, 1-epsilon)
    #true labels weighted by weights and counts per class
    y_true_weighted = (y_true * weights)/class_counts
    #multiply tensors element-wise and then sum
    loss_num = (y_true_weighted * K.log(y_pred_clipped))
    loss = K.sum(loss_num)/K.sum(weights)
    
    return loss

def split_csv():
    with open('../input/test_set.csv', 'r') as f:
        ignore = f.readline()
        object_id = None
        lines = []
        for line in f:
            fields = line.split(',')
            if object_id is None or fields[0] == object_id:
                lines.append(line)
                if object_id is None:
                    object_id = fields[0]
            else:
                with open('/data1/plasticc/input/test_csv/{}.csv'.format(object_id), 'w') as cs:
                    cs.writelines(lines)
                object_id = fields[0]
                lines = [line]
        if len(lines) > 0:
            with open('/data1/plasticc/input/test_csv/{}.csv'.format(object_id), 'w') as cs:
                    cs.writelines(lines)

def load_dmdt_image(object_id, read_dir):
    key = '{}/{}_dmdt.pkl'.format(read_dir, object_id)
    dmdt_img = None
    with open(key, 'rb') as f:
        dmdt_img = pickle.load(f)
    return dmdt_img

def load_dmdt_images(batch, read_dir):
    dmdt_imgs = [load_dmdt_image(object_id, read_dir) for object_id in batch]
    return np.array(dmdt_imgs)

def dmdt_predict_batch(batch, read_dir, model):
    dmdt_imgs = load_dmdt_images(batch, read_dir)
    predictions = model.predict(dmdt_imgs, batch_size=len(batch))
    del dmdt_imgs
    return predictions

def modify_prob(pred):
    max_pred = np.max(pred)
    max_pred_idx = np.argmax(pred)
    ohe_pred = np.zeros(pred.shape[0] + 1, dtype='float32')
    if max_pred >= 0.5:
        ohe_pred[:-1] = pred
    else:
        ohe_pred[max_pred_idx] = max_pred
        ohe_pred[-1] = 1 - max_pred
    return ohe_pred

def process_batch(batch_args):
    (batch, read_dir, model) = batch_args

    #model = load_model(model_file, custom_objects={'weighted_mc_log_loss': weighted_mc_log_loss})

    batch_preds = dmdt_predict_batch(batch, read_dir, model)
    ohe_preds  = [modify_prob(pred) for pred in batch_preds]
    return ohe_preds

def dmdt_predict(objects_file, read_dir, model_file, output_file, batch_size=512):
    objects = []
    with open(objects_file, 'r') as of:
        objects = of.read().split(',')

    #print(objects)

    split_size = len(objects)//batch_size
    batches = np.array_split(objects, split_size)
    
    model = load_model(model_file, custom_objects={'weighted_mc_log_loss': weighted_mc_log_loss})


    classes = [6, 15, 16, 42, 52, 53, 62, 64, 65, 67, 88, 90, 92, 95, 99]
    pred_cols = ['class_{}'.format(clazz) for clazz in classes]

    predictions = [] #process_batch(batch, read_dir, model) for batch in batches]

    with multiprocessing.Pool() as pool:
        with tqdm(total=len(batches), desc="Computing Predictions") as pbar:
            batch_args = []
            for batch in batches:
                pred = process_batch((batch, read_dir, model))
                #batch_args.append((batch, read_dir, model_file))
                #for i, pred in tqdm(enumerate(pool.map(process_batch, batch_args))):
                predictions.extend(pred)
                pbar.update()

    predictions = np.array(predictions).reshape(-1, len(classes))
    print(predictions.shape)

    with open('temp.pkl', 'wb') as ft:
        pickle.dump(predictions, ft)
    #labeldict = {pred_cols[i] : len(predictions[:,i].tolist()) for i in range(len(classes))}
    #print(labeldict)

    pred_df = pd.DataFrame(predictions, columns=pred_cols)
    pred_df['object_id'] = objects
    pred_df.to_csv(output_file, index=False)

'''
dmdt_predict(
    objects_file='/data1/plasticc/input/test_csv/objects.csv', 
    read_dir='/data1/plasticc/input/test', 
    model_file='keras.model',
    output_file='test_results.csv',
    batch_size=1024)'''

#sess.close()
#plasticc_db = create_engine("sqlite:///data/plasticc/input/plasticc.db", poolclass=QueuePool, pool_size=20, max_overflow=0)
#df_test = pd.read_sql('/data/plasticc/input/plasticc.db', usecols=['object_id', 'mjd', 'passband', 'flux'])
dmdtize_test('train_augmented/train_csv/objects.csv', read_dir='train_augmented/train_csv', base_dir='train_augmented/train')
#split_csv()
# In[6]:
#dmdtize(df)
'''
objects = df['object_id'].drop_duplicates().values

def load_dmdt_images(objects, base_dir='train'):
    dmdt_img_dict = OrderedDict()
    for obj in objects:
        key = '{}/{}_dmdt.pkl'.format(base_dir, obj)
        if os.path.isfile(key):
            with(open(key, 'rb')) as f:
                dmdt_img_dict[obj] = pickle.load(f)
    return dmdt_img_dict

dmdt_img_dict = load_dmdt_images(objects, 'train')

X = np.array(list(dmdt_img_dict.values()), dtype='int')

df_meta = pd.read_csv('../input/training_set_metadata.csv')
labels = pd.get_dummies(df_meta.loc[df_meta['object_id'].isin(dmdt_img_dict.keys()) , 'target'])

y = labels.values

#TODO split X and y into train/test set. (Maybe a val set ?)
splitter = StratifiedShuffleSplit(n_splits=1, test_size=0.2)
splits = list(splitter.split(X, y))[0]
train_ind, test_ind = splits

X_train = X[train_ind]
X_test  = X[test_ind]

y_train = y[train_ind]
y_test  = y[test_ind]

def build_model(n_dm_bins=23, n_dt_bins=24, n_passbands=6, n_classes=14):
    model = Sequential()
    model.add(Conv2D(32, kernel_size=2, strides=1,
                     padding="valid", activation="relu",
                     input_shape=(n_dm_bins, n_dt_bins, n_passbands)))
    model.add(MaxPooling2D(pool_size=2))
    model.add(Dropout(0.1))
    model.add(Conv2D(64, kernel_size=3, strides=1,
                    padding="valid", activation="relu"))
    model.add(MaxPooling2D(pool_size=2))
    model.add(Dropout(0.1))
    model.add(Flatten())
    model.add(Dense(32, activation="relu"))
    model.add(Dropout(0.25))
    model.add(Dense(n_classes, activation="softmax"))
    print(model.summary())
    return model

model = build_model()

n_classes=14
weights = np.ones(n_classes, dtype='float32') #assumes weights to be all ones as actual weights are hidden
epsilon = 1e-7
class_counts = df_meta.groupby('target')['object_id'].count().values #number of objects per class, calcualte this
#set backend to float 32
K.set_floatx('float32')

#weighted multi-class log loss
def weighted_mc_log_loss(y_true, y_pred):
    y_pred_clipped = K.clip(y_pred, epsilon, 1-epsilon)
    #true labels weighted by weights and counts per class
    y_true_weighted = (y_true * weights)/class_counts
    #multiply tensors element-wise and then sum
    loss_num = (y_true_weighted * K.log(y_pred_clipped))
    loss = K.sum(loss_num)/K.sum(weights)
    
    return loss

y_true = K.variable(np.eye(14, dtype='float32'))
y_pred = K.variable(np.eye(14, dtype='float32'))
res = weighted_mc_log_loss(y_true, y_pred)
print(K.eval(res))

model.compile(loss=weighted_mc_log_loss, optimizer=Adam(lr=0.0002), metrics=['accuracy'])

#TODO run model with training(/val)/test data
history = model.fit(X_train, y_train, batch_size=32, epochs=10)

loss_acc = model.evaluate(X_test, y_test, batch_size=32)

print(loss_acc)

pred = model.predict(X_test[:10])

print(pred)

time_stamp = datetime.now().strftime('%Y-%m-%d:%H:%M:%S')
save_file = 'model_{}.h5'.format(time_stamp)
model.save(save_file)
'''
