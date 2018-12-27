import os
import pickle
import multiprocessing
import csv

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import tensorflow as tf #We'll use tensorflow backend here

from tqdm import tnrange, tqdm_notebook, tqdm
from keras.backend import tensorflow_backend as K #to define custom loss function
from keras.models import Sequential, load_model
from keras.layers import Conv2D, MaxPooling2D, Dropout, Dense, Flatten
from keras.optimizers import Adam, SGD
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime

meta_file = '../input/training_set_metadata.csv'

with tf.Session(config=tf.ConfigProto(inter_op_parallelism_threads=16)) as sess:
    K.set_session(sess)

    df_meta = pd.read_csv(meta_file)
    n_classes=14
    #UPDATE - settings weights for classes 65 (idx=8) and 92(idx=12) to 2 based on LB probing post 
    weights = np.ones(n_classes, dtype='float32') 
    weights[8], weights[12] = 2, 2
    epsilon = 1e-7
    class_counts = df_meta.groupby('target')['object_id'].count().values #number of objects per class, calcualte this
    #proportion of objects per class
    class_proportions = class_counts/np.max(class_counts)
    
    def weighted_mc_log_loss(y_true, y_pred):
        y_pred_clipped = K.clip(y_pred, epsilon, 1-epsilon)
        #true labels weighted by weights and percent elements per class
        y_true_weighted = (y_true * weights)/class_proportions
        #multiply tensors element-wise and then sum
        loss_num = (y_true_weighted * K.log(y_pred_clipped))
        loss = -1*K.sum(loss_num)/K.sum(weights)
        
        return loss
        

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

        batch_preds = dmdt_predict_batch(batch, read_dir, model)
        ohe_preds  = [modify_prob(pred) for pred in batch_preds]
        return ohe_preds

    def dmdt_predict(objects_file, read_dir, model_file, output_file, batch_size=512):
        objects = []
        with open(objects_file, 'r') as of:
            objects = of.read().split(',')

        split_size = len(objects)//batch_size
        batches = np.array_split(objects, split_size)
        
        model = load_model(model_file, custom_objects={'weighted_mc_log_loss': weighted_mc_log_loss})


        classes = [6, 15, 16, 42, 52, 53, 62, 64, 65, 67, 88, 90, 92, 95, 99]
        pred_cols = ['class_{}'.format(clazz) for clazz in classes]

        predictions = []

        with multiprocessing.Pool() as pool:
            with tqdm(total=len(batches), desc="Computing Predictions") as pbar:
                batch_args = []
                for batch in batches:
                    pred = process_batch((batch, read_dir, model))
                    predictions.extend(pred)
                    pbar.update()

        predictions = np.array(predictions).reshape(-1, len(classes))
    
        pred_df = pd.DataFrame(predictions, columns=pred_cols)
        pred_df['object_id'] = objects
        pred_df.to_csv(output_file, index=False)

    dmdt_predict( 
        objects_file='../input/train/train_csv/objects.csv',
        read_dir='../input/train/train_dmdt', 
        model_file='../model/model_2018_12_27_17_11_02.h5',
        output_file='../output/test_results.csv',
        batch_size=1024)
