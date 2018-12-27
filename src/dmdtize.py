import os
import pickle
import multiprocessing

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from tqdm import tnrange, tqdm_notebook, tqdm
from sklearn.preprocessing import MinMaxScaler


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

def dmdtize_single_object(args):
    
    (object_id, read_dir, base_dir) = args

    key = '{}/{}_dmdt.pkl'.format(base_dir, object_id)
    if os.path.isfile(key):
        return
    
    df_object = pd.read_csv('{}/{}.csv'.format(read_dir, object_id), header=None, names=['object_id', 'mjd', 'passband', 'flux', 'flux_err', 'detected'])

    mms = MinMaxScaler(feature_range=(-8,8))
    df_object['local_tr_flux'] = mms.fit_transform(df_object['flux'].values.reshape(-1,1))
    
    dm_bins = [-8, -5, -3, -2.5, -2, -1.5, -1, -0.5, -0.3, -0.2, -0.1, 0, 0.1, 0.2, 0.3, 0.5, 1, 1.5, 2, 2.5, 3, 5, 8]
    dt_bins = [1/145, 2/145, 3/145, 4/145, 1/25, 2/25, 3/25, 1.5, 2.5, 3.5, 4.5, 5.5, 7, 10, 20, 30, 60, 90, 120, 240, 600, 960, 2000, 4000]
    result = df_object.groupby('passband').apply(lambda x : dmdtize_single_band(x, dm_bins, dt_bins, 'local_tr_flux' ))
    dmdt_img = np.stack(result.values, axis=-1)

    max_pairs = np.max(dmdt_img)

    if max_pairs == 0:
        print('max_pairs = 0 for object_id = {}'.format(object_id))

    dmdt_img = np.floor(255*dmdt_img/max_pairs + 0.99999).astype('int')

    with open(key, 'wb') as f:
        pickle.dump(dmdt_img, f)

def dmdtize(read_dir, base_dir='test' ):
    objects = []
    objects_file='{}/objects.csv'.format(read_dir)

    with open(objects_file, 'r') as of:
        objects = of.read().split(',')

    
    with multiprocessing.Pool() as pool:
        with tqdm(total=len(objects), desc="Generating DMDT Images") as pbar:
            df_args = []
            for obj in objects:
                df_args.append((obj, read_dir, base_dir))
            for i, _ in tqdm(enumerate(pool.imap_unordered(dmdtize_single_object, df_args))):
                pbar.update()

dmdtize(read_dir='../input/train/train_csv', base_dir='../input/train/train_dmdt')
