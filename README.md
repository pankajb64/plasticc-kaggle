# plasticc-kaggle
PLAsTiCC Kaggle Challenge Submission

This repository contains the source code of the approach described by  the Kernel - 
[CNN based Classification of Light Curves
](https://www.kaggle.com/pankajb64/cnn-based-classification-of-light-curves).

There's also an associated EDA Kernel [here](https://www.kaggle.com/pankajb64/plasticc-dmdt-eda)

## Dependencies

The requirements.txt is an auto-generated file based on [pigar](https://github.com/Damnever/pigar),
here are the steps I followed to create the environment.

- Install Anaconda 3
- Create a conda environment with python 3.6
- Install Tensorflow
- Install Keras

## Source

I assume the input directory has the following structure:
```
input/
  training_set.csv                                        #file containing the time series data
  training_set_metadata.csv                               #file containing metadata
  train_csv/                                              #base directory to generate the individual csv files (see below)
  train_dmdt/                                             #base directory to generate the DMDT Images (see below)
```

I broke the main kernel into four files:
- `split_csv.py` - breaks the time series data into one csv file per object (this is useful especially when the number of objects is huge - as was the case for the test set). It expects a base directory (`input/train_csv` by default), and generates an objects.csv containing a list of all unique object ids, as well as one csv file per object, containing time series data for that object.
- `dmdtize.py` - generates dmdt images for each object given its individual csv file. It expects a location where the csv files are store (`input/train_csv` by default) and a base directory to store the dmdt images (`input/train_dmdt` by default)
- `train.py` - Trains a Keras model on the DMDT Images. It expects a location where the dmdt images are stored (`input/train_dmdt` by default). The resulting model is saved to `model/model_<timestamp>.h5`
- `predict.py` - Uses the trained model to generate results on a set of images. It expects a location where the dmdt images are stored (`input/train_dmdt` by default), and a model (`model/model_<timestamp>.h5` by default). The results are stored to `output/test_results.csv`.

The general sequence to run the source files would be:
```
python split_csv.py
python dmdtize.py
python train.py
python predict.py
```

I have pre-computed the Images for the training set as a Kaggle Dataset - See [Plasticc DMDT Images](https://www.kaggle.com/pankajb64/plasticc_dmdt_images), so you can skip the first two source files if you just need to run on the training dataset

I also have the images for the test set, but they're understandably huge and hard to share. I have them on a EC2 instance, let me know if you'd like access to it.
