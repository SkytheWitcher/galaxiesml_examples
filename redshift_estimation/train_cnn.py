import tensorflow as tf
import tensorflow_probability as tfp
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import h5py
import keras
import os
import tensorboard
import argparse
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout, Input, Concatenate
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import StandardScaler
from tensorboard.plugins.hparams import api as hp

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (
    MODEL_PARAMS, TRAINING_PARAMS, get_dataset_paths,
    setup_gpu, validate_hdf5_datasets, MODEL_DIR, LOG_DIR
)

from photoz_utils import *
from DataMakerPlus import *

def parse_args():
    parser = argparse.ArgumentParser(description='Train basic CNN model for redshift estimation')
    parser.add_argument('--model_name', type=str, default='HSC_v6_CNN_delta_v1',
                      help='Name of the model for saving checkpoints and logs')
    parser.add_argument('--image_size', type=int, default=127,
                      help='Size of input images (default: 127)')
    return parser.parse_args()

def setup_paths(model_name):
    """Setup paths for model checkpoints and logs."""
    checkpoint_filepath = os.path.join(MODEL_DIR, model_name, 'checkpoints/model.weights.h5')
    checkpoint_dir = os.path.dirname(checkpoint_filepath)
    log_dir = os.path.join(LOG_DIR, model_name)
    
    # Create directories if they don't exist
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    
    return checkpoint_filepath, log_dir

def main():
    args = parse_args()
    
    # Setup GPU
    setup_gpu()
    
    # Get dataset paths
    dataset_paths = get_dataset_paths(args.image_size)
    
    # Validate datasets
    for path in dataset_paths.values():
        try:
            validate_hdf5_datasets(path)
        except (FileNotFoundError, ValueError) as e:
            print(f"Error: {e}")
            sys.exit(1)
    
    # Setup paths
    checkpoint_filepath, log_dir = setup_paths(args.model_name)
    
    # Model hyperparameters
    hparams = {
        'num_dense_units': MODEL_PARAMS['num_dense_units'],
        'batch_size': MODEL_PARAMS['batch_size'],
        'num_epochs': TRAINING_PARAMS['base_epochs'],
        'learning_rate': MODEL_PARAMS['learning_rate'],
        'z_max': MODEL_PARAMS['z_max']
    }
    
    # Load data
    with h5py.File(dataset_paths['train'], 'r') as f:
        train_len = len(f['specz_redshift'])
    print(f"Training samples: {train_len}")
    
    # Setup data generators
    param_names = [f"{band}_cmodel_mag" for band in ['g', 'r', 'i', 'z', 'y']]
    gen_args = {
        'image_key': 'image',
        'numerical_keys': param_names,
        'y_key': 'specz_redshift',
        'scaler': True,
        'labels_encoding': False,
        'batch_size': hparams['batch_size'],
        'shuffle': False
    }
    
    train_gen = HDF5DataGenerator(dataset_paths['train'], mode='train', **gen_args)
    val_gen = HDF5DataGenerator(dataset_paths['val'], mode='train', **gen_args)
    test_gen = HDF5DataGenerator(dataset_paths['test'], mode='test', **gen_args)
    
    # Build model
    input_cnn = Input(shape=MODEL_PARAMS['image_shape'])
    input_nn = Input(shape=(5,))
    
    # CNN
    conv1 = Conv2D(32, kernel_size=(3, 3), activation='tanh', padding='same', data_format='channels_first')(input_cnn)
    pool1 = MaxPooling2D(pool_size = (2,2), data_format='channels_first')(conv1)
    conv2 = Conv2D(64, kernel_size=(3, 3), activation='tanh', padding='same', data_format='channels_first')(pool1)
    pool2 = MaxPooling2D(pool_size = (2,2), data_format='channels_first')(conv2)
    conv3 = Conv2D(128, kernel_size=(3, 3), activation='tanh', padding='same', data_format='channels_first')(pool2)
    pool3 = MaxPooling2D(pool_size = (2,2), data_format='channels_first')(conv3)
    conv4 = Conv2D(256, kernel_size=(3, 3), activation='tanh', padding='same', data_format='channels_first')(pool3)
    pool4 = MaxPooling2D(pool_size = (2,2), data_format='channels_first')(conv4)
    conv5 = Conv2D(256, kernel_size=(3, 3), activation='tanh', padding='same', data_format='channels_first')(pool4)
    pool5 = MaxPooling2D(pool_size = (2,2), data_format='channels_first')(conv5)
    conv6 = Conv2D(512, kernel_size=(3, 3),activation='relu', padding='same', data_format='channels_first')(pool5)
    conv7 = Conv2D(512, kernel_size=(3, 3),activation='relu', padding='same', data_format='channels_first')(conv6)
    flatten = Flatten()(conv7)
    dense1 = Dense(512, activation='tanh')(flatten)
    dense2 = Dense(128, activation='tanh')(dense1)
    dense3 = Dense(32, activation='tanh')(dense2)
    
    # NN
    hidden1 = Dense(hparams['num_dense_units'], activation="relu")(input_nn)
    hidden2 = Dense(hparams['num_dense_units'], activation="relu")(hidden1)
    hidden3 = Dense(hparams['num_dense_units'], activation="relu")(hidden2)
    hidden4 = Dense(hparams['num_dense_units'], activation="relu")(hidden3)
    hidden5 = Dense(hparams['num_dense_units'], activation="relu")(hidden4)
    hidden6 = Dense(hparams['num_dense_units'], activation="relu")(hidden5)
    
    # Combine & Output
    concat = Concatenate()([dense3, hidden6])
    output = Dense(1)(concat)
    model = Model(inputs=[input_cnn, input_nn], outputs=[output])
    
    model.compile(
        optimizer=Adam(learning_rate=hparams['learning_rate']),
        loss=calculate_loss,
        metrics=[tf.keras.metrics.RootMeanSquaredError()]
    )
    
    # Callbacks
    model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_filepath,
        save_weights_only=True,
        monitor='loss',
        mode='min',
        save_freq='epoch',
        save_best_only=True,
        verbose=True
    )
    
    tensorboard_callback = tf.keras.callbacks.TensorBoard(
        log_dir=log_dir,
        histogram_freq=1
    )
    
    hparam_callback = hp.KerasCallback(log_dir, hparams)
    
    # Train model
    model.fit(
        train_gen,
        batch_size=hparams['batch_size'],
        epochs=hparams['num_epochs'],
        shuffle=True,
        verbose=1,
        validation_data=val_gen,
        callbacks=[tensorboard_callback, model_checkpoint_callback, hparam_callback]
    )

if __name__ == '__main__':
    main()