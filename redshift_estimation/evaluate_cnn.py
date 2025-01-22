import tensorflow as tf
import tensorflow_probability as tfp
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import h5py
import keras
import os
import argparse
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout, Input, Concatenate
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import StandardScaler

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (
    MODEL_PARAMS, get_dataset_paths,
    setup_gpu, validate_hdf5_datasets, MODEL_DIR
)

from photoz_utils import *
from DataMakerPlus import *

def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate CNN model for redshift estimation')
    parser.add_argument('--model_name', type=str, default='HSC_v6_CNN_delta_v1',
                      help='Name of the model to evaluate')
    parser.add_argument('--image_size', type=int, default=127,
                      help='Size of input images (default: 127)')
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Setup GPU with lower memory for evaluation
    os.environ['GALAXIESML_GPU_MEMORY'] = '2'  # Only need 2GB for evaluation
    setup_gpu()
    
    # Get dataset paths
    dataset_paths = get_dataset_paths(args.image_size)
    
    # Model checkpoint path
    checkpoint_filepath = os.path.join(MODEL_DIR, args.model_name, 'checkpoints/cp.ckpt')
    
    # Setup data generators
    param_names = [f"{band}_cmodel_mag" for band in ['g', 'r', 'i', 'z', 'y']]
    gen_args = {
        'image_key': 'image',
        'numerical_keys': param_names,
        'y_key': 'specz_redshift',
        'scaler': True,
        'labels_encoding': False,
        'batch_size': MODEL_PARAMS['batch_size'],
        'shuffle': False
    }
    
    test_gen = HDF5DataGenerator(dataset_paths['test'], mode='test', **gen_args)
    
    # Build model (same architecture as training)
    input_cnn = Input(shape=MODEL_PARAMS['image_shape'])
    input_nn = Input(shape=(5,))
    
    # CNN
    conv1 = Conv2D(32, kernel_size=(3, 3), activation='tanh', padding='same', data_format='channels_first')(input_cnn)
    pool1 = MaxPooling2D(pool_size=(2,2), data_format='channels_first')(conv1)
    conv2 = Conv2D(64, kernel_size=(3, 3), activation='tanh', padding='same', data_format='channels_first')(pool1)
    pool2 = MaxPooling2D(pool_size=(2,2), data_format='channels_first')(conv2)
    conv3 = Conv2D(128, kernel_size=(3, 3), activation='tanh', padding='same', data_format='channels_first')(pool2)
    pool3 = MaxPooling2D(pool_size=(2,2), data_format='channels_first')(conv3)
    conv4 = Conv2D(256, kernel_size=(3, 3), activation='tanh', padding='same', data_format='channels_first')(pool3)
    pool4 = MaxPooling2D(pool_size=(2,2), data_format='channels_first')(conv4)
    conv5 = Conv2D(256, kernel_size=(3, 3), activation='tanh', padding='same', data_format='channels_first')(pool4)
    pool5 = MaxPooling2D(pool_size=(2,2), data_format='channels_first')(conv5)
    conv6 = Conv2D(512, kernel_size=(3, 3), activation='relu', padding='same', data_format='channels_first')(pool5)
    conv7 = Conv2D(512, kernel_size=(3, 3), activation='relu', padding='same', data_format='channels_first')(conv6)
    flatten = Flatten()(conv7)
    dense1 = Dense(512, activation='tanh')(flatten)
    dense2 = Dense(128, activation='tanh')(dense1)
    dense3 = Dense(32, activation='tanh')(dense2)
    
    # NN
    hidden1 = Dense(MODEL_PARAMS['num_dense_units'], activation="relu")(input_nn)
    hidden2 = Dense(MODEL_PARAMS['num_dense_units'], activation="relu")(hidden1)
    hidden3 = Dense(MODEL_PARAMS['num_dense_units'], activation="relu")(hidden2)
    hidden4 = Dense(MODEL_PARAMS['num_dense_units'], activation="relu")(hidden3)
    hidden5 = Dense(MODEL_PARAMS['num_dense_units'], activation="relu")(hidden4)
    hidden6 = Dense(MODEL_PARAMS['num_dense_units'], activation="relu")(hidden5)
    
    # Combine & Output
    concat = Concatenate()([dense3, hidden6])
    output = Dense(1)(concat)
    model = Model(inputs=[input_cnn, input_nn], outputs=[output])
    
    model.compile(
        optimizer=Adam(learning_rate=MODEL_PARAMS['learning_rate']),
        loss=calculate_loss,
        metrics=[tf.keras.metrics.RootMeanSquaredError()]
    )
    
    # Load trained weights
    model.load_weights(checkpoint_filepath)
    
    # Make predictions
    pred = model.predict(test_gen)
    
    # Load test data for evaluation
    with h5py.File(dataset_paths['test'], 'r') as file:
        y_test = np.asarray(file['specz_redshift'][:])
        oid_test = np.asarray(file['object_id'][:])
    
    # Plot results
    plot_predictions(np.ravel(pred), y_test)
    
    # Calculate metrics
    metrics = get_point_metrics(pd.Series(np.ravel(pred)), pd.Series(y_test), binned=False)
    print("\nModel Performance Metrics:")
    print(metrics)
    
    # Save predictions and metrics
    predictions_dir = os.path.join(MODEL_DIR, args.model_name, 'predictions')
    os.makedirs(predictions_dir, exist_ok=True)
    
    df = pd.DataFrame({
        'photoz': np.ravel(pred),
        'specz': y_test,
        'object_id': oid_test
    })
    
    df.to_csv(os.path.join(predictions_dir, 'testing_predictions.csv'), index=False)
    metrics.to_csv(os.path.join(predictions_dir, 'testing_metrics.csv'), index=False)

if __name__ == '__main__':
    main()