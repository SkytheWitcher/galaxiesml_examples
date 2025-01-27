"""
Configuration settings for GalaxiesML Examples
"""

import os
from pathlib import Path

# Base paths with environment variable overrides
BASE_DIR = Path(__file__).parent.absolute()
DATA_DIR = os.getenv('GALAXIESML_DATA_DIR', os.path.join(BASE_DIR, 'data'))
MODEL_DIR = os.getenv('GALAXIESML_MODEL_DIR', os.path.join(BASE_DIR, 'models'))
LOG_DIR = os.getenv('GALAXIESML_LOG_DIR', os.path.join(BASE_DIR, 'logs'))

# Create directories if they don't exist
for directory in [DATA_DIR, MODEL_DIR, LOG_DIR]:
    os.makedirs(directory, exist_ok=True)

# Dataset paths
def get_dataset_paths(image_size=127):
    """Get dataset paths based on image size."""
    return {
        'train': os.path.join(DATA_DIR, f'5x{image_size}x{image_size}_training_with_morphology.hdf5'),
        'val': os.path.join(DATA_DIR, f'5x{image_size}x{image_size}_validation_with_morphology.hdf5'),
        'test': os.path.join(DATA_DIR, f'5x{image_size}x{image_size}_testing_with_morphology.hdf5')
    }

# GPU Settings
GPU_MEMORY_LIMIT = int(os.getenv('GALAXIESML_GPU_MEMORY', '15'))  # in GB

# Model Parameters
MODEL_PARAMS = {
    'image_shape': (127, 127, 5),
    'num_dense_units': 200,
    'batch_size': 256,
    'learning_rate': 0.0001,
    'z_max': 4
}

# Training Parameters
TRAINING_PARAMS = {
    'base_epochs': 200,     # for basic CNN
    'improved_epochs': 500,  # for improved CNN
    'bayesian_epochs': 200  # for bayesian CNN
}

# Dataset Parameters
REQUIRED_DATASETS = [
    'image',
    'specz_redshift',
    'object_id'
]

def validate_hdf5_datasets(file_path):
    """Validate that an HDF5 file contains all required datasets."""
    import h5py
    
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Dataset file not found: {file_path}")
        
    with h5py.File(file_path, 'r') as f:
        missing = [dataset for dataset in REQUIRED_DATASETS if dataset not in f]
        if missing:
            raise ValueError(f"Missing required datasets in {file_path}: {', '.join(missing)}")
            
    return True

def setup_gpu():
    """Configure GPU memory limit."""
    import tensorflow as tf
    
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            tf.config.experimental.set_virtual_device_configuration(
                gpus[0],
                [tf.config.experimental.VirtualDeviceConfiguration(GPU_MEMORY_LIMIT*1000)])
            logical_gpus = tf.config.experimental.list_logical_devices('GPU')
            print(f"Using {len(gpus)} Physical GPUs, {len(logical_gpus)} Logical GPUs")
            print(f"GPU memory limit set to {GPU_MEMORY_LIMIT} GB")
        except RuntimeError as e:
            print(f"GPU setup error: {e}")
            print("Falling back to CPU") 