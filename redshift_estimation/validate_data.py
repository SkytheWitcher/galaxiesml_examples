import tensorflow as tf
import tensorflow_probability as tfp
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import h5py
import keras
import os
import tensorboard
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout, Input, Concatenate
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import StandardScaler
from tensorboard.plugins.hparams import api as hp
import argparse
import sys

# Add the parent directory to the Python path so we can import config.py
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from typing import Dict, List, Tuple
from photoz_utils import *
from DataMakerPlus import *
from config import get_dataset_paths

def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Validate HDF5 datasets for redshift estimation')
    parser.add_argument('--image_size', type=int, default=127,
                      help='Size of input images to validate (default: 127)')
    return parser.parse_args()

def validate_hdf5_structure(file_path: str) -> Tuple[bool, List[str]]:
    """
    Validate the structure and contents of an HDF5 file.
    
    Args:
        file_path: Path to the HDF5 file
        
    Returns:
        Tuple of (is_valid, error_messages)
    """
    required_keys = {
        'image': (5, None, None),  # (channels, height, width)
        'specz_redshift': None,    # 1D array
        'object_id': None,         # 1D array
        'g_cmodel_mag': None,      # 1D array
        'r_cmodel_mag': None,
        'i_cmodel_mag': None,
        'z_cmodel_mag': None,
        'y_cmodel_mag': None
    }
    
    errors = []
    
    try:
        with h5py.File(file_path, 'r') as f:
            # Check for required keys
            missing_keys = set(required_keys.keys()) - set(f.keys())
            if missing_keys:
                errors.append(f"Missing required keys: {missing_keys}")
            
            # Validate shapes
            if 'image' in f:
                shape = f['image'].shape
                if len(shape) != 4:  # (samples, channels, height, width)
                    errors.append(f"Image data should be 4D, got shape {shape}")
                elif shape[1] != 5:
                    errors.append(f"Expected 5 channels, got {shape[1]}")
                elif shape[2] != shape[3]:
                    errors.append(f"Image height and width should be equal, got {shape[2]}x{shape[3]}")
            
            # Check all 1D arrays have same length
            if not errors:
                lengths = {k: len(f[k]) for k in f.keys() if k != 'image'}
                if len(set(lengths.values())) > 1:
                    errors.append(f"Inconsistent lengths across arrays: {lengths}")
                
                # Check image samples match other arrays
                if 'image' in f and f['image'].shape[0] != list(lengths.values())[0]:
                    errors.append(f"Image samples ({f['image'].shape[0]}) don't match other arrays ({list(lengths.values())[0]})")
            
            # Validate data ranges
            if 'specz_redshift' in f:
                redshifts = f['specz_redshift'][:]
                if np.any((redshifts < 0) | (redshifts > 10)):
                    errors.append("Found redshift values outside expected range [0, 10]")
            
            # Validate magnitudes
            for band in ['g', 'r', 'i', 'z', 'y']:
                key = f'{band}_cmodel_mag'
                if key in f:
                    mags = f[key][:]
                    if np.any((mags < 0) | (mags > 50)):
                        errors.append(f"Found {band}-band magnitudes outside expected range [0, 50]")
    
    except Exception as e:
        errors.append(f"Error reading file: {str(e)}")
    
    return len(errors) == 0, errors

def main():
    """Main function to validate all datasets."""
    args = parse_args()
    
    # Get dataset paths from config
    try:
        dataset_paths = get_dataset_paths(args.image_size)
    except Exception as e:
        print(f"Error getting dataset paths: {str(e)}")
        print("Please ensure GALAXIESML_DATA_DIR is set correctly")
        sys.exit(1)
    
    # Validate each dataset
    all_valid = True
    for split, path in dataset_paths.items():
        print(f"\nValidating {split} dataset: {path}")
        
        if not os.path.exists(path):
            print(f"❌ File not found!")
            all_valid = False
            continue
            
        is_valid, errors = validate_hdf5_structure(path)
        
        if is_valid:
            print(f"✓ Valid HDF5 structure")
        else:
            print("❌ Validation failed:")
            for error in errors:
                print(f"  - {error}")
            all_valid = False
    
    if all_valid:
        print("\n✓ All datasets are valid and ready for training!")
        sys.exit(0)
    else:
        print("\n❌ Some datasets failed validation. Please fix the errors above.")
        sys.exit(1)

if __name__ == '__main__':
    main()