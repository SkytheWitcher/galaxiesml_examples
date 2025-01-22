# GalaxiesML Examples

GalaxiesML is a dataset for use in machine learning in astronomy. This repository contains examples of how the GalaxiesML dataset can be used for photometric redshift estimation.

The dataset is publicly available on **Zenodo** with the DOI: **[10.5281/zenodo.11117528](https://doi.org/10.5281/zenodo.11117528)**.

## Installation

### Prerequisites
- Python 3.8 or higher
- NVIDIA GPU with 8+ GB memory
- CUDA 11.2 or higher
- cuDNN 8.1 or higher

1. Clone this repository:
```bash
git clone https://github.com/username/galaxiesml_examples.git
cd galaxiesml_examples
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Configure environment variables (optional):
```bash
export GALAXIESML_DATA_DIR=/path/to/your/data
export GALAXIESML_MODEL_DIR=/path/to/store/models
export GALAXIESML_LOG_DIR=/path/to/store/logs
export GALAXIESML_GPU_MEMORY=15  # GPU memory limit in GB
```

### Configuration System

The project uses a centralized configuration system through `config.py`:

1. **Environment Variables**:
   ```bash
   # Required - Set these before running any scripts
   export GALAXIESML_DATA_DIR=/path/to/your/data    # Where your HDF5 files are stored
   export GALAXIESML_MODEL_DIR=/path/to/store/models # Where models and checkpoints are saved
   export GALAXIESML_LOG_DIR=/path/to/store/logs     # Where training logs are stored
   export GALAXIESML_GPU_MEMORY=15                   # GPU memory limit in GB
   ```

2. **Model Parameters**:
   All model hyperparameters are defined in `config.py`:
   - Image shape and sizes
   - Learning rates
   - Batch sizes
   - Dense layer units
   - GPU memory settings

3. **Dataset Paths**:
   Instead of modifying paths in each script, use the configuration system:
   ```python
   from config import get_dataset_paths
   
   # Automatically gets paths based on GALAXIESML_DATA_DIR
   dataset_paths = get_dataset_paths(image_size=127)
   train_path = dataset_paths['train']
   val_path = dataset_paths['val']
   test_path = dataset_paths['test']
   ```

4. **GPU Setup**:
   GPU memory is configured automatically:
   ```python
   from config import setup_gpu
   setup_gpu()  # Uses GALAXIESML_GPU_MEMORY value
   ```

## Dataset Setup

1. Download the dataset from [Zenodo](https://doi.org/10.5281/zenodo.11117528)
2. Extract the downloaded files
3. Place the HDF5 files in your data directory (specified by `GALAXIESML_DATA_DIR`):
```
$GALAXIESML_DATA_DIR/
├── 5x127x127_training_with_morphology.hdf5
├── 5x127x127_validation_with_morphology.hdf5
└── 5x127x127_testing_with_morphology.hdf5
```

The configuration system will automatically find your datasets using the `GALAXIESML_DATA_DIR` environment variable.

## Models

### Redshift Estimation

The repository includes several iterations of Convolutional Neural Networks (CNNs) for redshift estimation:

1. **Basic CNN (train_cnn.py)**
   - Initial implementation with standard CNN architecture
   - Uses both image data (5x127x127) and photometric features
   - Training time: ~4-6 hours on NVIDIA V100
   - Memory requirement: ~8GB GPU memory
   - Best for: Initial experimentation and baseline results

2. **CNN v2 (train_cnn_v2.py)**
   - Improved architecture with dropout layers
   - Better regularization to prevent overfitting
   - Extended training (500 epochs vs 200)
   - Training time: ~8-10 hours on NVIDIA V100
   - Memory requirement: ~10GB GPU memory
   - Best for: Production use and better accuracy

3. **CNN v3 (train_cnn_v3.py)**
   - Implements probabilistic outputs using Bayesian neural networks
   - Provides uncertainty estimates with predictions
   - Training time: ~12-15 hours on NVIDIA V100
   - Memory requirement: ~12GB GPU memory
   - Best for: Research applications requiring uncertainty quantification

### Neural Network (train_nn.py)
- Fully connected neural network using only photometric features
- Training time: ~1-2 hours on NVIDIA V100
- Memory requirement: ~4GB GPU memory
- Best for: Quick experiments or when image data isn't available

## Model Training

1. Validate your data:
```bash
# Validates all datasets in GALAXIESML_DATA_DIR
python redshift_estimation/validate_data.py
```

2. Train a model:
```bash
# Basic CNN
python redshift_estimation/train_cnn.py --model_name my_model --image_size 127

# Improved CNN with better regularization
python redshift_estimation/train_cnn_v2.py --model_name my_model_v2 --image_size 127

# Probabilistic CNN with uncertainty estimation
python redshift_estimation/train_cnn_v3.py --model_name my_model_v3 --image_size 127

# Simple neural network (photometry only)
python redshift_estimation/train_nn.py --model_name my_model_nn
```

3. Evaluate results:
```bash
python redshift_estimation/evaluate_cnn.py --model_name my_model --image_size 127
```

All models support these common arguments:
- `--model_name`: Name of your model (used for saving checkpoints and logs)
- `--image_size`: Size of input images (default: 127)
- `--batch_size`: Override the default batch size from config.py

## Pre-trained Models

Pre-trained model weights are not included in this repository. You can either:
1. Train your own models using the provided scripts
2. Download pre-trained weights from [link to be provided]

## Directory Structure

```
galaxiesml_examples/
├── config.py           # Configuration settings
├── requirements.txt    # Python dependencies
├── redshift_estimation/
│   ├── train_cnn.py   # Basic CNN implementation
│   ├── train_cnn_v2.py # Improved CNN
│   ├── train_cnn_v3.py # Probabilistic CNN
│   ├── train_nn.py    # Fully connected NN
│   ├── evaluate_*.py  # Evaluation scripts
│   └── photoz_utils.py # Utility functions
├── data/              # Directory for datasets
├── models/            # Saved model weights
└── logs/              # Training logs
```

## Troubleshooting

Common issues and solutions:

1. **Environment Setup**
   - Ensure all environment variables are set correctly:
     ```bash
     echo $GALAXIESML_DATA_DIR    # Should point to your data directory
     echo $GALAXIESML_MODEL_DIR   # Should point to your models directory
     echo $GALAXIESML_LOG_DIR     # Should point to your logs directory
     echo $GALAXIESML_GPU_MEMORY  # Should show your GPU memory limit
     ```
   - Check that directories exist and have write permissions

2. **GPU Memory Errors**
   - Lower `GALAXIESML_GPU_MEMORY` environment variable
   - Reduce batch size using `--batch_size` argument
   - Use a GPU with more memory

3. **Missing Datasets**
   - Ensure HDF5 files are in `$GALAXIESML_DATA_DIR`
   - Check file names match expected patterns
   - Run `validate_data.py` to verify dataset format

4. **Training Issues**
   - Start with `train_nn.py` for quick validation
   - Monitor GPU memory usage with `nvidia-smi`
   - Check TensorBoard logs in `$GALAXIESML_LOG_DIR`

## Hardware Requirements

- Recommended: NVIDIA GPU with 8+ GB memory (V100 or better for full training)
- GPU memory usage can be configured via `GALAXIESML_GPU_MEMORY` environment variable
- Can run on CPU but training will be significantly slower (>10x longer)
- Minimum 32GB system RAM recommended

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.


