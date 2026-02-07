# Neural Network Implementation

A pure Python implementation of a Convolutional Neural Network (CNN) from scratch using NumPy, designed for Fashion-MNIST classification.

## Features

- Custom CNN implementation with Conv2D, ReLU, MaxPool2D, and Dense layers
- RKNN (Rockchip Neural Network) optimized convolution operator with float16 precision
- Manual backpropagation implementation
- Training on Fashion-MNIST dataset

## Requirements

This project works on both **Debian** and **Ubuntu** (and other Linux distributions), as well as macOS and Windows.

### System Requirements

- Python 3.6 or higher
- pip (Python package manager)

### Python Dependencies

- NumPy >= 1.19.0

## Installation

### On Debian/Ubuntu

```bash
# Update package list
sudo apt update

# Install Python 3 and pip (if not already installed)
sudo apt install python3 python3-pip

# Install required Python packages
pip3 install -r requirements.txt
```

### On other systems

```bash
# Install required Python packages
pip install -r requirements.txt
```

## Dataset

The project requires Fashion-MNIST dataset in CSV format:
- `fashion-mnist_train.csv` - Training data (required for training)
- `fashion-mnist_test.csv` - Test data (included in repository)

You can download the Fashion-MNIST dataset from [Kaggle](https://www.kaggle.com/zalando-research/fashionmnist) or other sources.

## Usage

### Testing Platform Compatibility

To verify the code works correctly on your system (Debian/Ubuntu or other):

```bash
python3 test_compatibility.py
```

This test script verifies that all components work correctly without requiring the training dataset.

### Running the CNN Training

**Note:** You need to download `fashion-mnist_train.csv` first (see Dataset section above).

```bash
python3 CNN.py
```

This will:
1. Load the training and test data
2. Initialize a simple CNN model
3. Train on 1000 samples from the training set
4. Evaluate on 1000 test samples

### Using the RKNN Convolution Operator

```bash
python3 rknn_conv_operator.py
```

This demonstrates the RKNN-optimized float16 convolution operator.

## Model Architecture

The SimpleCNN model consists of:
- Conv2D layer (1→8 channels, 3×3 kernel)
- ReLU activation
- MaxPool2D (2×2)
- Conv2D layer (8→16 channels, 3×3 kernel)
- ReLU activation
- MaxPool2D (2×2)
- Flatten
- Dense layer (400→128)
- ReLU activation
- Dense layer (128→10)
- Softmax

## RKNN Convolution Operator

The `rknn_conv_operator.py` module provides:
- `RKNNConvolutionOperator`: Standard convolution with float16 precision
- `RKNNDepthwiseConvolutionOperator`: Depthwise convolution variant
- Support for padding, stride, dilation, and grouped convolutions
- Optimized for RKNN framework deployment

## Platform Compatibility

This project is platform-agnostic and runs on:
- ✅ Debian (all versions)
- ✅ Ubuntu (all versions)
- ✅ Other Linux distributions
- ✅ macOS
- ✅ Windows

The code uses only standard Python libraries (NumPy) and does not depend on any OS-specific features.

## License

Open source - feel free to use and modify.
