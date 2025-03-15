# ResNet Model for CIFAR-10 Classification

This repository contains our implementation for the Deep Learning Spring 2025 Project 1, where we develop a modified ResNet architecture to achieve high accuracy on the CIFAR-10 dataset while staying under 5 million parameters.


## Project Overview

We've implemented a Wide ResNet variant that balances model size and performance on the CIFAR-10 image classification task. Our model:
- Uses a parameter-efficient ResNet design with 3.38 million parameters
- Implements residual blocks with batch normalization
- Incorporates modern training techniques including mixup and label smoothing

## Repository Structure

- `train.py`: Script to train the model from scratch
- `predict.py`: Script to generate predictions on test data
- `best_model.pth`: Our best performing model weights
- `requirements.txt`: Required dependencies
- `README.md`: Project documentation

## Model Architecture

Our model is a WideResNet with:
- 3 stages of residual blocks with [3, 3, 3] layers per stage
- Width multiplier of 1.0 to control model size
- Dropout rate of 0.2 for regularization
- Total parameters: ~3.38 million

## Training Methodology

We used the following training strategy:
- SGD optimizer with momentum 0.9
- Cosine annealing learning rate schedule
- Initial learning rate of 0.1
- Weight decay of 5e-4
- 500 epochs of training
- Mixup data augmentation with alpha=0.2
- Label smoothing with value 0.1

## Data Augmentation

For training data, we applied multiple augmentation techniques:
- Random cropping with padding
- Random horizontal flips
- Random rotations
- Color jittering
- Random erasing

## Results

Our model achieves:
- Validation accuracy: ~99%
- Test accuracy: 88.913%

## How to Run

### Setup
```bash
pip install -r requirements.txt
```

### Training
```bash
python train.py
```

### Generating Predictions
```bash
python predict.py
```

## Requirements
- PyTorch
- NumPy
- Matplotlib
- Pandas
- PIL
- torchsummary

## Citation
Our implementation was inspired by:
- [ResNet paper](https://arxiv.org/abs/1512.03385)
- [Wide ResNet paper](https://arxiv.org/abs/1605.07146)