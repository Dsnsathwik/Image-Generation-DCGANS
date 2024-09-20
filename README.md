# DCGAN Image Generator

A deep learning project utilizing Deep Convolutional Generative Adversarial Networks (DCGANs) to generate realistic images from random noise. This repository provides a complete implementation of DCGANs using PyTorch, with customizable settings for training and image generation.

## Table of Contents

- Project Overview
- Features
- Setup
- Usage
- Training
- Results
- Datasets
- Customization
- Contributing
- License

# Project Overview

Generative Adversarial Networks (GANs) are a powerful class of models used to generate new data samples similar to a given dataset. DCGANs, which apply convolutional layers, are widely known for producing high-quality images, especially when trained on image datasets. This project implements DCGANs using PyTorch to generate synthetic images.

## Features

- DCGAN Architecture: Implements both Generator and Discriminator networks using convolutional layers.
- Image Generation: Random noise vectors are transformed into realistic images.
- Customizable Training Pipeline: Easily configure the training parameters (epochs, learning rate, batch size).
- Real-time Image Visualization: View generated images at different stages of training.
- Model Saving: Periodically save models during training for reuse.

# Setup
## Prerequisites
Make sure you have Python 3.x installed, along with the following libraries:

- PyTorch
- NumPy
- Matplotlib
- TorchVision
- tqdm

You can install the necessary dependencies with the following command:

```bash
pip install -r requirements.txt
```
## Clone the Repository

```bash
git clone https://github.com/your-username/dcgan-image-generator.git
cd dcgan-image-generator
```

# Usage
## Training
To start training the DCGAN on a dataset, run the following command:

```bash
python train.py --dataset <dataset-path> --epochs <number-of-epochs>
```

You can adjust parameters like batch size and learning rate in the config.json file or directly from the command line:

```bash
python train.py --batch_size 64 --learning_rate 0.0002 --image_size 64
```

# Image Generation
Once the model is trained, you can generate new images using the trained generator:

```bash
python generate.py --model_path <path-to-saved-model>
```

Generated images will be saved in the output/ directory by default.

# Results
As the training progresses, you can monitor the loss curves and generated samples. Below is an example of generated images after several epochs of training on the CelebA dataset:

Training Progress
(Insert example of loss curves here)

Generated Images
(Insert example of generated images here)

# Datasets
You can use any image dataset with this implementation, including:

- MNIST
- CelebA
- CIFAR-10
To add your own dataset, place the images in a directory and provide the path using the --dataset option during training.

# Customization
You can modify the model architecture, hyperparameters, and other settings by editing the following files:

- config.json: Customize batch size, learning rate, and other hyperparameters.
- model.py: Modify the DCGAN Generator and Discriminator network architectures.
- train.py: Update training logic or add custom logging/visualization methods.

# Contributing
Contributions are welcome! If you'd like to improve this project or add new features, feel free to submit a pull request. Please ensure your code follows the project’s coding standards and is well-documented.

1. Fork the repository.
2. Create your feature branch (git checkout -b feature/new-feature).
3. Commit your changes (git commit -am 'Add new feature').
4. Push to the branch (git push origin feature/new-feature).
5. Open a pull request.

# License
This project is licensed under the [MIT](https://choosealicense.com/licenses/mit/) License - see the LICENSE file for details.
