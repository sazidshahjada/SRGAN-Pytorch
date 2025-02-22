# SRGAN-PyTorch

This repository implements a Super-Resolution Generative Adversarial Network (SRGAN) using PyTorch. Follow the instructions below to train the model and run inference on your images.

## Table of Contents

- [Requirements](#requirements)
- [Installation](#installation)
- [Dataset Preparation](#dataset-preparation)
- [Training](#training)
- [Testing](#testing)
- [Single Image Inference](#single-image-inference)
- [Directory Structure](#directory-structure)
- [License](#license)
- [Acknowledgements](#acknowledgements)

## Requirements

- Python 3.x
- [PyTorch](https://pytorch.org/get-started/previous-versions/)
- Additional dependencies listed in [requirements.txt](requirements.txt)

## Installation

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd SRGAN-PyTorch
   ```
2. Install the dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Dataset Preparation

1. **High-resolution images:**
   - Place your high-resolution images in the following directory:
   ```bash
   data/High_Res_Images
   ```

2. **Generating Low-resolution Images (Optional):**
   - Downscale the high-resolution images to create low-resolution counterparts:
   ```bash
   python3 downsample_highres.py
   ```

3. **Ensure Correct Directory Structure:**
   Before starting training, verify that your dataset directories are correctly set up.

## Training

To start training the SRGAN model, run:

```bash
python gan_train.py
```

- The training script utilizes images from the dataset directories.
- Model checkpoints are saved in the `./checkpoints` directory.
- Training logs for TensorBoard are stored in the `./runs` directory.

## Testing

To evaluate the trained SRGAN model, run:

```bash
python gan_test.py --hr_dir <path-to-high-res-folder> --lr_dir <path-to-low-res-folder> --model <path-to-checkpoint>
```

- The test script loads the trained model and evaluates it on the test dataset.

## Single Image Inference

To super-resolve a single image, use the following command:

```bash
python single_image_inference.py --image_path <path-to-image>
```

- The output will be saved in the `./outputs` directory.

## Directory Structure

Ensure your directory structure is as follows:

```
SRGAN-PyTorch/
├── checkpoints/                # Model checkpoint files (saved during training)
├── data/                       # Directory for high-resolution images
├── outputs/                    # Generated super-resolved images
├── pretrained/                 # Pretrained model files (if any)
├── runs/                       # TensorBoard logs
├── samples/                    # Sampled images for training/inference
├── test_images/                # Images used for single image inference/testing
├── utils/                      # Utility scripts (dataset preparation, evaluation metrics, etc.)
├── gan_models.py               # Definition of Generator, Discriminator, and feature extractor
├── losses.py                   # Loss functions for training
├── gan_train.py                # Training script
├── gan_test.py                 # Test script for evaluating model quality
├── single_image_inference.py   # Inference script for super-resolving a single image
├── requirements.txt            # Python dependencies
└── README.md                   # This readme file
```

## License

This project is licensed under the MIT License.

## Acknowledgements

- The SRGAN architecture is based on [Photo-Realistic Single Image Super-Resolution Using a Generative Adversarial Network](https://arxiv.org/abs/1609.04802) by Ledig et al.
- This implementation is inspired by various open-source SRGAN projects available online.

