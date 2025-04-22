import torch

# Define all hyperparameters here
HR_DIR = "./data/HR_images"
VAL_DIR = "./val_images/HR_images"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 16
STEP_SIZE = 30
HR_IMAGE_SIZE = (256, 256)
LR_IMAGE_SIZE = (64, 64)
NUM_EPOCHS = 100
LEARNING_RATE = 1e-4
BETA1 = 0.5
BETA2 = 0.999
ALPHA = 1e-3  # Weight for adversarial loss
BETA = 5e-1   # Weight for pixel-wise loss