import torch

# Define all hyperparameters here
HR_DIR = "./train_samples/HR_images"
VAL_DIR = "./val_images/HR_images"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 32
HR_IMAGE_SIZE = (96, 96)
LR_IMAGE_SIZE = (24, 24)
NUM_EPOCHS = 100
LEARNING_RATE = 1e-4
BETA1 = 0.5
BETA2 = 0.999
ALPHA = 0.001