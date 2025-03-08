import torch

# Define all hyperparameters here
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 16
HR_IMAGE_SIZE = (64, 64)
LR_IMAGE_SIZE = (16, 16)
NUM_EPOCHS = 1000
LEARNING_RATE = 1e-4
BETA1 = 0.5
BETA2 = 0.999
ALPHA = 0.001