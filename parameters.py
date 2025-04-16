import torch

# Define all hyperparameters here
HR_DIR = "./train_samples/HR_images"
VAL_DIR = "./val_images/HR_images"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 16
HR_IMAGE_SIZE = (256, 256)
LR_IMAGE_SIZE = (64, 64)
NUM_EPOCHS = 100
LEARNING_RATE = 0.00008
BETA1 = 0.5
BETA2 = 0.999
ALPHA = 0.001