import os
import shutil
import torch
import torch.optim as optim
import torch.nn.functional as F
from torchvision import transforms
import torchvision.utils as vutils
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt

from tqdm import tqdm
from parameters import HR_DIR, VAL_DIR, HR_IMAGE_SIZE, LR_IMAGE_SIZE, DEVICE, NUM_EPOCHS, BATCH_SIZE, LEARNING_RATE, BETA1, BETA2, ALPHA
from utils.prepare_dataset import PairedDataset
from utils.eval_metrics import calculate_psnr, calculate_ssim
from gan_models import Generator, Discriminator
from losses import GeneratorLoss, DiscriminatorLoss
from gan_train import train_sr_gan

# Override parameters if needed
NUM_EPOCHS = 100  # additional epochs to train
LEARNING_RATE = 1e-4

# Setup directories
CHECKPOINT_DIR = "./checkpoints"
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

LOG_DIR = "./runs/srgan_experiment"
os.makedirs(LOG_DIR, exist_ok=True)

RESULTS_DIR = "./results"
os.makedirs(RESULTS_DIR, exist_ok=True)

writer = SummaryWriter(log_dir=LOG_DIR)

# Initialize dataloaders
train_dataset = PairedDataset(hr_dir=HR_DIR, hr_image_size=HR_IMAGE_SIZE, lr_image_size=LR_IMAGE_SIZE)
train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_dataset = PairedDataset(hr_dir=VAL_DIR, hr_image_size=HR_IMAGE_SIZE, lr_image_size=LR_IMAGE_SIZE)
val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

# Initialize models, losses, optimizers, schedulers
generator = Generator().to(DEVICE)
discriminator = Discriminator(HR_IMAGE_SIZE).to(DEVICE)
generator_loss = GeneratorLoss(alpha=ALPHA).to(DEVICE)
discriminator_loss = DiscriminatorLoss().to(DEVICE)

generator_optimizer = optim.Adam(generator.parameters(), lr=LEARNING_RATE, betas=(BETA1, BETA2))
discriminator_optimizer = optim.Adam(discriminator.parameters(), lr=LEARNING_RATE, betas=(BETA1, BETA2))

generator_scheduler = torch.optim.lr_scheduler.StepLR(generator_optimizer, step_size=30, gamma=0.1)
discriminator_scheduler = torch.optim.lr_scheduler.StepLR(discriminator_optimizer, step_size=30, gamma=0.1)

# Optionally resume training from a checkpoint
model_name = "srgan_epoch_50.pth"       # change the path if needed
resume_checkpoint = os.path.join(CHECKPOINT_DIR, model_name)  
init_epoch = 0
if os.path.exists(resume_checkpoint):
    checkpoint = torch.load(resume_checkpoint)
    generator.load_state_dict(checkpoint["generator_state_dict"])
    discriminator.load_state_dict(checkpoint["discriminator_state_dict"])
    generator_optimizer.load_state_dict(checkpoint["generator_optimizer_state_dict"])
    discriminator_optimizer.load_state_dict(checkpoint["discriminator_optimizer_state_dict"])
    init_epoch = checkpoint["epoch"] + 1
    print(f"Resumed from checkpoint: {resume_checkpoint} at epoch {init_epoch}")
else:
    print("No checkpoint found, starting from scratch.")

if __name__ == "__main__":
    import warnings
    warnings.filterwarnings("ignore")
    
    # Clean up GPU memory
    torch.cuda.empty_cache()

    print("Starting SRGAN retraining...")
    total_epochs = init_epoch + NUM_EPOCHS
    print(f"Training for {total_epochs} total epochs (resuming at epoch {init_epoch+1})")
    print(f"High-res images: {len(os.listdir(HR_DIR))}, Size: {HR_IMAGE_SIZE}")
    print(f"Low-res images: {len(os.listdir(HR_DIR))}, Size: {LR_IMAGE_SIZE}")
    print(f"Training on: {DEVICE}")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Log directory: {LOG_DIR}")
    print(f"Checkpoints directory: {CHECKPOINT_DIR}")
    print()
    
    # Continue training from init_epoch until total_epochs
    gen_losses, disc_losses, eval_epochs, psnr_scores, ssim_scores = train_sr_gan(
        generator, discriminator, generator_loss, discriminator_loss,
        HR_DIR, VAL_DIR, start_epoch=init_epoch, num_epochs=total_epochs
    )
    writer.close()
    
    import matplotlib
    matplotlib.use("Agg")
    os.makedirs("graphs", exist_ok=True)
    
    # Save Generator Loss graph
    plt.figure()
    plt.plot(range(init_epoch+1, total_epochs+1), gen_losses, label="Generator Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Generator Loss per Epoch")
    plt.grid(True)
    plt.legend()
    plt.savefig(f"graphs/generator_loss_{total_epochs}.png")
    plt.close()
    
    # Save Discriminator Loss graph
    plt.figure()
    plt.plot(range(init_epoch+1, total_epochs+1), disc_losses, label="Discriminator Loss", color="orange")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Discriminator Loss per Epoch")
    plt.grid(True)
    plt.legend()
    plt.savefig(f"graphs/discriminator_loss_{total_epochs}.png")
    plt.close()
    
    # Save PSNR graph
    plt.figure()
    plt.plot(eval_epochs, psnr_scores, marker="o", label="PSNR", color="green")
    plt.xlabel("Epoch")
    plt.ylabel("PSNR")
    plt.title("PSNR During Evaluation")
    plt.grid(True)
    plt.legend()
    plt.savefig(f"graphs/psnr_{total_epochs}.png")
    plt.close()
    
    # Save SSIM graph
    plt.figure()
    plt.plot(eval_epochs, ssim_scores, marker="o", label="SSIM", color="red")
    plt.xlabel("Epoch")
    plt.ylabel("SSIM")
    plt.title("SSIM During Evaluation")
    plt.grid(True)
    plt.legend()
    plt.savefig(f"graphs/ssim_{total_epochs}.png")
    plt.close()
    
    print("\nGraphs saved in the 'graphs' folder.\n")
    torch.cuda.empty_cache()