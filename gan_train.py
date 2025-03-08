import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter

from tqdm import tqdm
from gan_models import Generator, Discriminator
from utils.prepare_dataset import PairedDataset
from losses import GeneratorLoss, DiscriminatorLoss
from utils.eval_metrics import calculate_psnr, calculate_ssim
from parameters import *


# Directory to save checkpoints and logs
CHECKPOINT_DIR = "./checkpoints"
LOG_DIR = "./runs/srgan_experiment"
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
writer = SummaryWriter(log_dir=LOG_DIR)

# Define training loop for SRGAN
def train_sr_gan(hr_dir):
    # Initialize models, losses, and optimizers
    generator = Generator().to(DEVICE)
    discriminator = Discriminator().to(DEVICE)
    generator_loss = GeneratorLoss(alpha=ALPHA).to(DEVICE)
    discriminator_loss = DiscriminatorLoss().to(DEVICE)
    generator_optimizer = optim.Adam(generator.parameters(), lr=LEARNING_RATE, betas=(BETA1, BETA2))
    discriminator_optimizer = optim.Adam(discriminator.parameters(), lr=LEARNING_RATE, betas=(BETA1, BETA2))
    
    # Learning rate schedulers
    generator_scheduler = torch.optim.lr_scheduler.StepLR(generator_optimizer, step_size=30, gamma=0.1)
    discriminator_scheduler = torch.optim.lr_scheduler.StepLR(discriminator_optimizer, step_size=30, gamma=0.1)
    
    # Initialize dataloader
    dataset = PairedDataset(hr_dir, hr_image_size=HR_IMAGE_SIZE, lr_image_size=LR_IMAGE_SIZE)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    global_step = 0

    for epoch in range(NUM_EPOCHS):
        generator.train()
        discriminator.train()
        
        for batch in tqdm(dataloader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS}"):
            hr_batch = batch["hr"].to(DEVICE)
            lr_batch = batch["lr"].to(DEVICE)
            
            # Train discriminator
            discriminator_optimizer.zero_grad()
            fake_hr = generator(lr_batch)
            real_preds = discriminator(hr_batch)
            fake_preds = discriminator(fake_hr.detach())
            d_loss = discriminator_loss(real_preds, fake_preds)
            d_loss.backward()
            discriminator_optimizer.step()
            
            # Train generator
            generator_optimizer.zero_grad()
            fake_hr = generator(lr_batch)
            fake_preds = discriminator(fake_hr)
            g_loss = generator_loss(fake_hr, hr_batch, fake_preds)
            g_loss.backward()
            generator_optimizer.step()
            
            # Log losses
            writer.add_scalar("Loss/Discriminator", d_loss.item(), global_step)
            writer.add_scalar("Loss/Generator", g_loss.item(), global_step)
            global_step += 1
        
        # Update learning rates
        generator_scheduler.step()
        discriminator_scheduler.step()
        
        # Evaluation
        if (epoch + 1) % 5 == 0:  # Evaluate every 5 epochs
            generator.eval()
            with torch.no_grad():
                psnr_values = []
                ssim_values = []
                for batch in dataloader:
                    hr_batch = batch["hr"].to(DEVICE)
                    lr_batch = batch["lr"].to(DEVICE)
                    fake_hr = generator(lr_batch)
                    for i in range(hr_batch.size(0)):
                        psnr_values.append(calculate_psnr(hr_batch[i], fake_hr[i]))
                        ssim_values.append(calculate_ssim(hr_batch[i], fake_hr[i]))
                avg_psnr = torch.tensor(psnr_values).mean().item()
                avg_ssim = torch.tensor(ssim_values).mean().item()
            
            writer.add_scalar("Eval/PSNR", avg_psnr, epoch)
            writer.add_scalar("Eval/SSIM", avg_ssim, epoch)
            print(f"Epoch [{epoch+1}/{NUM_EPOCHS}] - Avg. PSNR: {avg_psnr:.2f}, Avg. SSIM: {avg_ssim:.4f}")
        
        # Save checkpoint
        if (epoch + 1) % 100 == 0:
            checkpoint_path = os.path.join(CHECKPOINT_DIR, f"srgan_epoch_{epoch+1}.pth")
            torch.save({
                "generator_state_dict": generator.state_dict(),
                "discriminator_state_dict": discriminator.state_dict(),
                "generator_optimizer_state_dict": generator_optimizer.state_dict(),
                "discriminator_optimizer_state_dict": discriminator_optimizer.state_dict(),
                "epoch": epoch,
            }, checkpoint_path)
            print(f"Checkpoint saved: {checkpoint_path}")


if __name__ == "__main__":
    import warnings
    warnings.filterwarnings("ignore")

    hr_dir = "./data/High_Res_Images"

    print("Starting SRGAN training...")
    print(f"Training for {NUM_EPOCHS} epochs")
    print(f"High-res images: {len(os.listdir(hr_dir))}, Size: {HR_IMAGE_SIZE}")
    print(f"Low-res images: {len(os.listdir(hr_dir))}, Size: {LR_IMAGE_SIZE}") 
    print(f"Training On: {DEVICE}")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Log directory: {LOG_DIR}")
    print(f"Checkpoints directory: {CHECKPOINT_DIR}")
    print()

    train_sr_gan(hr_dir)
    writer.close()