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
from parameters import *
from gan_models import Generator, Discriminator
from utils.prepare_dataset import PairedDataset
from losses import GeneratorLoss, DiscriminatorLoss
from utils.eval_metrics import calculate_psnr, calculate_ssim

# Setup directories
CHECKPOINT_DIR = "./checkpoints"
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
for file in os.listdir(CHECKPOINT_DIR):
    os.remove(os.path.join(CHECKPOINT_DIR, file))

LOG_DIR = "./runs/srgan_experiment"
os.makedirs(LOG_DIR, exist_ok=True)
for file in os.listdir(LOG_DIR):
    os.remove(os.path.join(LOG_DIR, file))
writer = SummaryWriter(log_dir=LOG_DIR)

RESULTS_DIR = "./results"
os.makedirs(RESULTS_DIR, exist_ok=True)
for entry in os.listdir(RESULTS_DIR):
    entry_path = os.path.join(RESULTS_DIR, entry)
    if os.path.isfile(entry_path):
        os.remove(entry_path)
    elif os.path.isdir(entry_path):
        shutil.rmtree(entry_path)

# Initialize models, losses, optimizers, and schedulers
generator = Generator().to(DEVICE)
discriminator = Discriminator().to(DEVICE)
generator_loss = GeneratorLoss(alpha=ALPHA).to(DEVICE)
discriminator_loss = DiscriminatorLoss().to(DEVICE)
generator_optimizer = optim.Adam(generator.parameters(), lr=LEARNING_RATE, betas=(BETA1, BETA2))
discriminator_optimizer = optim.Adam(discriminator.parameters(), lr=LEARNING_RATE, betas=(BETA1, BETA2))

generator_scheduler = torch.optim.lr_scheduler.StepLR(generator_optimizer, step_size=30, gamma=0.1)
discriminator_scheduler = torch.optim.lr_scheduler.StepLR(discriminator_optimizer, step_size=30, gamma=0.1)

# Initialize dataloaders
train_dataset = PairedDataset(hr_dir=HR_DIR, hr_image_size=HR_IMAGE_SIZE, lr_image_size=LR_IMAGE_SIZE)
train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_dataset = PairedDataset(hr_dir=VAL_DIR, hr_image_size=HR_IMAGE_SIZE, lr_image_size=LR_IMAGE_SIZE)
val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

def train_sr_gan(generator, discriminator, generator_loss, discriminator_loss, hr_dir, val_dir, num_epochs=NUM_EPOCHS):
    global_step = 0
    avg_gen_losses = []
    avg_disc_losses = []
    eval_epochs = []
    eval_psnr_list = []
    eval_ssim_list = []
    
    for epoch in range(num_epochs):
        generator.train()
        discriminator.train()
        epoch_g_loss_sum = 0.0
        epoch_d_loss_sum = 0.0
        num_batches = 0
        
        for batch in tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{num_epochs}"):
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
            
            # Log losses to TensorBoard and accumulate for plotting
            writer.add_scalar("Loss/Discriminator", d_loss.item(), global_step)
            writer.add_scalar("Loss/Generator", g_loss.item(), global_step)
            epoch_g_loss_sum += g_loss.item()
            epoch_d_loss_sum += d_loss.item()
            num_batches += 1
            global_step += 1
        
        # Compute average losses for the epoch
        avg_gen_loss = epoch_g_loss_sum / num_batches
        avg_disc_loss = epoch_d_loss_sum / num_batches
        avg_gen_losses.append(avg_gen_loss)
        avg_disc_losses.append(avg_disc_loss)
        
        # Print average losses at the end of the epoch
        print(f"Generator Loss: {avg_gen_loss:.4f}, Discriminator Loss: {avg_disc_loss:.4f}")
        
        # Update learning rate schedulers
        generator_scheduler.step()
        discriminator_scheduler.step()
        
        # Run evaluation every 10 epochs
        if (epoch + 1) % 10 == 0:
            generator.eval()
            epoch_results_dir = os.path.join(RESULTS_DIR, f"epoch_{epoch+1}")
            os.makedirs(epoch_results_dir, exist_ok=True)
            
            with torch.no_grad():
                psnr_values = []
                ssim_values = []
                for batch_idx, batch in enumerate(val_dataloader):
                    hr_batch = batch["hr"].to(DEVICE)
                    lr_batch = batch["lr"].to(DEVICE)
                    fake_hr = generator(lr_batch)
                    for i in range(hr_batch.size(0)):
                        psnr_values.append(calculate_psnr(hr_batch[i], fake_hr[i]))
                        ssim_values.append(calculate_ssim(hr_batch[i], fake_hr[i]))
                        
                        # Resize LR image to match HR dimensions before concatenation
                        lr_resized = F.interpolate(lr_batch[i].unsqueeze(0),
                                                   size=hr_batch[i].shape[-2:],
                                                   mode='bicubic',
                                                   align_corners=False)
                        # Concatenate LR, Generated HR, and Original HR images along width (dim=3)
                        result_image = torch.cat((lr_resized, fake_hr[i].unsqueeze(0), hr_batch[i].unsqueeze(0)), dim=3)
                        image_path = os.path.join(epoch_results_dir, f"image_{batch_idx * val_dataloader.batch_size + i + 1}.png")
                        vutils.save_image(result_image, image_path, normalize=True)
                
                avg_psnr = torch.tensor(psnr_values).mean().item()
                avg_ssim = torch.tensor(ssim_values).mean().item()
                
                writer.add_scalar("Eval/PSNR", avg_psnr, epoch)
                writer.add_scalar("Eval/SSIM", avg_ssim, epoch)
                print(f"\nEvaluation - Avg. PSNR: {avg_psnr:.2f}, Avg. SSIM: {avg_ssim:.4f}")
                print(f"Results saved in: {epoch_results_dir}\n")

                
                eval_epochs.append(epoch+1)
                eval_psnr_list.append(avg_psnr)
                eval_ssim_list.append(avg_ssim)
        
        # Save model checkpoint every 50 epochs
        if (epoch + 1) % 50 == 0:
            checkpoint_path = os.path.join(CHECKPOINT_DIR, f"srgan_epoch_{epoch+1}.pth")
            torch.save({
                "generator_state_dict": generator.state_dict(),
                "discriminator_state_dict": discriminator.state_dict(),
                "generator_optimizer_state_dict": generator_optimizer.state_dict(),
                "discriminator_optimizer_state_dict": discriminator_optimizer.state_dict(),
                "epoch": epoch,
            }, checkpoint_path)
            print(f"Checkpoint saved: {checkpoint_path}")
    
    return avg_gen_losses, avg_disc_losses, eval_epochs, eval_psnr_list, eval_ssim_list

if __name__ == "__main__":
    import warnings
    warnings.filterwarnings("ignore")

    print("Starting SRGAN training...")
    print(f"Training for {NUM_EPOCHS} epochs")
    print(f"High-res images: {len(os.listdir(HR_DIR))}, Size: {HR_IMAGE_SIZE}")
    print(f"Low-res images: {len(os.listdir(HR_DIR))}, Size: {LR_IMAGE_SIZE}") 
    print(f"Training on: {DEVICE}")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Log directory: {LOG_DIR}")
    print(f"Checkpoints directory: {CHECKPOINT_DIR}")
    print()

    gen_losses, disc_losses, eval_epochs, psnr_scores, ssim_scores = train_sr_gan(generator, discriminator, generator_loss, discriminator_loss, HR_DIR, VAL_DIR, NUM_EPOCHS)
    writer.close()
    
    import matplotlib
    matplotlib.use("Agg")

    # Create directory for graphs
    os.makedirs("graphs", exist_ok=True)
    
    # Save Generator Loss graph
    plt.figure()
    plt.plot(range(1, NUM_EPOCHS+1), gen_losses, label="Generator Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Generator Loss per Epoch")
    plt.grid(True)
    plt.legend()
    plt.savefig("graphs/generator_loss.png")
    plt.close()
    
    # Save Discriminator Loss graph
    plt.figure()
    plt.plot(range(1, NUM_EPOCHS+1), disc_losses, label="Discriminator Loss", color="orange")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Discriminator Loss per Epoch")
    plt.grid(True)
    plt.legend()
    plt.savefig("graphs/discriminator_loss.png")
    plt.close()
    
    # Save PSNR graph
    plt.figure()
    plt.plot(eval_epochs, psnr_scores, marker="o", label="PSNR", color="green")
    plt.xlabel("Epoch")
    plt.ylabel("PSNR")
    plt.title("PSNR During Evaluation")
    plt.grid(True)
    plt.legend()
    plt.savefig("graphs/psnr.png")
    plt.close()
    
    # Save SSIM graph
    plt.figure()
    plt.plot(eval_epochs, ssim_scores, marker="o", label="SSIM", color="red")
    plt.xlabel("Epoch")
    plt.ylabel("SSIM")
    plt.title("SSIM During Evaluation")
    plt.grid(True)
    plt.legend()
    plt.savefig("graphs/ssim.png")
    plt.close()
    
    print("\nGraphs saved in the 'graphs' folder.\n")