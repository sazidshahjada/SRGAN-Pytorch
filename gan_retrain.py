import os
import shutil
import torch
import torch.optim as optim
import torch.nn.functional as F
from torchvision import transforms
import torchvision.utils as vutils
from torch.nn.utils import clip_grad_norm_
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt

from tqdm import tqdm
from parameters import HR_DIR, VAL_DIR, HR_IMAGE_SIZE, LR_IMAGE_SIZE, DEVICE, BATCH_SIZE, STEP_SIZE, LEARNING_RATE, BETA1, BETA2, ALPHA, BETA
from utils.prepare_dataset import PairedDataset, denormalize, denormalize_gen, MEAN, STD
from utils.eval_metrics import calculate_psnr, calculate_ssim
from gan_models import Generator, Discriminator
from losses import GeneratorLoss, DiscriminatorLoss

# Override parameters if needed
NUM_EPOCHS = 100  # Additional epochs to train
LEARNING_RATE = 0.0000008


# Setup directories
CHECKPOINT_DIR = "./checkpoints"
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

LOG_DIR = "./runs/srgan_experiment"
if os.path.exists(LOG_DIR):
    shutil.rmtree(LOG_DIR)  # Clean previous logs to avoid overlap
os.makedirs(LOG_DIR, exist_ok=True)

RESULTS_DIR = "./results"
os.makedirs(RESULTS_DIR, exist_ok=True)

GRAPHS_DIR = "./graphs"
os.makedirs(GRAPHS_DIR, exist_ok=True)

writer = SummaryWriter(log_dir=LOG_DIR)

# Initialize dataloaders
train_dataset = PairedDataset(hr_dir=HR_DIR, hr_image_size=HR_IMAGE_SIZE, lr_image_size=LR_IMAGE_SIZE)
train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_dataset = PairedDataset(hr_dir=VAL_DIR, hr_image_size=HR_IMAGE_SIZE, lr_image_size=LR_IMAGE_SIZE)
val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

# Initialize models, losses, optimizers, schedulers
generator = Generator().to(DEVICE)
discriminator = Discriminator(HR_IMAGE_SIZE).to(DEVICE)
generator_loss = GeneratorLoss(alpha=ALPHA, beta=BETA).to(DEVICE)  # Updated with beta for pixel-wise loss
discriminator_loss = DiscriminatorLoss().to(DEVICE)

generator_optimizer = optim.Adam(generator.parameters(), lr=LEARNING_RATE, betas=(BETA1, BETA2))
discriminator_optimizer = optim.Adam(discriminator.parameters(), lr=LEARNING_RATE, betas=(BETA1, BETA2))

generator_scheduler = torch.optim.lr_scheduler.StepLR(generator_optimizer, step_size=STEP_SIZE, gamma=0.1)
discriminator_scheduler = torch.optim.lr_scheduler.StepLR(discriminator_optimizer, step_size=STEP_SIZE, gamma=0.1)

# Optionally resume training from a checkpoint
model_name = "srgan_epoch_100.pth"  # Change the path if needed
resume_checkpoint = os.path.join(CHECKPOINT_DIR, model_name)
init_epoch = 0
if os.path.exists(resume_checkpoint):
    checkpoint = torch.load(resume_checkpoint, map_location=DEVICE)
    generator.load_state_dict(checkpoint["generator_state_dict"])
    discriminator.load_state_dict(checkpoint["discriminator_state_dict"])
    generator_optimizer.load_state_dict(checkpoint["generator_optimizer_state_dict"])
    discriminator_optimizer.load_state_dict(checkpoint["discriminator_optimizer_state_dict"])
    init_epoch = checkpoint["epoch"] + 1
    print(f"Resumed from checkpoint: {resume_checkpoint} at epoch {init_epoch}")
else:
    print("No checkpoint found, starting from scratch.")

generator_scheduler = lr_scheduler.StepLR(generator_optimizer, step_size=30, gamma=0.1)
discriminator_scheduler = lr_scheduler.StepLR(discriminator_optimizer, step_size=30, gamma=0.1)

# Updated train_sr_gan function with gradient clipping and proper epoch handling
def train_sr_gan(generator, discriminator, generator_loss, discriminator_loss, hr_dir, val_dir, start_epoch=0, num_epochs=NUM_EPOCHS):
    global_step = 0
    avg_gen_losses = []
    avg_disc_losses = []
    eval_epochs = []
    eval_psnr_list = []
    eval_ssim_list = []
    
    for epoch in range(start_epoch, num_epochs):
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
            clip_grad_norm_(discriminator.parameters(), max_norm=1.0)  # Add gradient clipping
            discriminator_optimizer.step()
            
            # Train generator
            generator_optimizer.zero_grad()
            fake_hr = generator(lr_batch)
            fake_preds = discriminator(fake_hr)
            g_loss = generator_loss(fake_hr, hr_batch, fake_preds)
            g_loss.backward()
            clip_grad_norm_(generator.parameters(), max_norm=1.0)  # Add gradient clipping
            generator_optimizer.step()
            
            # Log losses and accumulate numbers
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

        print(f" - Generator Loss: {avg_gen_loss:.8f}, Discriminator Loss: {avg_disc_loss:.8f}")
        
        # Step the schedulers
        generator_scheduler.step()
        discriminator_scheduler.step()
        
        # Save checkpoint
        checkpoint_path = os.path.join(CHECKPOINT_DIR, f"srgan_epoch_{epoch+1}.pth")
        torch.save({
            "epoch": epoch,
            "generator_state_dict": generator.state_dict(),
            "discriminator_state_dict": discriminator.state_dict(),
            "generator_optimizer_state_dict": generator_optimizer.state_dict(),
            "discriminator_optimizer_state_dict": discriminator_optimizer.state_dict(),
            "generator_scheduler_state_dict": generator_scheduler.state_dict(),
            "discriminator_scheduler_state_dict": discriminator_scheduler.state_dict(),
        }, checkpoint_path)
        
        # Evaluate every 5 epochs
        if (epoch + 1) % 5 == 0:
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
                        psnr_values.append(calculate_psnr(denormalize(hr_batch[i], MEAN, STD), denormalize_gen(fake_hr[i], MEAN, STD)))
                        ssim_values.append(calculate_ssim(denormalize(hr_batch[i], MEAN, STD), denormalize_gen(fake_hr[i], MEAN, STD)))
                        
                        # Resize LR image to match HR dimensions before concatenation
                        lr_resized = F.interpolate(lr_batch[i].unsqueeze(0),
                                                   size=hr_batch[i].shape[-2:],
                                                   mode='bicubic',
                                                   align_corners=False)
                        # Concatenate denormalized images along width (dim=3)
                        result_image = torch.cat((denormalize(lr_resized, MEAN, STD),
                                                  denormalize_gen(fake_hr[i].unsqueeze(0), MEAN, STD),
                                                  denormalize(hr_batch[i].unsqueeze(0), MEAN, STD)),
                                                  dim=3)
                        image_path = os.path.join(epoch_results_dir, f"image_{batch_idx * val_dataloader.batch_size + i + 1}.png")
                        vutils.save_image(result_image, image_path)
                
                avg_psnr = torch.tensor(psnr_values).mean().item()
                avg_ssim = torch.tensor(ssim_values).mean().item()
                
                writer.add_scalar("Eval/PSNR", avg_psnr, epoch)
                writer.add_scalar("Eval/SSIM", avg_ssim, epoch)
                print(f"\nEvaluation - Avg. PSNR: {avg_psnr:.2f}, Avg. SSIM: {avg_ssim:.4f}")
                print(f"Results saved in: {epoch_results_dir}\n")
                
                eval_epochs.append(epoch+1)
                eval_psnr_list.append(avg_psnr)
                eval_ssim_list.append(avg_ssim)

        # Save checkpoint every 10 epochs
        if (epoch + 1) % 10 == 0:
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
    
    # Adjust the range of epochs for plotting to match the actual epochs trained
    trained_epochs = list(range(init_epoch + 1, total_epochs + 1))
    
    # Save Generator Loss graph
    plt.figure()
    plt.plot(trained_epochs, gen_losses, label="Generator Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Generator Loss per Epoch")
    plt.grid(True)
    plt.legend()
    plt.savefig(os.path.join(GRAPHS_DIR, f"generator_loss_{total_epochs}.png"))
    plt.close()
    
    # Save Discriminator Loss graph
    plt.figure()
    plt.plot(trained_epochs, disc_losses, label="Discriminator Loss", color="orange")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Discriminator Loss per Epoch")
    plt.grid(True)
    plt.legend()
    plt.savefig(os.path.join(GRAPHS_DIR, f"discriminator_loss_{total_epochs}.png"))
    plt.close()
    
    # Save PSNR graph
    plt.figure()
    plt.plot(eval_epochs, psnr_scores, marker="o", label="PSNR", color="green")
    plt.xlabel("Epoch")
    plt.ylabel("PSNR")
    plt.title("PSNR During Evaluation")
    plt.grid(True)
    plt.legend()
    plt.savefig(os.path.join(GRAPHS_DIR, f"psnr_{total_epochs}.png"))
    plt.close()
    
    # Save SSIM graph
    plt.figure()
    plt.plot(eval_epochs, ssim_scores, marker="o", label="SSIM", color="red")
    plt.xlabel("Epoch")
    plt.ylabel("SSIM")
    plt.title("SSIM During Evaluation")
    plt.grid(True)
    plt.legend()
    plt.savefig(os.path.join(GRAPHS_DIR, f"ssim_{total_epochs}.png"))
    plt.close()
    
    print("\nGraphs saved in the 'graphs' folder.\n")
    torch.cuda.empty_cache()