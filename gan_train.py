import os
import shutil
import torch
import torch.optim as optim
import torch.nn.functional as F
import torchvision.utils as vutils
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
from tqdm import tqdm

from parameters import HR_DIR, VAL_DIR, HR_IMAGE_SIZE, LR_IMAGE_SIZE, DEVICE, NUM_EPOCHS, BATCH_SIZE, LEARNING_RATE, BETA1, BETA2, ALPHA, BETA
from utils.prepare_dataset import MEAN, STD, PairedDataset, denormalize, denormalize_gen
from utils.eval_metrics import calculate_psnr, calculate_ssim, calculate_lpips
from gan_models import Generator, Discriminator
from losses import GeneratorLoss, DiscriminatorLoss

# Clear GPU memory
torch.cuda.empty_cache()

# Setup directories
CHECKPOINT_DIR = "./checkpoints"
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

LOG_DIR = "./runs/srgan_experiment"
if os.path.exists(LOG_DIR):
    try:
        shutil.rmtree(LOG_DIR)
    except Exception as e:
        print(f"Error clearing {LOG_DIR}: {e}")
os.makedirs(LOG_DIR, exist_ok=True)

RESULTS_DIR = "./results"
if os.path.exists(RESULTS_DIR):
    try:
        shutil.rmtree(RESULTS_DIR)
    except Exception as e:
        print(f"Error clearing {RESULTS_DIR}: {e}")
os.makedirs(RESULTS_DIR, exist_ok=True)

GRAPHS_DIR = "./graphs"
os.makedirs(GRAPHS_DIR, exist_ok=True)

writer = SummaryWriter(log_dir=LOG_DIR)

# Initialize models, losses, optimizers, and schedulers
generator = Generator().to(DEVICE)
discriminator = Discriminator(HR_IMAGE_SIZE).to(DEVICE)
generator_loss = GeneratorLoss(alpha=ALPHA, beta=BETA).to(DEVICE)
discriminator_loss = DiscriminatorLoss().to(DEVICE)

# TTUR: Different learning rates
generator_optimizer = optim.Adam(generator.parameters(), lr=LEARNING_RATE, betas=(BETA1, BETA2))
discriminator_optimizer = optim.Adam(discriminator.parameters(), lr=LEARNING_RATE/10, betas=(BETA1, BETA2))

generator_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    generator_optimizer, mode='max', factor=0.5, patience=10, verbose=True, min_lr=1e-6
)
discriminator_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    discriminator_optimizer, mode='max', factor=0.5, patience=10, verbose=True, min_lr=1e-6
)

# Initialize dataloaders
train_dataset = PairedDataset(hr_dir=HR_DIR, hr_image_size=HR_IMAGE_SIZE, lr_image_size=LR_IMAGE_SIZE)
train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_dataset = PairedDataset(hr_dir=VAL_DIR, hr_image_size=HR_IMAGE_SIZE, lr_image_size=LR_IMAGE_SIZE)
val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

# Optionally resume training from a checkpoint
model_name = "srgan_epoch_50.pth"
resume_checkpoint = os.path.join(CHECKPOINT_DIR, model_name)
start_epoch = 0
if os.path.exists(resume_checkpoint):
    checkpoint = torch.load(resume_checkpoint, map_location=DEVICE)
    generator.load_state_dict(checkpoint["generator_state_dict"])
    discriminator.load_state_dict(checkpoint["discriminator_state_dict"])
    generator_optimizer.load_state_dict(checkpoint["generator_optimizer_state_dict"])
    discriminator_optimizer.load_state_dict(checkpoint["discriminator_optimizer_state_dict"])
    generator_scheduler.load_state_dict(checkpoint["generator_scheduler_state_dict"])
    discriminator_scheduler.load_state_dict(checkpoint["discriminator_scheduler_state_dict"])
    start_epoch = checkpoint["epoch"] + 1
    print(f"Resumed from checkpoint: {resume_checkpoint} at epoch {start_epoch}")
else:
    print("No checkpoint found, starting from scratch.")

def train_sr_gan(generator, discriminator, generator_loss, discriminator_loss, hr_dir, val_dir, start_epoch=0, num_epochs=NUM_EPOCHS):
    global_step = 0
    avg_gen_losses = []
    avg_disc_losses = []
    eval_epochs = []
    eval_psnr_list = []
    eval_ssim_list = []
    eval_lpips_list = []
    
    for epoch in range(start_epoch, num_epochs):
        generator.train()
        discriminator.train()
        epoch_g_loss_sum = 0.0
        epoch_d_loss_sum = 0.0
        num_batches = 0
        
        for batch in tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            hr_batch = batch["hr"].to(DEVICE)
            lr_batch = batch["lr"].to(DEVICE)
            
            # Train generator every iteration
            generator_optimizer.zero_grad()
            fake_hr = generator(lr_batch)
            fake_preds = discriminator(fake_hr)
            g_loss = generator_loss(fake_hr, hr_batch, fake_preds)
            g_loss.backward()
            torch.nn.utils.clip_grad_norm_(generator.parameters(), max_norm=0.1)
            generator_optimizer.step()
            
            # Train discriminator every 2 steps
            if global_step % 2 == 0:
                discriminator_optimizer.zero_grad()
                real_preds = discriminator(hr_batch)
                fake_preds = discriminator(fake_hr.detach())
                d_loss = discriminator_loss(real_preds, fake_preds)
                d_loss.backward()
                torch.nn.utils.clip_grad_norm_(discriminator.parameters(), max_norm=0.1)
                discriminator_optimizer.step()
            else:
                d_loss = torch.tensor(0.0)
            
            # Log losses and accumulate numbers
            writer.add_scalar("Loss/Discriminator", d_loss.item(), global_step)
            writer.add_scalar("Loss/Generator", g_loss.item(), global_step)
            epoch_g_loss_sum += g_loss.item()
            epoch_d_loss_sum += d_loss.item()
            num_batches += 1
            global_step += 1
        
        # Average losses for the epoch
        avg_gen_loss = epoch_g_loss_sum / num_batches
        avg_disc_loss = epoch_d_loss_sum / num_batches
        avg_gen_losses.append(avg_gen_loss)
        avg_disc_losses.append(avg_disc_loss)
        
        print(f" - Generator Loss: {avg_gen_loss:.8f}, Discriminator Loss: {avg_disc_loss:.8f}")
        
        # Evaluation every 5 epochs
        if (epoch + 1) % 5 == 0:
            generator.eval()
            epoch_results_dir = os.path.join(RESULTS_DIR, f"epoch_{epoch+1}")
            os.makedirs(epoch_results_dir, exist_ok=True)
            
            with torch.no_grad():
                psnr_values = []
                ssim_values = []
                lpips_values = []
                for batch_idx, batch in enumerate(val_dataloader):
                    hr_batch = batch["hr"].to(DEVICE)
                    lr_batch = batch["lr"].to(DEVICE)
                    fake_hr = generator(lr_batch)
                    for i in range(hr_batch.size(0)):
                        hr_denorm = denormalize(hr_batch[i], MEAN, STD)
                        fake_denorm = denormalize_gen(fake_hr[i], MEAN, STD)
                        psnr_values.append(calculate_psnr(hr_denorm, fake_denorm))
                        ssim_values.append(calculate_ssim(hr_denorm, fake_denorm))
                        lpips_score = calculate_lpips(hr_denorm, fake_denorm, device=DEVICE)
                        lpips_values.append(lpips_score)
                        
                        lr_resized = F.interpolate(lr_batch[i].unsqueeze(0),
                                                   size=hr_batch[i].shape[-2:],
                                                   mode='bicubic',
                                                   align_corners=False)
                        result_image = torch.cat((denormalize(lr_resized, MEAN, STD),
                                                  denormalize_gen(fake_hr[i].unsqueeze(0), MEAN, STD),
                                                  denormalize(hr_batch[i].unsqueeze(0), MEAN, STD)),
                                                  dim=3)
                        image_path = os.path.join(epoch_results_dir, f"image_{batch_idx * val_dataloader.batch_size + i + 1}.png")
                        vutils.save_image(result_image, image_path)
                
                avg_psnr = torch.tensor(psnr_values).mean().item()
                avg_ssim = torch.tensor(ssim_values).mean().item()
                avg_lpips = torch.tensor(lpips_values).mean().item()
                
                writer.add_scalar("Eval/PSNR", avg_psnr, epoch)
                writer.add_scalar("Eval/SSIM", avg_ssim, epoch)
                writer.add_scalar("Eval/LPIPS", avg_lpips, epoch)
                print(f"\nEvaluation - Avg. PSNR: {avg_psnr:.2f}, Avg. SSIM: {avg_ssim:.4f}, Avg. LPIPS: {avg_lpips:.4f}")
                print(f"Results saved in: {epoch_results_dir}\n")
                
                eval_epochs.append(epoch+1)
                eval_psnr_list.append(avg_psnr)
                eval_ssim_list.append(avg_ssim)
                eval_lpips_list.append(avg_lpips)
                
                generator_scheduler.step(avg_psnr)
                discriminator_scheduler.step(avg_psnr)
        
        # Save checkpoint every 20 epochs
        if (epoch + 1) % 20 == 0:
            checkpoint_path = os.path.join(CHECKPOINT_DIR, f"srgan_epoch_{epoch+1}.pth")
            torch.save({
                "generator_state_dict": generator.state_dict(),
                "discriminator_state_dict": discriminator.state_dict(),
                "generator_optimizer_state_dict": generator_optimizer.state_dict(),
                "discriminator_optimizer_state_dict": discriminator_optimizer.state_dict(),
                "generator_scheduler_state_dict": generator_scheduler.state_dict(),
                "discriminator_scheduler_state_dict": discriminator_scheduler.state_dict(),
                "epoch": epoch,
            }, checkpoint_path)
            print(f"Checkpoint saved: {checkpoint_path}")
    
    return avg_gen_losses, avg_disc_losses, eval_epochs, eval_psnr_list, eval_ssim_list, eval_lpips_list

if __name__ == "__main__":
    import warnings
    warnings.filterwarnings("ignore")
    
    torch.cuda.empty_cache()
    
    print("Starting SRGAN training...")
    total_epochs = start_epoch + NUM_EPOCHS
    print(f"Training for {total_epochs} epochs (starting at epoch {start_epoch + 1})")
    print(f"High-res images: {len(os.listdir(HR_DIR))}, Size: {HR_IMAGE_SIZE}")
    print(f"Low-res images: {len(os.listdir(HR_DIR))}, Size: {LR_IMAGE_SIZE}") 
    print(f"Training on: {DEVICE}")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Log directory: {LOG_DIR}")
    print(f"Checkpoints directory: {CHECKPOINT_DIR}")
    print()
    
    gen_losses, disc_losses, eval_epochs, psnr_scores, ssim_scores, lpips_scores = train_sr_gan(
        generator, discriminator, generator_loss, discriminator_loss, HR_DIR, VAL_DIR, start_epoch, NUM_EPOCHS
    )
    writer.close()
    
    import matplotlib
    matplotlib.use("Agg")
    
    trained_epochs = list(range(start_epoch + 1, total_epochs + 1))
    
    plt.figure()
    plt.plot(trained_epochs, gen_losses, label="Generator Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Generator Loss per Epoch")
    plt.grid(True)
    plt.legend()
    plt.savefig(os.path.join(GRAPHS_DIR, f"generator_loss_epoch_{total_epochs}.png"))
    plt.close()
    
    plt.figure()
    plt.plot(trained_epochs, disc_losses, label="Discriminator Loss", color="orange")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Discriminator Loss per Epoch")
    plt.grid(True)
    plt.legend()
    plt.savefig(os.path.join(GRAPHS_DIR, f"discriminator_loss_epoch_{total_epochs}.png"))
    plt.close()
    
    plt.figure()
    plt.plot(eval_epochs, psnr_scores, marker="o", label="PSNR", color="green")
    plt.xlabel("Epoch")
    plt.ylabel("PSNR")
    plt.title("PSNR During Evaluation")
    plt.grid(True)
    plt.legend()
    plt.savefig(os.path.join(GRAPHS_DIR, f"psnr_epoch_{total_epochs}.png"))
    plt.close()
    
    plt.figure()
    plt.plot(eval_epochs, ssim_scores, marker="o", label="SSIM", color="red")
    plt.xlabel("Epoch")
    plt.ylabel("SSIM")
    plt.title("SSIM During Evaluation")
    plt.grid(True)
    plt.legend()
    plt.savefig(os.path.join(GRAPHS_DIR, f"ssim_epoch_{total_epochs}.png"))
    plt.close()
    
    plt.figure()
    plt.plot(eval_epochs, lpips_scores, marker="o", label="LPIPS", color="purple")
    plt.xlabel("Epoch")
    plt.ylabel("LPIPS")
    plt.title("LPIPS During Evaluation")
    plt.grid(True)
    plt.legend()
    plt.savefig(os.path.join(GRAPHS_DIR, f"lpips_epoch_{total_epochs}.png"))
    plt.close()
    
    print("\nGraphs saved in the 'graphs' folder.\n")
    torch.cuda.empty_cache()