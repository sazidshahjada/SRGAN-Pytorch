import os
import torch
import torch.nn.functional as F
import torchvision.utils as vutils
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np

from parameters import VAL_DIR, HR_IMAGE_SIZE, LR_IMAGE_SIZE, DEVICE, BATCH_SIZE
from utils.prepare_dataset import PairedDataset, denormalize, denormalize_gen, MEAN, STD
from utils.eval_metrics import calculate_psnr, calculate_ssim
from gan_models import Generator

# Setup directories
EVAL_RESULTS_DIR = "./eval_results"
os.makedirs(EVAL_RESULTS_DIR, exist_ok=True)

# Initialize the generator and load the trained weights
def load_generator(checkpoint_path):
    generator = Generator().to(DEVICE)
    checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
    generator.load_state_dict(checkpoint["generator_state_dict"])
    generator.eval()
    return generator

def evaluate_generator(generator, val_dir, checkpoint_path, output_dir=EVAL_RESULTS_DIR):
    # Initialize validation dataset and dataloader
    val_dataset = PairedDataset(hr_dir=val_dir, hr_image_size=HR_IMAGE_SIZE, lr_image_size=LR_IMAGE_SIZE)
    val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # Lists to store metrics
    psnr_values = []
    ssim_values = []

    # Open a file to log metrics
    metrics_file = os.path.join(output_dir, "evaluation_metrics.txt")
    with open(metrics_file, "w") as f:
        f.write(f"Evaluation Results for Generator: {checkpoint_path}\n")
        f.write(f"Validation Dataset: {val_dir}\n")
        f.write(f"Total Images: {len(val_dataset)}\n\n")

    print(f"Evaluating generator on {len(val_dataset)} validation images...")
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(val_dataloader, desc="Evaluating")):
            hr_batch = batch["hr"].to(DEVICE)
            lr_batch = batch["lr"].to(DEVICE)
            
            # Generate super-resolved images
            fake_hr = generator(lr_batch)
            
            # Compute metrics for each image in the batch
            for i in range(hr_batch.size(0)):
                psnr = calculate_psnr(hr_batch[i], fake_hr[i])
                ssim = calculate_ssim(hr_batch[i], fake_hr[i])
                psnr_values.append(psnr.item())
                ssim_values.append(ssim)
                
                # Save side-by-side comparison images
                # Resize LR image to match HR dimensions for visualization
                lr_resized = F.interpolate(lr_batch[i].unsqueeze(0),
                                         size=HR_IMAGE_SIZE,
                                         mode='bicubic',
                                         align_corners=False)
                
                # Denormalize images for visualization
                lr_img = denormalize(lr_resized, MEAN, STD)
                fake_img = denormalize_gen(fake_hr[i].unsqueeze(0), MEAN, STD)
                hr_img = denormalize(hr_batch[i].unsqueeze(0), MEAN, STD)
                
                # Concatenate images side by side (LR, Generated, HR)
                result_image = torch.cat((lr_img, fake_img, hr_img), dim=3)
                image_path = os.path.join(output_dir, f"image_{batch_idx * BATCH_SIZE + i + 1}.png")
                vutils.save_image(result_image, image_path)
                
                # Log metrics for this image
                with open(metrics_file, "a") as f:
                    f.write(f"Image {batch_idx * BATCH_SIZE + i + 1}: PSNR = {psnr.item():.2f}, SSIM = {ssim:.4f}\n")
    
    # Compute average metrics
    avg_psnr = np.mean(psnr_values)
    avg_ssim = np.mean(ssim_values)
    
    # Log average metrics
    with open(metrics_file, "a") as f:
        f.write(f"\nAverage PSNR: {avg_psnr:.2f}\n")
        f.write(f"Average SSIM: {avg_ssim:.4f}\n")
    
    print(f"\nEvaluation Complete!")
    print(f"Average PSNR: {avg_psnr:.2f}")
    print(f"Average SSIM: {avg_ssim:.4f}")
    print(f"Results saved in: {output_dir}")
    print(f"Metrics logged in: {metrics_file}")

    # Plot PSNR and SSIM distributions
    plt.figure(figsize=(10, 5))
    
    plt.subplot(1, 2, 1)
    plt.hist(psnr_values, bins=20, color='blue', alpha=0.7)
    plt.title("PSNR Distribution")
    plt.xlabel("PSNR")
    plt.ylabel("Frequency")
    
    plt.subplot(1, 2, 2)
    plt.hist(ssim_values, bins=20, color='green', alpha=0.7)
    plt.title("SSIM Distribution")
    plt.xlabel("SSIM")
    plt.ylabel("Frequency")
    
    plt.tight_layout()
    plt.savefig("graphs/eval_metrics_distribution.png")
    plt.close()

if __name__ == "__main__":
    # Path to the trained generator checkpoint
    CHECKPOINT_PATH = "./checkpoints/srgan_epoch_100.pth"  # Adjust to your checkpoint path
    
    # Load the generator
    generator = load_generator(CHECKPOINT_PATH)
    
    # Evaluate the generator
    evaluate_generator(generator, VAL_DIR, CHECKPOINT_PATH)