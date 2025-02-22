import torch
from torch.utils.data import DataLoader
import argparse
from gan_models import Generator
from utils.prepare_dataset import PairedDataset
from utils.eval_metrics import calculate_psnr, calculate_ssim
from gan_train import HR_IMAGE_SIZE, LR_IMAGE_SIZE

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DEFAULT_BATCH_SIZE = 4

def load_generator(model_path):
    generator = Generator().to(DEVICE)
    checkpoint = torch.load(model_path, map_location=DEVICE)
    generator.load_state_dict(checkpoint)
    generator.eval()
    return generator

def test_batch(hr_dir, lr_dir, model_path, batch_size):
    dataset = PairedDataset(hr_dir, lr_dir, hr_image_size=HR_IMAGE_SIZE, lr_image_size=LR_IMAGE_SIZE)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    
    generator = load_generator(model_path)
    psnr_values = []
    ssim_values = []
    
    with torch.no_grad():
        for batch in dataloader:
            hr_batch = batch["hr"].to(DEVICE)
            lr_batch = batch["lr"].to(DEVICE)
            fake_hr = generator(lr_batch)
            
            # Evaluate each image in the batch
            for i in range(fake_hr.size(0)):
                # Ensure both tensors have shape (1, C, H, W)
                real = hr_batch[i].unsqueeze(0)
                fake = fake_hr[i].unsqueeze(0)
                psnr = calculate_psnr(real, fake)
                ssim = calculate_ssim(real, fake)
                psnr_values.append(psnr)
                ssim_values.append(ssim)
            # Process only one batch; remove break if you need evaluation on full dataset
            break

    avg_psnr = sum(psnr_values) / len(psnr_values)
    avg_ssim = sum(ssim_values) / len(ssim_values)
    print(f"Avg PSNR: {avg_psnr:.2f} dB")
    print(f"Avg SSIM: {avg_ssim:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test SRGAN on a batch of images")
    parser.add_argument("--hr_dir", type=str, required=True, help="Directory with high-resolution images")
    parser.add_argument("--lr_dir", type=str, required=True, help="Directory with low-resolution images")
    parser.add_argument("--model", type=str, required=True, help="Path to the saved generator weights")
    parser.add_argument("--batch_size", type=int, default=DEFAULT_BATCH_SIZE, help="Batch size for testing")
    args = parser.parse_args()
    
    test_batch(args.hr_dir, args.lr_dir, args.model, args.batch_size)