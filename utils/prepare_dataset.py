import os
import torch
from PIL import Image
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from parameters import HR_IMAGE_SIZE, LR_IMAGE_SIZE, DEVICE


MEAN = [0.41766175627708435, 0.4126753509044647, 0.39600545167922974]
STD =  [0.2138405591249466, 0.1984556019306183, 0.2018120288848877]

class PairedDataset(Dataset):
    def __init__(self, hr_dir, hr_image_size=HR_IMAGE_SIZE, lr_image_size=LR_IMAGE_SIZE):
        self.hr_dir = hr_dir
        self.hr_transform = transforms.Compose([
            transforms.Resize(hr_image_size, interpolation=Image.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize(mean=MEAN, std=STD)
        ])
        self.lr_transform = transforms.Compose([
            transforms.Resize(lr_image_size, interpolation=Image.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize(mean=MEAN, std=STD)
        ])
        # List of high-resolution image filenames
        self.image_names = os.listdir(hr_dir)
    
    def __len__(self):
        return len(self.image_names)
    
    def __getitem__(self, idx):
        img_name = self.image_names[idx]
        hr_path = os.path.join(self.hr_dir, img_name)
        
        # Open the image once and generate both HR and LR versions
        pil_img = Image.open(hr_path).convert("RGB")
        hr_img = self.hr_transform(pil_img)
        lr_img = self.lr_transform(pil_img)
        
        return {'hr': hr_img, 'lr': lr_img}

def denormalize(tensor, mean=MEAN, std=STD):
    # Convert MEAN and STD to torch tensors and reshape for broadcasting
    mean = torch.tensor(mean).view(-1, 1, 1).to(tensor.device)
    std = torch.tensor(std).view(-1, 1, 1).to(tensor.device)
    # Revert normalization
    tensor_denorm = tensor * std + mean
    return tensor_denorm.clamp(0, 1)

def denormalize_gen(img, mean=MEAN, std=STD):
    device = DEVICE
    mean = torch.tensor(mean, device=device).view(-1, 1, 1)
    std = torch.tensor(std, device=device).view(-1, 1, 1)
    img = img * std + mean  # Denormalize to [mean - std, mean + std]
    # Scale to [0, 1] by normalizing based on the min and max of the denormalized range
    img_min = torch.tensor([m - s for m, s in zip(mean.flatten(), std.flatten())], device=device).view(-1, 1, 1)
    img_max = torch.tensor([m + s for m, s in zip(mean.flatten(), std.flatten())], device=device).view(-1, 1, 1)
    img = (img - img_min) / (img_max - img_min)  # Scale to [0, 1]
    return img.clamp(0, 1)


if __name__ == "__main__":
    hr_dir = "/home/iot/SRGAN_implementation/data/High_Res_Images"  # Adjust to your HR images folder
    batch_size = 16
    num_workers = 4

    dataset = PairedDataset(hr_dir)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    
    for batch in dataloader:
        hr_batch = batch["hr"]
        lr_batch = batch["lr"]
        print("HR batch shape:", hr_batch.shape)
        print("LR batch shape:", lr_batch.shape)
        break