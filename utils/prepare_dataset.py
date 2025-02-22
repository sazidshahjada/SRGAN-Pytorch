import os
from PIL import Image
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader

class PairedDataset(Dataset):
    def __init__(self, hr_dir, hr_image_size=(256, 256), lr_image_size=(64, 64)):
        self.hr_dir = hr_dir
        self.hr_transform = transforms.Compose([
            transforms.Resize(hr_image_size, interpolation=Image.BICUBIC),
            transforms.ToTensor()
        ])
        self.lr_transform = transforms.Compose([
            transforms.Resize(lr_image_size, interpolation=Image.BICUBIC),
            transforms.ToTensor()
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