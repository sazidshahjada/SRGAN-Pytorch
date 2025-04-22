import os
import torch
from PIL import Image
from tqdm import tqdm
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset

class UnsplashDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.image_names = os.listdir(root_dir)
        self.transform = transform

    def __len__(self):
        return len(self.image_names)
    
    def __getitem__(self, idx):
        path = os.path.join(self.root_dir, self.image_names[idx])
        image = Image.open(path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image

transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
])


if __name__ == "__main__":
    data_path = "./data/HR_images"
    dataset = UnsplashDataset(data_path, transform=transform)
    loader = DataLoader(dataset, batch_size=16, shuffle=False)

    mean = torch.zeros(3)
    std = torch.zeros(3)
    nb_samples = 0.
    for data in tqdm(loader, desc="Calculating statistics", unit="batch"):
        batch_samples = data.size(0)
        data = data.view(batch_samples, data.size(1), -1)
        mean += data.mean(2).sum(0)
        std += data.std(2).sum(0)
        nb_samples += batch_samples

    mean /= nb_samples
    std /= nb_samples

    print("Mean:", mean.tolist())
    print("Std:", std.tolist())
