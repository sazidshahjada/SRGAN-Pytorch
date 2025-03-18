import os
import random
import shutil
from tqdm import tqdm

source_dir = "./data/High_Res_Images"
destination_dir = "./val_images/HR_images"
num_samples = 50

os.makedirs(destination_dir, exist_ok=True)

image_extensions = (".jpg", ".jpeg", ".png")
all_images = [f for f in os.listdir(source_dir) if f.lower().endswith(image_extensions)]

sampled_images = random.sample(all_images, min(num_samples, len(all_images)))

for img in tqdm(sampled_images):
    shutil.copy(os.path.join(source_dir, img), os.path.join(destination_dir, img))

print(f"Copied {len(sampled_images)} images to {destination_dir}")
