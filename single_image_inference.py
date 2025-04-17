import torch
from torchvision import transforms
from PIL import Image
from gan_models import Generator
from parameters import DEVICE, LR_IMAGE_SIZE
from utils.prepare_dataset import denormalize, MEAN, STD


def load_generator(model_path):
    generator = Generator().to(DEVICE)
    checkpoint = torch.load(model_path, map_location=DEVICE)
    # Extract only the generator's state dictionary
    generator.load_state_dict(checkpoint["generator_state_dict"])
    generator.eval()
    return generator

def super_resolve(generator, lr_image_path, output_path):
    # Load and preprocess LR image
    lr_image = Image.open(lr_image_path).convert("RGB")
    transform_lr = transforms.Compose([
        transforms.Resize(LR_IMAGE_SIZE, interpolation=Image.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize(mean=MEAN, std=STD)
    ])
    lr_tensor = transform_lr(lr_image).unsqueeze(0).to(DEVICE)
    
    # Generate HR image
    with torch.no_grad():
        fake_hr_tensor = denormalize(generator(lr_tensor))
    
    # Post process: remove batch dim, clamp and convert to PIL Image
    fake_hr_tensor = fake_hr_tensor.squeeze(0).cpu().clamp(0, 1)
    to_pil = transforms.ToPILImage()
    fake_hr_image = to_pil(fake_hr_tensor)
    fake_hr_image.save(output_path)
    print(f"Saved super-resolved image to {output_path}")

if __name__ == "__main__":
    import os

    checkpoint_dir = "./checkpoints"
    model_file = "srgan_epoch_100.pth"
    lr_image_dir = "./test_images"
    lr_image_file = "white_64x64.jpg"
    lr_image = os.path.join(lr_image_dir, lr_image_file)
    
    model_path = os.path.join(checkpoint_dir, model_file)
    output = f"./outputs/{os.path.splitext(lr_image_file)[0]}-{os.path.splitext(model_file)[0]}.jpg"
    print("Loading SRGAN generator...")
    generator = load_generator(model_path)
    print("Model loaded successfully!")
    print("Generating super-resolved image...")
    super_resolve(generator, lr_image, output)
    print()