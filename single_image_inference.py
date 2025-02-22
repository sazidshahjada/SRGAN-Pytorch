import torch
from torchvision import transforms
from PIL import Image
from gan_models import Generator

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LR_IMAGE_SIZE = (64, 64)
# HR_IMAGE_SIZE is defined by your generator output (e.g. 256 x 256)

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
        transforms.ToTensor()
    ])
    lr_tensor = transform_lr(lr_image).unsqueeze(0).to(DEVICE)
    
    # Generate HR image
    with torch.no_grad():
        fake_hr_tensor = generator(lr_tensor)
    
    # Post process: remove batch dim, clamp and convert to PIL Image
    fake_hr_tensor = fake_hr_tensor.squeeze(0).cpu().clamp(0, 1)
    to_pil = transforms.ToPILImage()
    fake_hr_image = to_pil(fake_hr_tensor)
    fake_hr_image.save(output_path)
    print(f"Saved super-resolved image to {output_path}")

if __name__ == "__main__":
    import os

    checkpoint_dir = "/home/iot/SRGAN_implementation/checkpoints"
    model_file = "srgan_epoch_9.pth"
    lr_image_dir = "/home/iot/SRGAN_implementation/test_images"
    lr_image_file = "monalisa.jpeg"
    lr_image = os.path.join(lr_image_dir, lr_image_file)
    
    model_path = os.path.join(checkpoint_dir, model_file)
    output = f"/home/iot/SRGAN_implementation/outputs/{os.path.splitext(lr_image_file)[0]}-{os.path.splitext(model_file)[0]}.jpg"
    print("Loading SRGAN generator...")
    generator = load_generator(model_path)
    print("Model loaded successfully!")
    print("Generating super-resolved image...")
    super_resolve(generator, lr_image, output)
    print()