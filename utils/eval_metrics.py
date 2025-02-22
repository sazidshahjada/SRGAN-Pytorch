import torch
import torch.nn.functional as F
import numpy as np
import cv2
from skimage.metrics import structural_similarity as ssim

def calculate_psnr(img1, img2):
    mse = F.mse_loss(img1, img2)
    if mse == 0:
        return float('inf')
    return 20 * torch.log10(1.0 / torch.sqrt(mse))

def calculate_ssim(img1, img2):
    img1 = img1.squeeze(0).permute(1, 2, 0).cpu().numpy()
    img2 = img2.squeeze(0).permute(1, 2, 0).cpu().numpy()
    img1 = cv2.cvtColor(img1, cv2.COLOR_RGB2GRAY)
    img2 = cv2.cvtColor(img2, cv2.COLOR_RGB2GRAY)
    return ssim(img1, img2, data_range=img1.max() - img1.min())

