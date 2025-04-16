import torch
import torch.nn as nn
import torch.nn.functional as F
from gan_models import VGG19FeatureExtractor
from parameters import DEVICE


class GeneratorLoss(nn.Module):
    def __init__(self, alpha=1e-3):
        super(GeneratorLoss, self).__init__()
        self.feature_extractor = VGG19FeatureExtractor().eval().to(DEVICE)
        self.mse_loss = nn.MSELoss()  # Option 1: Use MSE for content (SRGAN paper)
        self.l1_loss = nn.L1Loss()  # Option 2: Keep L1 for content (notebook)
        self.alpha = alpha

    def forward(self, fake_hr, real_hr, fake_preds):
        # Use MSE for content loss (uncomment l1_loss if you prefer the notebook's approach)
        # content_loss = self.mse_loss(self.feature_extractor(fake_hr), self.feature_extractor(real_hr))
        content_loss = self.l1_loss(self.feature_extractor(fake_hr), self.feature_extractor(real_hr))
        adversarial_loss = self.mse_loss(fake_preds, torch.ones_like(fake_preds))
        return content_loss + self.alpha * adversarial_loss


class ImprovedGeneratorLoss(nn.Module):
    def __init__(self, alpha=1e-3, beta=1e-2):
        super(ImprovedGeneratorLoss, self).__init__()
        self.feature_extractor = VGG19FeatureExtractor().eval().to(DEVICE)
        self.mse_loss = nn.MSELoss()
        self.l1_loss = nn.L1Loss()
        self.bce_loss = nn.BCEWithLogitsLoss()  # More stable than BCELoss
        self.alpha = alpha  # Weight for adversarial loss
        self.beta = beta    # Weight for L1 loss

    def forward(self, fake_hr, real_hr, fake_preds):
        content_loss = self.mse_loss(self.feature_extractor(fake_hr), self.feature_extractor(real_hr))
        pixel_loss = self.l1_loss(fake_hr, real_hr)
        adversarial_loss = self.bce_loss(fake_preds, torch.ones_like(fake_preds))
        return content_loss + self.beta * pixel_loss + self.alpha * adversarial_loss


class DiscriminatorLoss(nn.Module):
    def __init__(self):
        super(DiscriminatorLoss, self).__init__()
        self.mse_loss = nn.MSELoss()

    def forward(self, real_preds, fake_preds):
        real_loss = self.mse_loss(real_preds, torch.ones_like(real_preds))
        fake_loss = self.mse_loss(fake_preds, torch.zeros_like(fake_preds))
        return (real_loss + fake_loss) / 2