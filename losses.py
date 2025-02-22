import torch
import torch.nn as nn
import torch.nn.functional as F
from gan_models import VGG19FeatureExtractor

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class GeneratorLoss(nn.Module):
    def __init__(self, alpha=0.0006):
        super(GeneratorLoss, self).__init__()
        self.feature_extractor = VGG19FeatureExtractor().eval().to(DEVICE)
        self.mse_loss = nn.MSELoss()
        self.bce_loss = nn.BCEWithLogitsLoss()
        self.alpha = alpha

    def forward(self, fake_hr, real_hr, fake_preds):
        perceptual_loss = self.mse_loss(self.feature_extractor(fake_hr), self.feature_extractor(real_hr))
        adversarial_loss = self.bce_loss(fake_preds, torch.ones_like(fake_preds))
        return perceptual_loss + self.alpha * adversarial_loss

class DiscriminatorLoss(nn.Module):
    def __init__(self):
        super(DiscriminatorLoss, self).__init__()
        self.bce_loss = nn.BCELoss()

    def forward(self, real_preds, fake_preds):
        real_loss = self.bce_loss(real_preds, torch.ones_like(real_preds))
        fake_loss = self.bce_loss(fake_preds, torch.zeros_like(fake_preds))
        return real_loss + fake_loss
