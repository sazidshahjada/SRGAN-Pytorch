import torch
import torch.nn as nn
from torchvision import models

import warnings
warnings.filterwarnings("ignore")



class _ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding=0):
        super(_ConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.bn = nn.BatchNorm2d(out_channels)
        self.leaky_relu = nn.LeakyReLU(0.2)

    def forward(self, x):
        return self.leaky_relu(self.bn(self.conv(x)))
    


class _ResidualBlock(nn.Module):
    def __init__(self, channels):
        super(_ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, 1, 1)
        self.bn1 = nn.BatchNorm2d(channels, 0.8)
        self.conv2 = nn.Conv2d(channels, channels, 3, 1, 1)
        self.bn2 = nn.BatchNorm2d(channels, 0.8)
        self.prelu = nn.PReLU()

    def forward(self, x):
        residual = self.prelu(self.bn1(self.conv1(x)))
        residual = self.bn2(self.conv2(residual))
        return x + residual



class _UpsampleBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding=0):
        super(_UpsampleBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.pixel_shuffle = nn.PixelShuffle(upscale_factor=2)
        self.bn = nn.BatchNorm2d(out_channels)
        self.prelu = nn.PReLU()

    def forward(self, x):
        return self.prelu(self.pixel_shuffle(self.bn(self.conv(x))))
    


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, 9, 1, 4)
        self.prelu = nn.PReLU()
        self.residuals = nn.Sequential(*[_ResidualBlock(64) for _ in range(16)])
        self.conv2 = nn.Conv2d(64, 64, 3, 1, 1)
        self.bn = nn.BatchNorm2d(64)
        self.upsample = nn.Sequential(
            _UpsampleBlock(64, 256, 3, 1, 1),
            _UpsampleBlock(64, 256, 3, 1, 1),
        )
        self.conv3 = nn.Conv2d(64, 3, 9, 1, 4)

    def forward(self, x):
        x = self.prelu(self.conv1(x))
        residual = x
        x = self.residuals(x)
        x = self.bn(self.conv2(x))
        x = x + residual
        x = self.upsample(x)
        x = self.conv3(x)
        return x
    


class Discriminator(nn.Module):
    def __init__(self, input_shape: tuple):
        super(Discriminator, self).__init__()
        self.input_shape = input_shape
        in_height, in_width = input_shape
        final_height = in_height // 16
        final_width = in_width // 16

        self.conv_blocks = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            _ConvBlock(64, 64, kernel_size=3, stride=2, padding=1),
            _ConvBlock(64, 128, kernel_size=3, stride=1, padding=1),
            _ConvBlock(128, 128, kernel_size=3, stride=2, padding=1),
            _ConvBlock(128, 256, kernel_size=3, stride=1, padding=1),
            _ConvBlock(256, 256, kernel_size=3, stride=2, padding=1),
            _ConvBlock(256, 512, kernel_size=3, stride=1, padding=1),
            _ConvBlock(512, 512, kernel_size=3, stride=2, padding=1)
        )

        self.fc_input_size = 512 * final_height * final_width
        self.fc = nn.Sequential(
            nn.Linear(self.fc_input_size, 1024),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(1024, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.conv_blocks(x)
        x = torch.flatten(x, start_dim=1)
        x = self.fc(x)
        return x


class Discriminator_LS(nn.Module):
    def __init__(self, input_shape):
        super(Discriminator_LS, self).__init__()

        self.input_shape = (3, input_shape, input_shape)
        in_channels, in_height, in_width = self.input_shape
        patch_h, patch_w = int(in_height / 2 ** 4), int(in_width / 2 ** 4)
        self.output_shape = (1, patch_h, patch_w)

        def discriminator_block(in_filters, out_filters, first_block=False):
            layers = []
            layers.append(nn.Conv2d(in_filters, out_filters, kernel_size=3, stride=1, padding=1))
            if not first_block:
                layers.append(nn.BatchNorm2d(out_filters))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            layers.append(nn.Conv2d(out_filters, out_filters, kernel_size=3, stride=2, padding=1))
            layers.append(nn.BatchNorm2d(out_filters))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        layers = []
        in_filters = in_channels
        for i, out_filters in enumerate([64, 128, 256, 512]):
            layers.extend(discriminator_block(in_filters, out_filters, first_block=(i == 0)))
            in_filters = out_filters

        layers.append(nn.Conv2d(out_filters, 1, kernel_size=3, stride=1, padding=1))

        self.model = nn.Sequential(*layers)

    def forward(self, img):
        return self.model(img)


class VGG19FeatureExtractor(nn.Module):
    def __init__(self):
        super(VGG19FeatureExtractor, self).__init__()
        vgg19 = models.vgg19(pretrained=True).features
        self.feature_extractor = nn.Sequential(*list(vgg19.children())[:36])

        for param in self.feature_extractor.parameters():
            param.requires_grad = False

    def forward(self, x):
        return self.feature_extractor(x)