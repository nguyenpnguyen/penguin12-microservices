import torch
import torch.nn as nn
import torch.nn.functional as F

class SEBlock(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // reduction, in_channels),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = F.adaptive_avg_pool2d(x, 1).view(b, c)   # squeeze
        y = self.fc(y).view(b, c, 1, 1)              # excitation
        return x * y                                 # scale

class Penguin12CNN(nn.Module):
    def __init__(self):
        super(Penguin12CNN, self).__init__()

        def conv_block(in_channels, out_channels, kernel_size=3, dropout=0.3, use_se=False):
            padding = kernel_size // 2
            layers = [
                nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding),
                nn.BatchNorm2d(out_channels),
                nn.SiLU(),
                nn.Conv2d(out_channels, out_channels, kernel_size, padding=padding),
                nn.BatchNorm2d(out_channels),
                nn.SiLU(),
            ]
            if use_se:
                layers.append(SEBlock(out_channels))
            layers += [
                nn.MaxPool2d(2),
                nn.Dropout(dropout)
            ]
            return nn.Sequential(*layers)

        # feature extractor
        self.features = nn.Sequential(
            conv_block(3, 32, kernel_size=3, dropout=0.2, use_se=False),    # 64x64
            conv_block(32, 64, kernel_size=3, dropout=0.25, use_se=True),   # 32x32
            conv_block(64, 128, kernel_size=3, dropout=0.3, use_se=True),   # 16x16
            conv_block(128, 256, kernel_size=3, dropout=0.4, use_se=True),  # 8x8
            conv_block(256, 512, kernel_size=3, dropout=0.5, use_se=True),  # 4x4
        )

        # global average + max pooling
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.global_max_pool = nn.AdaptiveMaxPool2d((1, 1))

    def forward(self, x):
        x = self.features(x)

        # dual pooling
        x_avg = self.global_avg_pool(x)
        x_max = self.global_max_pool(x)
        x = torch.cat([x_avg, x_max], dim=1)

        return x
