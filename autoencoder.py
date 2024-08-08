import torch
import torch.nn as nn

class Autoencoder(nn.Module):
  def __init__(self):
    super(Autoencoder, self).__init__()
    # Encoder layers
    self.encoder = nn.Sequential(
      nn.Conv2d(3, 32, kernel_size=3, padding=1, stride=1),
      nn.BatchNorm2d(32),
      nn.ReLU(True),
      nn.MaxPool2d(kernel_size=2, stride=2),
      nn.Conv2d(32, 64, kernel_size=3, padding=1, stride=1),
      nn.BatchNorm2d(64),
      nn.ReLU(True),
      nn.MaxPool2d(kernel_size=2, stride=2),
      nn.Conv2d(64, 128, kernel_size=3, padding=1, stride=1),
      nn.BatchNorm2d(128),
      nn.ReLU(True),
      nn.MaxPool2d(kernel_size=2, stride=2),
      nn.Conv2d(128, 256, kernel_size=3, padding=1, stride=1),
      nn.BatchNorm2d(256),
      nn.ReLU(True),
      nn.MaxPool2d(kernel_size=2, stride=2)
    )
    # Decoder layers
    self.decoder = nn.Sequential(
      nn.ConvTranspose2d(256, 128, kernel_size=3, output_padding=1, stride=2),
      nn.BatchNorm2d(128),
      nn.ReLU(True),
      nn.ConvTranspose2d(128, 64, kernel_size=3, output_padding=1, stride=2),
      nn.BatchNorm2d(64),
      nn.ReLU(True),
      nn.ConvTranspose2d(64, 32, kernel_size=3, output_padding=1, stride=2),
      nn.BatchNorm2d(32),
      nn.ReLU(True),
      nn.ConvTranspose2d(32, 1, kernel_size=3, output_padding=1, stride=2),
      nn.BatchNorm2d(1),
      nn.ReLU(True),
      nn.MaxPool2d(kernel_size=31, stride=1)
    )
  def forward(self, x):
    x = self.encoder(x)
    x = self.decoder(x)
    return x
