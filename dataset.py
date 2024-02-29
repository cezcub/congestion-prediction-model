import os
import numpy
import torch
from torchvision import transforms
class CongestionDataset():
    def __init__(self,
                 root_dir,
                 input_transform = transforms.ToTensor(),
                 output_transform =transforms.ToTensor()
                ):
        self.img_labels_dir = os.path.join(
            root_dir,
            'label/'
        )
        self.img_dir = os.path.join(
            root_dir,
            'feature/'
        )
        self.input_transform = input_transform
        self.output_transform = output_transform
    def __len__(self):
        return len([name for name in os.listdir(self.img_labels_dir)])
    def __getitem__(self, idx):
        with open(os.path.join(self.img_dir, sorted(os.listdir(self.img_dir))[idx]), 'rb') as f:
            image = numpy.load(f)
        with open(os.path.join(self.img_labels_dir, sorted(os.listdir(self.img_labels_dir))[idx]), 'rb') as f:
            label = numpy.load(f)
        if self.input_transform:
            image = self.input_transform(image)
        if self.output_transform:
            label = self.output_transform(label)
        return image.to(torch.float16), label.to(torch.float16)
