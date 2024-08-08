import os
import numpy
import torch
import random
from torchvision import transforms
class CongestionDataset():
    def __init__(self, root_dir, num_buckets=0, example=False, input_transform=transforms.ToTensor(), output_transform=transforms.ToTensor()):
        self.img_dir = os.path.join(root_dir, 'base_maps/feature/')
        if num_buckets==0:
            self.img_labels_dir = os.path.join(root_dir, 'base_maps/label/')
        else:
            self.img_labels_dir = os.path.join(root_dir, f'{num_buckets}buckets/label/')
        self.input_transform = input_transform
        self.output_transform = output_transform
    def __len__(self):
        return len([name for name in os.listdir(self.img_labels_dir)])
    def __getitem__(self, idx):
        self.idx = idx
        self.feature_path = os.path.join(self.img_dir, sorted(os.listdir(self.img_dir))[idx])
        self.label_path = os.path.join(self.img_labels_dir, sorted(os.listdir(self.img_labels_dir))[idx])
        with open(self.feature_path, 'rb') as f:
            image = numpy.load(f)
        with open(self.label_path, 'rb') as f:
            label = numpy.load(f)
        seed = random.randint(0, 2**32)
        if self.input_transform: 
            torch.manual_seed(seed)
            image = self.input_transform(image)
        if self.output_transform:
            torch.manual_seed(seed)
            label = self.output_transform(label)
        return image.to(torch.float16), label.to(torch.float16)
    def get_idx(self):
        id_string =  self.feature_path.split('/')[-1]
        return id_string.split('-')[0]
