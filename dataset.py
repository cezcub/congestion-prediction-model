import os
import numpy

class CongestionDataset():
    def __init__(self, root_dir, transform=None):
        self.cong_labels = os.path.join(root_dir, 'labels/label/')
        self.cong_dir = os.path.join(root_dir, 'features/feature/')
        self.transform = transform

    def __len__(self):
        return len([name for name in os.listdir(self.cong_labels)])


    def __getitem__(self, idx):
        with open(os.path.join(self.cong_dir, sorted(os.listdir(self.cong_dir))[idx]), 'rb') as f:
            image = numpy.load(f)
        with open(os.path.join(self.cong_labels, sorted(os.listdir(self.cong_labels))[idx]), 'rb') as f2:
            label = numpy.load(f2)
        if self.transform:
            self.transform(image)
        return (image, label)
