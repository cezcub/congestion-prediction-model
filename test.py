import matplotlib.pyplot as plt
from dataset import CongestionDataset
from congestion_model import CongestionModel
import numpy as np
import torch
from torchvision import transforms
from torch.utils.data import DataLoader
from pytorch_msssim import SSIM

ssim = SSIM(data_range=1, size_average=True, channel=1)
dataset = CongestionDataset(root_dir = "../data/example", num_buckets = 0, example=True, input_transform = transforms.Compose([transforms.ToTensor(), transforms.Resize((256, 256))]), output_transform = transforms.Compose([transforms.ToTensor(), transforms.Resize((256, 256))]))
train_loader = DataLoader(dataset=dataset, batch_size=1)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = CongestionModel(device).to(device)
model.load_state_dict(torch.load('model_weight/congestion_weights.pt'))
for batch_idx, (features, labels) in enumerate(train_loader):
    features = features.to(device=device)
    labels = labels.to(device=device)
pred = model(features)


fig, ax = plt.subplots(1, 2, figsize=(9, 4.5), tight_layout=True)
pred = model.sigmoid(pred)
score = 1-ssim(pred.type(torch.float16),labels.type(torch.float16))
labels[labels>0.2]=1
labels[labels<0.2]=0
ax[0].imshow(pred[0,0].detach().cpu())
ax[1].imshow(labels[0,0].cpu())
ax[0].title.set_text(f'Pred     SSIM:{score}')
ax[1].title.set_text('Label')
plt.savefig(f"./comparethresh.png")
plt.clf()
