from dataset import CongestionDataset
from torch.utils.data import DataLoader
from torchvision import transforms

root_dir = '../data/'
#data_transform = transforms.Compose([transforms.Resize((200, 200, 1)), transforms.ToTensor()])
custom_dataset = CongestionDataset(root_dir, transform=None)

data = DataLoader(dataset=custom_dataset, batch_size=64)


