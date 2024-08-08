import torch
import random
from congestion_model import CongestionModel
from vit import ViTForClassification
from unet import UNet
from dataset import CongestionDataset
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
from pytorch_msssim import SSIM
import argparse
import torchvision.transforms.v2 as transforms

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def train(rootpath,batch_size,num_epochs,lr,modelname,fig_savepath,weight_savepath,weight_decay=0.001):


    #data
    dataset = CongestionDataset(
        root_dir = "../data/",
        num_buckets = 0,
        input_transform = transforms.Compose([
            transforms.ToImage(),
            transforms.ToDtype(torch.float32, scale=True),
            transforms.Resize((256, 256)),
            transforms.RandomHorizontalFlip(0.5),
            transforms.RandomVerticalFlip(0.5),
            transforms.RandomRotation(90)
        ]),
        output_transform = transforms.Compose([
            transforms.ToImage(),
            transforms.ToDtype(torch.float32, scale=True),
            transforms.Resize((256, 256)),
            transforms.RandomHorizontalFlip(0.5),
            transforms.RandomVerticalFlip(0.5),
            transforms.RandomRotation(90)
        ])
    )
    len_train_set = int(len(dataset)*0.7)
    len_test_set = int(len(dataset)-len_train_set)
    train_set, test_set = torch.utils.data.random_split(dataset, [len_train_set, len_test_set])
    train_loader = DataLoader(dataset=train_set, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(dataset=test_set, batch_size=batch_size, shuffle=True)

    if modelname == 'FPN':
        model = CongestionModel(device).to(device)
    elif modelname == 'ViT':
        model = ViTForClassification(config={
            "patch_size": 32,
            "hidden_size": 48,
            "num_hidden_layers": 24,
            "num_attention_heads": 32,
            "intermediate_size": 4 * 48,
            "hidden_dropout_prob": 0.0,
            "attention_probs_dropout_prob": 0.0,
            "initializer_range": 0.02,
            "image_size": 256,
            "num_classes": 65536,
            "num_channels": 3,
            "qkv_bias": True}).to(device)
    elif modelname == 'UNet':
        model = UNet().to(device)

    #criterion
    ssim = SSIM(data_range=1, size_average=True, channel=1)
    criterion = torch.nn.BCEWithLogitsLoss()
    #optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    #FP16
    scaler = torch.cuda.amp.GradScaler()

    print('Start training')
    train_losses = []
    valid_losses = []
    train_ssims = []
    test_ssims = []
    best_test_Loss = 99999999999999
    best_train_Loss = 99999999999999

    for e in tqdm(range(num_epochs), desc='Epoch'):
        t = 0
        s = 0
        n1 = 0
        for batch_idx, (features, labels) in enumerate(train_loader):
            features = features.to(device=device)
            labels = labels.to(device=device)
            # forward
            with torch.cuda.amp.autocast():
                pred = model(features)
                pred2 = torch.sigmoid(pred)
                train_loss = criterion(pred,labels)
                train_ssim = (1-ssim(pred2,labels.type(torch.float16)))
            if train_ssim < 0.1:
                compare_img(pred2,labels,model,ssim,dataset.get_idx())
            # backward
            optimizer.zero_grad()
            scaler.scale(train_loss).backward()
            scaler.step(optimizer)
            scaler.update()

            t += train_loss.item()
            s += train_ssim.item()
            n1 += 1
        train_losses.append(t/n1)
        train_ssims.append(s/n1)

        #eval
        model.eval()
        v = 0
        s2 = 0
        n2 = 0
        for batch_idx, (features, labels) in enumerate(test_loader):
            features = features.to(device=device)
            labels = labels.to(device=device)
            with torch.cuda.amp.autocast():
                pred = model(features)
                pred2 = torch.sigmoid(pred)
                test_ssim = (1-ssim(pred2,labels.type(torch.float16)))
                test_loss = criterion(pred, labels)
            if dataset.get_idx() == '10370':
                compare_img(pred2,labels,model,ssim,dataset.get_idx())
            v += test_loss.item()
            s2 += test_ssim.item()
            n2 += 1
        valid_losses.append(v/n2)
        test_ssims.append(s2/n2)


        print("\n")
        print(f'Epoch {e}: Train Loss: {t/n1}  | Test Loss: {v/n2}')
        print(f'Epoch {e}: Train SSIM: {s/n1} | Test SSIM: {s2/n2}')
        if s2/n2 < best_test_Loss:
            print(f'Best Epoch {e}: Test SSIM: {s2/n2}')
            torch.save(model.state_dict(), f'{weight_savepath}/congestion_weights.pt')
            best_test_Loss = s2/n2
        if t/n1 < best_train_Loss:
            print(f'Best Epoch {e}: Train Loss: {t/n1}')
            torch.save(model.state_dict(), f'{weight_savepath}/congestion_train_weights.pt')
            best_train_Loss = t/n1

    fig = plt.figure()
    epochnum = list(range(0,len(train_losses)))
    plt.plot(epochnum, train_losses, color='black', linewidth=1)
    plt.plot(epochnum, valid_losses, color='red', linewidth=1)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.ylim([0,1.5])
    plt.legend("Train", loc='best',fontsize=16)
    plt.legend(("Val"), loc='best',fontsize=16)
    plt.title("Loss")
    plt.grid(linestyle=':')
    plt.savefig(f"{num_epochs}|{lr}|{weight_decay}|{modelname}loss.png")
    plt.clf()

    fig = plt.figure()
    epochnum = list(range(0,len(train_losses)))
    plt.plot(epochnum, train_ssims, color='black', linewidth=1)
    plt.plot(epochnum, test_ssims, color='red', linewidth=1)
    plt.xlabel('Epoch')
    plt.ylabel('SSIM')
    plt.ylim([0, 1])
    plt.title("SSIM")
    plt.grid(linestyle=':')
    plt.savefig(f"{num_epochs}|{lr}|{weight_decay}|{modelname}SSIM.png")
    plt.clf()

def compare_img(pred, labels, model, scorer, idx):
    fig, ax = plt.subplots(1, 2, figsize=(9, 4.5), tight_layout=True)
    score = 1-scorer(pred.type(torch.float16),labels.type(torch.float16))
    ax[0].imshow(pred[0,0].detach().cpu())
    ax[1].imshow(labels[0,0].cpu())
    ax[0].title.set_text(f'Pred     SSIM: {score.item()}')
    ax[1].title.set_text('Label')
    plt.savefig(f"goodSSIM.png")
    plt.clf()

def parse_args():
    description = "Input the Path for Prediction"
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--root_path", default="./CircuitNet/dataset/congestion", type=str, help='The path of the data file')
    parser.add_argument("--batch_size", default=32, type=int, help='The batch size')
    parser.add_argument("--num_epochs", default=45, type=int, help='The training epochs')
    parser.add_argument("--weight_path", default="./model_weight", type=str, help='The path to save the model weight')
    parser.add_argument("--fig_path", default="/save_img", type=str, help='The path of the figure file')
    parser.add_argument("--learning_rate", default=0.001, type=float, help='learning rate [0,1]')
    args = parser.parse_args()
    return args
if __name__ == "__main__":
    import time
    start = time.time()
    args = parse_args()
    params = {'FPN': 0.001, 'ViT': 0.001, 'UNet': 0.0003}
    for k, v in params.items():
        train(rootpath=args.root_path,batch_size=args.batch_size,num_epochs=args.num_epochs,lr=v,modelname=k,
                fig_savepath=args.fig_path,weight_savepath=args.weight_path)
        end = time.time()
        print("training cost time：%f sec" % (end - start))
