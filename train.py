from pathlib import Path
from dataset import UNETDataset
from model import UNet

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

"""准备训练数据"""
img_dir = Path(r"F:\科研\Pytorch-UNet\unet-img\data\membrane\train\image")
mask_dir = Path(r"F:\科研\Pytorch-UNet\unet-img\data\membrane\train\label")

train_dataset = UNETDataset(img_dir,mask_dir)
train_loader = DataLoader(train_dataset,batch_size=4,shuffle=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = UNet(n_channals=1,n_classes=1).to(device)
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(),lr=1e-3)

num_epochs = 5
for epoch in range(num_epochs):
    model.train()
    epoch_loss = 0.0
    for image,mask in tqdm(train_loader,desc="Training"):
        image=image.to(device)
        mask=mask.to(device)

        out_put = model(image)
        loss = criterion(out_put,mask)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
    avg_loss = epoch_loss / len(train_loader)
    print(f"Epoch {epoch+1} Loss: {avg_loss:.4f}")





