#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# 2022-07-15 15:31
import torch
import torch.optim as optim
import config
from torch.utils.data import DataLoader
from tqdm import tqdm
from dataset import Tusimple
from scnn import SCNN
from util import PolyLR

device = torch.device("cuda:0")
train_dataset = Tusimple("train")
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

test_dataset = Tusimple("test")
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=True)

net = SCNN(pretrained=True)
net = net.to(device)
net.train()
optimizer = optim.SGD(net.parameters(), lr=1e-3, momentum=0.9, weight_decay=1e-4)

lr_scheduler = optim.lr_scheduler.OneCycleLR(
    optimizer,
    max_lr=5e-1,
    total_steps=400,
)


for i in range(3):
    print(f"epoch: {i}")
    progressbar = tqdm(range(len(train_loader)))
    for idx, sample in enumerate(train_loader):
        img = sample["img"].to(device)
        label = sample["label"].to(device)
        exist = sample["exist"].to(device)
        optimizer.zero_grad()
        seg_pred, exist_pred, loss = net(img, label, exist)
        loss.backward()
        optimizer.step()
        progressbar.set_description(
            "loss: {:.3f}, lr: {:.3f}".format(
                loss.item(), lr_scheduler.get_last_lr()[0]
            )
        )
        progressbar.update(1)
        lr_scheduler.step()

torch.save(
    {
        "net": net.state_dict(),
        "optim": optimizer.state_dict(),
    },
    "test.pth",
)
