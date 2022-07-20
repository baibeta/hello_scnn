#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# 2022-07-15 15:31
import torch
import torch.optim as optim
import config
import argparse
from torch.utils.data import DataLoader
from tqdm import tqdm
from dataset import Tusimple
from scnn_vgg import SCNNVgg
from scnn_mobilenet import SCNNMobileNet

device = torch.device("cuda:0")


def evaluate(args):
    print("evaluating")
    test_dataset = Tusimple("test")
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=True)

    net = None
    if args.model == "vgg":
        net = SCNNVgg(pretrained=True)
    if args.model == "mobilenet":
        net = SCNNMobileNet(pretrained=True)

    net = net.to(device)
    net.eval()
    save_dict = torch.load(net.get_model_name())
    net.load_state_dict(save_dict["net"])

    progress = tqdm(range(len(test_loader)))
    total_loss = 0.0
    for idx, sample in enumerate(test_loader):
        img = sample["img"].to(device)
        label = sample["label"].to(device)
        exist = sample["exist"].to(device)
        seg_pred, exist_pred, loss = net(img, label, exist)
        progress.set_description(f"loss: {loss.item():.3f}")
        progress.update(1)
        total_loss += loss.item()
    progress.set_description(f"mean loss: {total_loss/len(test_loader):.3f}")


def train(args):
    train_dataset = Tusimple("train")
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

    net = None
    if args.model == "vgg":
        net = SCNNVgg(pretrained=True)
    if args.model == "mobilenet":
        net = SCNNMobileNet(pretrained=True)

    net = net.to(device)
    net.train()
    optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=1e-4)

    lr_scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=config.MAX_LR,
        total_steps=args.epoch * len(train_loader),
    )

    best_loss = 65535
    if not args.reset:
        try:
            save_dict = torch.load(net.get_model_name())
            net.load_state_dict(save_dict["net"])
            optimizer.load_state_dict(save_dict["optimizer"])
            lr_scheduler.load_state_dict(save_dict["lr_scheduler"])
            best_loss.load_state_dict(save_dict["best_loss"])
            print("load pth done")
        except:
            pass

    print("training")
    for i in range(args.epoch):
        print(f"epoch: {i}")
        progress = tqdm(range(len(train_loader)))
        for idx, sample in enumerate(train_loader):
            img = sample["img"].to(device)
            label = sample["label"].to(device)
            exist = sample["exist"].to(device)
            optimizer.zero_grad()
            seg_pred, exist_pred, loss = net(img, label, exist)
            loss.backward()
            optimizer.step()
            progress.set_description(
                f"loss: {loss.item():.3f}, lr: {lr_scheduler.get_last_lr()[0]:.3f}"
            )
            progress.update(1)
            lr_scheduler.step()

            if loss.item() < best_loss:
                best_loss = loss.item()
                torch.save(
                    {
                        "net": net.state_dict(),
                        "optimimzer": optimizer.state_dict(),
                        "lr_scheduler": lr_scheduler.state_dict(),
                        "best_loss": best_loss,
                    },
                    net.get_model_name(),
                )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--epoch",
        type=int,
        default=config.EPOCH,
    )
    parser.add_argument(
        "--batch",
        type=int,
        default=config.BATCH,
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=config.LR,
    )
    parser.add_argument(
        "--reset",
        action="store_true",
    )
    parser.add_argument("--model", choices=["vgg", "mobilenet"], default="mobilenet")
    args = parser.parse_args()

    train(args)
    evaluate(args)
