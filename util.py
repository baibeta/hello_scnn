#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# 2022-07-15 11:49
import cv2
import torch
import torchvision


def resize(img, size):
    img = cv2.resize(img, size, interpolation=cv2.INTER_CUBIC)
    return img


def to_tensor(img):
    # NOTE: opencv2 的格式为 HWC, 模型的输入为 CHW
    img = img.transpose(2, 0, 1)
    img = torch.from_numpy(img).type(torch.float) / 255.0
    return img


def normalize(img, mean, std):
    return torchvision.transforms.Normalize(mean, std)(img)
