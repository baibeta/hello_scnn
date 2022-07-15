#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# 2021-12-30 19:14
import json
import os

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
import util
import config


class Tusimple(Dataset):
    def __init__(self, path, mode):
        super(Tusimple, self).__init__()
        self.data_path = path
        self.mode = mode
        self.loadData()

    def loadData(self):
        self.img_list = []
        self.segLabel_list = []
        self.exist_list = []

        listfile = os.path.join(
            self.data_path, "seg_label", "list", "{}_gt.txt".format(self.mode)
        )
        with open(listfile) as f:
            for line in f:
                line = line.strip()
                l = line.split(" ")
                # segLabel_list 格式是 [xxx.png, yyy.png, zzz.png ...]
                # exist_list 格式是 [[0 0 0 0], [1,1,0,0],[1,0,1,1],...]
                self.img_list.append(os.path.join(self.data_path, l[0][1:]))
                self.segLabel_list.append(os.path.join(self.data_path, l[1][1:]))
                self.exist_list.append([int(x) for x in l[2:]])

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):
        img = cv2.imread(self.img_list[idx])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = util.normalize(
            util.to_tensor(util.resize(img, (config.IMAGE_W, config.IMAGE_H))),
            config.MEAN,
            config.STD,
        )
        # NOTE: imread 返回的是一个 BGR 的图片, 但这里只需要 b,g,r 的某一个来代
        # 表像素所属的分类, 因为 bgr 在 generate_label 时写的是值是相同的
        label = cv2.imread(self.segLabel_list[idx])[:, :, 0]
        label = util.resize(label, (config.IMAGE_W, config.IMAGE_H))
        label = torch.from_numpy(label).type(torch.long)

        exist = np.array(self.exist_list[idx])
        exist = torch.from_numpy(exist).type(torch.float32)

        sample = {
            "img": img,
            "label": label,
            "exist": exist,
            "img_name": self.img_list[idx],
        }
        return sample


if __name__ == "__main__":
    data = Tusimple(config.DATA_PATH, "train")
    print(data[1])
