#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# 2021-12-30 15:06
#
# NOTE: code from https://github.com/harryhan618/SCNN_Pytorch
#
import cv2
import torch
import torchvision

import numpy as np
import torch.nn.functional as F

from scnn import SCNN
import util
import config

net = SCNN(pretrained=False)
save_dict = torch.load("test.pth")
net.load_state_dict(save_dict["net"])
net.eval()

# NOTE: 关于 scnn 的输出
#
# scnn 的输出有两个: seg_pred 和 exist_pred
#
# seg_pred: [5, 288, 800], 表示每一个像素所属 class 的概率.
#
# 0 表示属于 backgroud 的概率, 即这个像素不属于任何一条车道线.
# 1 表示属于车道线 0 的概率
# ...
# 4 表示属于车道线 3 的概率, 一共有 4 条车道线
#
# exist_pred: [1, 4], 表示检测到有相应车道的概率
#
# seg_pred 这种针对所有像素的分类概率叫做 prob_map. lane detection 本质上是一个
# 语义分割的问题
#
# 有了 prob_map 后, 那些 argmax(prob_map,axis=0) == 1 的点就是预测出来的车道线 0
# 上的点

with torch.no_grad():
    img = cv2.imread("hello_tusimple.jpg")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    img = util.resize(img, (config.IMAGE_W, config.IMAGE_H))
    data = util.normalize(util.to_tensor(img), config.MEAN, config.STD)
    data.unsqueeze_(0)

    seg_pred, exist_pred, _ = net(data)
    seg_pred = F.softmax(seg_pred, dim=1)
    seg_pred = seg_pred.detach().cpu().numpy()
    exist_pred = exist_pred.detach().cpu().numpy()

    seg_pred = seg_pred[0]
    exist = [1 if exist_pred[0, i] > 0.5 else 0 for i in range(4)]

    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    lane_img = np.zeros_like(img)

    color = np.array(
        [[255, 125, 0], [0, 255, 0], [0, 0, 255], [0, 255, 255]], dtype="uint8"
    )
    coord_mask = np.argmax(seg_pred, axis=0)
    for i in range(0, 4):
        # if exist_pred[0, i] > 0.5:
        lane_img[coord_mask == (i + 1)] = color[i]

    img = cv2.addWeighted(src1=lane_img, alpha=0.5, src2=img, beta=1.0, gamma=0.0)
    cv2.imwrite("test.jpg", img)
    cv2.imshow("", img)

    while True:
        k = cv2.waitKey(0) & 0xFF
        if k == ord("q"):
            cv2.destroyAllWindows()
            break
