import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

# NOTE: arxiv
#
# scnn: 2017/12
# https://arxiv.org/pdf/1712.06080.pdf
#
# dilated_net: 2017/5
# https://towardsdatascience.com/review-dilated-convolution-semantic-segmentation-9d5a5bd768f5
# https://arxiv.org/pdf/1606.00915
#
# fcn for semantic segmentation: 2015/3
# https://towardsdatascience.com/review-fcn-semantic-segmentation-eb8c9b50d2d1
# https://arxiv.org/pdf/1411.4038.pdf
#

import config


class SCNN(nn.Module):
    def __init__(self, ms_ks=9, pretrained=True):
        """
        Argument
            ms_ks: kernel size in message passing conv
        """
        super(SCNN, self).__init__()
        self.pretrained = pretrained
        self.net_init(ms_ks)
        if not pretrained:
            self.weight_init()

        self.scale_background = 0.4
        self.scale_seg = 1.0
        self.scale_exist = 0.1

        self.ce_loss = nn.CrossEntropyLoss(
            weight=torch.tensor([self.scale_background, 1, 1, 1, 1])
        )
        self.bce_loss = nn.BCELoss()

    def forward(self, img, seg_gt=None, exist_gt=None):
        # img: [1, 3, 288, 800]
        x = self.backbone(img)
        # x: [1, 512, 36, 100]
        x = self.layer1(x)
        # x: [1, 128, 36, 100]
        x = self.message_passing_forward(x)
        # x: [1, 128, 36, 100]
        x = self.layer2(x)
        #
        # x: [1, 5, 36, 100]
        # interpolate 在这里是在做 up sampling, 36*8=288, 100*8=800
        #
        # 在 semantic segmentation 网络中最后都需要 up sampling 操作, 才能生成针
        # 对每个像素的 prob_map
        #
        # 如果暂时忽略 message_passing_forward 不关注, scnn 实际就是不断的做
        # conv2d, 把 channel 降成 5 之后, 再做一个 up sampling 恢复原始的尺寸
        #
        seg_pred = F.interpolate(x, scale_factor=8, mode="bilinear", align_corners=True)
        # seg_pred: [1, 5, 288, 800]
        x = self.layer3(x)
        # x: [1, 5, 18, 50]
        x = x.view(-1, self.fc_input_feature)
        # x: [1, 4500]
        exist_pred = self.fc(x)
        # exist_pred: [1, 4]

        if seg_gt is not None and exist_gt is not None:
            # training
            loss_seg = self.ce_loss(seg_pred, seg_gt)
            loss_exist = self.bce_loss(exist_pred, exist_gt)
            loss = loss_seg * self.scale_seg + loss_exist * self.scale_exist
        else:
            # inferencing
            loss_seg = torch.tensor(0, dtype=img.dtype, device=img.device)
            loss_exist = torch.tensor(0, dtype=img.dtype, device=img.device)
            loss = torch.tensor(0, dtype=img.dtype, device=img.device)

        return seg_pred, exist_pred, loss_seg, loss_exist, loss

    def message_passing_forward(self, x):
        # NOTE: message_passing_forward 是 SCNN 最核心的部分
        # vertical 和 reverse zip 起来, 一共需要做 4 遍 message_passing:
        # up, down, left, right
        # see https://arxiv.org/pdf/1712.06080.pdf page 3
        Vertical = [True, True, False, False]
        Reverse = [False, True, False, True]
        for ms_conv, v, r in zip(self.message_passing, Vertical, Reverse):
            x = self.message_passing_once(x, ms_conv, v, r)
        return x

    def message_passing_once(self, x, conv, vertical=True, reverse=False):
        """
        Argument:
        ----------
        x: input tensor
        vertical: vertical message passing or horizontal
        reverse: False for up-down or left-right, True for down-up or right-left
        """
        nB, C, H, W = x.shape
        # NOTE: down 时的计算步骤:
        # 1. 按 H 方向切成 H 个 (C,1,W) 的 slice
        # 2. out[0]=slice[0]
        # 3. out[1]=slice[1]+conv(out[0])
        # 4. out[2]=slice[2]+conv(out[1])
        # ...
        #
        # 计算完以后 out shape 和 input shape (基本) 是一样的, 因为 stride = 1,
        # padding=k//2, 而 O=(w+2p-k)/s+1.
        #
        # 另外整个过程和 rnn 很像: slice 是 input 序列, out 是 hidden state 序列
        #
        # 按论文的说法, 这样操作更容易捕获长条形状的物体 (比如车道线) 或大的物体
        # 的空间信息
        #
        # Intuition:
        # 为了让 cnn 能捕获大物体的信息, 需要有较大的 receptive field:
        #
        # 1. 通过 pooling
        # 2. 通过更大的 kernel
        #
        # 前者会导致信息丢失
        # 后者计算量太大
        #
        # SCNN 的作法是通过类似 rnn 的作法让空间中不同位置的信息能传递(合并)在一
        # 起, 让网络能看到整个空间的信息
        #
        if vertical:
            slices = [x[:, :, i : (i + 1), :] for i in range(H)]
            dim = 2
        else:
            slices = [x[:, :, :, i : (i + 1)] for i in range(W)]
            dim = 3
        if reverse:
            slices = slices[::-1]

        out = [slices[0]]
        for i in range(1, len(slices)):
            out.append(slices[i] + F.relu(conv(out[i - 1])))
        if reverse:
            out = out[::-1]
        return torch.cat(out, dim=dim)

    def net_init(self, ms_ks):
        input_w, input_h = config.IMAGE_W, config.IMAGE_H
        self.fc_input_feature = 5 * int(input_w / 16) * int(input_h / 16)
        # NOTE: backbone 用的是 vgg16 的特征提取部分, 但删除了 33,43 两层
        # maxpooling, 因为 pooling 会导致的空间信息丢失; 同时替换最后几层 conv2d
        # 为 dilation conv2d, 因为删除 pooling 会导致 receptive field 变小, 用
        # dilation conv2d 可以改善这种情况.
        #
        # 关于 pooling 与 semantic segmentation 参考 fcn
        #
        # 关于 dilation conv 与 semantic segmentation 参考 dilated_net
        #
        self.backbone = models.vgg16_bn(pretrained=self.pretrained).features
        # ----------------- process backbone -----------------
        for i in [34, 37, 40]:
            conv = self.backbone._modules[str(i)]
            dilated_conv = nn.Conv2d(
                conv.in_channels,
                conv.out_channels,
                conv.kernel_size,
                stride=conv.stride,
                padding=tuple(p * 2 for p in conv.padding),
                dilation=2,
                bias=(conv.bias is not None),
            )
            dilated_conv.load_state_dict(conv.state_dict())
            self.backbone._modules[str(i)] = dilated_conv
        self.backbone._modules.pop("33")
        self.backbone._modules.pop("43")

        # ----------------- SCNN part -----------------
        self.layer1 = nn.Sequential(
            nn.Conv2d(512, 1024, 3, padding=4, dilation=4, bias=False),
            nn.BatchNorm2d(1024),
            nn.ReLU(),
            nn.Conv2d(1024, 128, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(),  # (nB, 128, 36, 100)
        )

        # ----------------- add message passing -----------------
        self.message_passing = nn.ModuleList()
        self.message_passing.add_module(
            "up_down",
            nn.Conv2d(128, 128, (1, ms_ks), padding=(0, ms_ks // 2), bias=False),
        )
        self.message_passing.add_module(
            "down_up",
            nn.Conv2d(128, 128, (1, ms_ks), padding=(0, ms_ks // 2), bias=False),
        )
        self.message_passing.add_module(
            "left_right",
            nn.Conv2d(128, 128, (ms_ks, 1), padding=(ms_ks // 2, 0), bias=False),
        )
        self.message_passing.add_module(
            "right_left",
            nn.Conv2d(128, 128, (ms_ks, 1), padding=(ms_ks // 2, 0), bias=False),
        )
        # (nB, 128, 36, 100)

        # ----------------- SCNN part -----------------
        self.layer2 = nn.Sequential(
            nn.Dropout2d(0.1), nn.Conv2d(128, 5, 1)  # get (nB, 5, 36, 100)
        )

        self.layer3 = nn.Sequential(
            nn.Softmax(dim=1),  # (nB, 5, 36, 100)
            nn.AvgPool2d(2, 2),  # (nB, 5, 18, 50)
        )
        self.fc = nn.Sequential(
            nn.Linear(self.fc_input_feature, 128),
            nn.ReLU(),
            nn.Linear(128, 4),
            nn.Sigmoid(),
        )

    def weight_init(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.reset_parameters()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data[:] = 1.0
                m.bias.data.zero_()
