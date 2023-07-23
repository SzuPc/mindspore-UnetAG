# Copyright 2020 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================

""" Parts of the U-Net model """

import mindspore.nn as nn
import mindspore.ops.operations as F
from mindspore.common.initializer import TruncatedNormal
from mindspore.nn import CentralCrop
import mindspore.ops as ops


# from test_model import Attention_block
# 这个算子在2.0版本中弃用了 mindspore.dataset.vision.CenterCrop(）进行替换

# class Attention_block(nn.Cell):
#     def __init__(self, F_g, F_l, F_int):
#         super(Attention_block,self).__init__()
#         self.W_g = nn.Sequential(
#             nn.Conv2d(F_g, F_int, kernel_size=1,stride=1,padding=0,bias=True),
#             nn.BatchNorm2d(F_int)
#             )

#         self.W_x = nn.Sequential(
#             nn.Conv2d(F_l, F_int, kernel_size=1,stride=1,padding=0,bias=True),
#             nn.BatchNorm2d(F_int)
#         )

#         self.psi = nn.Sequential(
#             nn.Conv2d(F_int, 1, kernel_size=1,stride=1,padding=0,bias=True),
#             nn.BatchNorm2d(1),
#             nn.Sigmoid()
#         )


class Attention_block(nn.Cell):
    def __init__(self, F_g, F_l, F_int):
        super(Attention_block, self).__init__()
        self.W_g = nn.SequentialCell(
            nn.Conv2d(F_g, F_int, kernel_size=3, has_bias=True, bias_init="zeros", pad_mode="same",
                      weight_init="normal"),
            nn.BatchNorm2d(F_int)
        )

        self.W_x = nn.SequentialCell(
            nn.Conv2d(F_l, F_int, kernel_size=3, has_bias=True, bias_init="zeros", pad_mode="same",
                      weight_init="normal"),
            nn.BatchNorm2d(F_int)
        )

        self.psi = nn.SequentialCell(
            nn.Conv2d(F_int, 1, kernel_size=3, has_bias=True, bias_init="zeros", pad_mode="same", weight_init="normal"),
            nn.BatchNorm2d(1),
        )
        self.relu = nn.ReLU()

    def construct(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        psi = ops.sigmoid(psi)

        return x * psi


class DoubleConv(nn.Cell):

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        init_value_0 = TruncatedNormal(0.06)
        init_value_1 = TruncatedNormal(0.06)
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.SequentialCell(
            [nn.Conv2d(in_channels, mid_channels, kernel_size=3, has_bias=True,
                       weight_init=init_value_0, bias_init="zeros", pad_mode="valid"),
             nn.ReLU(),
             nn.Conv2d(mid_channels, out_channels, kernel_size=3, has_bias=True,
                       weight_init=init_value_1, bias_init="zeros", pad_mode="valid"),
             nn.ReLU()]
        )

    def construct(self, x):
        return self.double_conv(x)


class Down(nn.Cell):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.maxpool_conv = nn.SequentialCell(
            [nn.MaxPool2d(kernel_size=2, stride=2),
             DoubleConv(in_channels, out_channels)]
        )

    def construct(self, x):
        return self.maxpool_conv(x)


class Up1(nn.Cell):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()
        self.concat = F.Concat(axis=1)
        self.factor = 56.0 / 64.0
        self.center_crop = CentralCrop(central_fraction=self.factor)
        self.print_fn = F.Print()
        self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        self.up = nn.Conv2dTranspose(in_channels, in_channels // 2, kernel_size=2, stride=2,
                                     weight_init="normal", bias_init="zeros")
        self.relu = nn.ReLU()
        self.Att1 = Attention_block(F_g=512, F_l=512, F_int=256)

    def construct(self, x1, x2):
        x1 = self.up(x1)
        x1 = self.relu(x1)
        x2 = self.center_crop(x2)
        x2 = self.Att1(g=x1, x=x2)
        x = self.concat((x1, x2))
        return self.conv(x)


class Up2(nn.Cell):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()
        self.concat = F.Concat(axis=1)
        self.factor = 104.0 / 136.0
        self.center_crop = CentralCrop(central_fraction=self.factor)
        self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        self.up = nn.Conv2dTranspose(in_channels, in_channels // 2, kernel_size=2, stride=2,
                                     weight_init="normal", bias_init="zeros")
        self.relu = nn.ReLU()
        self.Att2 = Attention_block(F_g=256, F_l=256, F_int=128)

    def construct(self, x1, x2):
        x1 = self.up(x1)
        x1 = self.relu(x1)
        x2 = self.center_crop(x2)
        x2 = self.Att2(g=x1, x=x2)
        x = self.concat((x1, x2))
        return self.conv(x)


class Up3(nn.Cell):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()
        self.concat = F.Concat(axis=1)
        self.factor = 200 / 280
        self.center_crop = CentralCrop(central_fraction=self.factor)
        self.print_fn = F.Print()
        self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        self.up = nn.Conv2dTranspose(in_channels, in_channels // 2, kernel_size=2, stride=2,
                                     weight_init="normal", bias_init="zeros")
        self.relu = nn.ReLU()
        self.Att3 = Attention_block(F_g=128, F_l=128, F_int=64)

    def construct(self, x1, x2):
        x1 = self.up(x1)
        x1 = self.relu(x1)
        x2 = self.center_crop(x2)
        x2 = self.Att3(g=x1, x=x2)
        x = self.concat((x1, x2))
        return self.conv(x)


class Up4(nn.Cell):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()
        self.concat = F.Concat(axis=1)
        self.factor = 392 / 568
        self.center_crop = CentralCrop(central_fraction=self.factor)
        self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        self.up = nn.Conv2dTranspose(in_channels, in_channels // 2, kernel_size=2, stride=2,
                                     weight_init="normal", bias_init="zeros")
        self.relu = nn.ReLU()
        self.Att4 = Attention_block(F_g=64, F_l=64, F_int=32)

    def construct(self, x1, x2):
        x1 = self.up(x1)
        x1 = self.relu(x1)
        x2 = self.center_crop(x2)
        x2 = self.Att4(g=x1, x=x2)
        x = self.concat((x1, x2))
        return self.conv(x)


class OutConv(nn.Cell):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        init_value = TruncatedNormal(0.06)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, has_bias=True, weight_init=init_value,
                              bias_init="zeros")

    def construct(self, x):
        x = self.conv(x)
        return x
