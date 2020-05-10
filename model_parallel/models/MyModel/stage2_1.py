# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import torch


class Stage2_1(torch.nn.Module):
    def __init__(self):
        super(Stage2_1, self).__init__()
        self.layer15 = torch.nn.Conv2d(2048, 1024, kernel_size=(1, 1), stride=(2, 2), bias=False)
        self.layer16 = torch.nn.BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.layer17 = torch.nn.ReLU(inplace=True)
        self.layer18 = torch.nn.Conv2d(1024, 1024, kernel_size=(1, 1), stride=(2, 2), bias=False)
        self.layer19 = torch.nn.BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.layer20 = torch.nn.ReLU(inplace=True)

        self.layer21 = torch.nn.Conv2d(512, 1024, kernel_size=(1, 1), stride=(2, 2), bias=False)
        self.layer22 = torch.nn.BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.layer23 = torch.nn.ReLU(inplace=True)
        self.layer24 = torch.nn.Conv2d(1024, 512, kernel_size=(1, 1), stride=(2, 2), bias=False)
        self.layer25 = torch.nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.layer26 = torch.nn.ReLU(inplace=True)

        self.layer2 = torch.nn.Conv2d(1024, 2048, kernel_size=(1, 1), stride=(2, 2), bias=False)
        self.layer3 = torch.nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.layer4 = torch.nn.ReLU(inplace=True)
        self.layer5 = torch.nn.Conv2d(512, 2048, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.layer6 = torch.nn.BatchNorm2d(2048, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.layer7 = torch.nn.BatchNorm2d(2048, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.layer9 = torch.nn.ReLU(inplace=True)
        self.layer10 = torch.nn.Conv2d(2048, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.layer11 = torch.nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.layer12 = torch.nn.ReLU(inplace=True)
        self.layer13 = torch.nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.layer14 = torch.nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

        self._initialize_weights()

    def forward(self, input0, input1):
        out0 = input1.clone()
        out1 = input0.clone()

        out15 = self.layer15(out1)
        out16 = self.layer16(out15)
        out17 = self.layer17(out16)
        out18 = self.layer18(out17)
        out19 = self.layer19(out18)
        out20 = self.layer20(out19)

        out21 = self.layer21(out0)
        out22 = self.layer22(out21)
        out23 = self.layer23(out22)
        out24 = self.layer24(out23)
        out25 = self.layer25(out24)
        out26 = self.layer26(out25)

        out2 = self.layer2(out20)
        out3 = self.layer3(out26)
        out4 = self.layer4(out3)
        out5 = self.layer5(out4)
        out6 = self.layer6(out2)
        out7 = self.layer7(out5)
        out6 += out7
        out9 = self.layer9(out6)
        out10 = self.layer10(out9)
        out11 = self.layer11(out10)
        out12 = self.layer12(out11)
        out13 = self.layer13(out12)
        out14 = self.layer14(out13)
        return (out9, out14)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, torch.nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, torch.nn.BatchNorm2d):
                torch.nn.init.constant_(m.weight, 1)
                torch.nn.init.constant_(m.bias, 0)

    def param_to_cuda1(self):
        for key, param in self._parameters.items():
            if param is not None:
                with torch.no_grad():
                    param.data.to('cuda:1')

            if param.grad is not None:
                with torch.no_grad():
                    param.grad.to('cuda:1')

        for key, buf in self._buffers.items():
            if buf is not None:
                #eachmodule.module().buffers()[key].to('cuda:1')
                buf.to('cuda:1')
