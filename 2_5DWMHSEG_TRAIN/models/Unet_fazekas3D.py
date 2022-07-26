from collections import OrderedDict
import torch.nn as nn
import torch.nn.functional as F



class Mish(nn.Module):
    def __init__(self):
        super(Mish, self).__init__()
        self.tanh = nn.Tanh()
        self.softplus = nn.Softplus()
        # self.y = 0 
    def forward(self, x):
        y = x*self.tanh(self.softplus(x))
        # self.y = y.detach() #(B, C, H, W)
        return y

class UNet3D(nn.Module):

    def __init__(self, in_channels=50, out_channels=4, init_features = 200, use_softmax_head = True):
        super(UNet3D, self).__init__()
        self.path = ''
        self.encoder1 = UNet3D._blockMish(in_channels, init_features, name="enc1")
        self.fcend = nn.Linear(in_features = init_features*512, out_features = out_channels)


    def forward(self, x):
        x = self.encoder1(x)
        # global average pooling
        # print(x.shape)
        x = x.mean([-2, -1])
        # print(x.shape)
        # flatten for input to fully-connected layer
        x = x.view(x.size(0), -1)
        x = self.fcend(x)
        return x



    @staticmethod
    def _blockMish(in_channels, features, name):

        # batchnorm1 = (name + "norm1", nn.BatchNorm2d(num_features=features))
        # batchnorm2 = (name + "norm2", nn.BatchNorm2d(num_features=features))


        batchnorm1 = (name + "norm1", nn.BatchNorm3d(features))
        batchnorm2 = (name + "norm2", nn.BatchNorm3d(features))

        return nn.Sequential(
            OrderedDict(
                [
                    (
                        name + "conv1",
                        nn.Conv3d(
                            in_channels=in_channels,
                            out_channels=features,
                            kernel_size=3,
                            padding=1,
                            bias=False,
                        ),
                    ),
                    batchnorm1,
                    (name + "relu1", Mish()),
                    #(name + "gelu1_info", LayerInfo(name+"gelu1_info")),
                    (
                        name + "conv2",
                        nn.Conv3d(
                            in_channels=features,
                            out_channels=features,
                            kernel_size=3,
                            padding=1,
                            bias=False,
                        ),
                    ),
                    batchnorm2,
                    (name + "relu2", Mish()),
                    #(name + "gelu2_info", LayerInfo(name+"gelu2_info")),
                ]
            )
        )