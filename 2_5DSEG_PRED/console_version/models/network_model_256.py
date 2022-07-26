


from collections import OrderedDict
import torch
import torch.nn as nn


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



class UNet(nn.Module):
    """https://github.com/mateuszbuda/brain-segmentation-pytorch/blob/master/unet.py"""

    def __init__(self, in_channels=3, out_channels=1, init_features=32):
        super(UNet, self).__init__()
        features = init_features
        self.path = ''



        self.encoder1 = UNet._blockMish(in_channels, features, name="enc1")

        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder2 = UNet._blockMish(features, features * 2, name="enc2")

        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder3 = UNet._blockMish(features * 2, features * 4, name="enc3")

        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder4 = UNet._blockMish(features * 4, features * 8, name="enc4")

        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.bottleneck = UNet._blockMish(features * 8, features * 16, name="bottleneck")

        self.upconv4 = nn.ConvTranspose2d(
        features * 16, features * 8, kernel_size=2, stride=2
        )
        self.upconv3 = nn.ConvTranspose2d(
        features * 8, features * 4, kernel_size=2, stride=2
        )
        self.upconv2 = nn.ConvTranspose2d(
        features * 4, features * 2, kernel_size=2, stride=2
        )
        self.upconv1 = nn.ConvTranspose2d(
        features * 2, features, kernel_size=2, stride=2
        )

        self.decoder4 = UNet._blockMish((features * 8) * 2, features * 8, name="dec4")


        self.decoder3 = UNet._blockMish((features * 4) * 2, features * 4, name="dec3")


        self.decoder2 = UNet._blockMish((features * 2) * 2, features * 2, name="dec2")


        self.decoder1 = UNet._blockMish(features * 2, features, name="dec1")

        self.conv = nn.Conv2d(
            in_channels=features, out_channels=out_channels, kernel_size=1
        )

        # self.gammablock = Gammablock(in_channels)
        #self.augmentflipblock = AugmentFlipModule()

        ##### DIAGNOSTICS
        #self.diagnostics = diagnostics
        self.epoch_counter = 0
        self.grads = {}
        self.batches = 0


    def forward(self, x):
        #x1 = self.augmentflipblock(x) # 
        #del x1

        # s = x.detach().cpu().numpy()
        # x1 = self.gammablock(x)
        # x = torch.cat((x, x1), dim = 1)

        # fig, ax = plt.subplots(1, 2)
        # ax[0].imshow(x.detach().cpu().numpy()[0, 1, :, :])
        # ax[1].imshow(s[0, 1, :, :])
        # plt.show()



        enc1 = self.encoder1(x)
        enc2 = self.encoder2(self.pool1(enc1))
        enc3 = self.encoder3(self.pool2(enc2))
        enc4 = self.encoder4(self.pool3(enc3))
        bottleneck = self.bottleneck(self.pool4(enc4))
        self.bottleneckflattened = bottleneck.detach().cpu().numpy()
        dec4 = self.upconv4(bottleneck)
        dec4 = torch.cat((dec4, enc4), dim=1)
        dec4 = self.decoder4(dec4)
        dec3 = self.upconv3(dec4)
        dec3 = torch.cat((dec3, enc3), dim=1)
        dec3 = self.decoder3(dec3)
        dec2 = self.upconv2(dec3)
        dec2 = torch.cat((dec2, enc2), dim=1)
        dec2 = self.decoder2(dec2)
        dec1 = self.upconv1(dec2)
        dec1 = torch.cat((dec1, enc1), dim=1)
        dec1 = self.decoder1(dec1)
        #y = torch.sigmoid(self.conv(dec1))
        y = self.conv(dec1)
        return y

    def downscale(self, x):
        x = self.maxpool(x)
        return x
    


    @staticmethod
    def _blockMish(in_channels, features, name):

        # batchnorm1 = (name + "norm1", nn.BatchNorm2d(num_features=features))
        # batchnorm2 = (name + "norm2", nn.BatchNorm2d(num_features=features))

        batchnorm1 = (name + "norm1", nn.BatchNorm2d(features))
        batchnorm2 = (name + "norm2", nn.BatchNorm2d(features))

        return nn.Sequential(
            OrderedDict(
                [
                    (
                        name + "conv1",
                        nn.Conv2d(
                            in_channels=in_channels,
                            out_channels=features,
                            kernel_size=3,
                            padding=1,
                            #padding_mode = 'reflect',
                            bias=False,
                        ),
                    ),
                    batchnorm1,
                    (name + "mish1", Mish()),
                    #(name + "gelu1_info", LayerInfo(name+"gelu1_info")),
                    (
                        name + "conv2",
                        nn.Conv2d(
                            in_channels=features,
                            out_channels=features,
                            kernel_size=3,
                            padding=1,
                            #padding_mode = 'reflect',
                            bias=False,
                        ),
                    ),
                    batchnorm2,
                    (name + "mish2", Mish()),
                    #(name + "gelu2_info", LayerInfo(name+"gelu2_info")),
                ]
            )
        )