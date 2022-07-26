
from collections import OrderedDict
import torch
import torch.nn as nn
from monai.networks.blocks import  SimpleASPP, SEBlock, ChannelSELayer


class UUNet(nn.Module):
    """https://arxiv.org/pdf/2006.04868.pdf"""

    def __init__(self, in_channels=3, out_channels=1, init_features=64):
        super(UUNet, self).__init__()
        features = init_features # 64
        features2 = features * 2 # 128
        features3 = features2 * 2 # 256
        features4 = features3 * 2 # 512
        self.path = ''

        # in_channels = 6

            # self.avgpoolup1 = nn.AvgPool2d(kernel_size = 2, stride=2)

            # self.encoderup1 =  nn.Sequential(
            #                             nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            #                             UNet._blockMish(features, features * 4, name="encoderup1"))
                                        
            
            # self.bottleneckup = SimpleASPP(512, features * 4, int(features / 4), kernel_sizes=[1, 3, 3, 3], dilations=[1, 2, 4, 6])
            # self.decoderup1 = UNet._blockMish(features * 4, features, name="decoderup1")


        # VGG19
        self.encoder1gg19 = nn.Sequential(
                            nn.Conv2d(
                            in_channels=in_channels,
                            out_channels=features,
                            kernel_size=3,
                            padding=1,
                            bias=False),
                            nn.BatchNorm2d(features),
                            nn.SiLU(inplace = True),
                            nn.Conv2d(
                            in_channels=features,
                            out_channels=features,
                            kernel_size=3,
                            padding=1,
                            bias=False),
                            nn.BatchNorm2d(features),
                            nn.SiLU(inplace = True)
                            )
        
        self.encoder2gg19 = nn.Sequential(
                            nn.Conv2d(
                            in_channels=features,
                            out_channels=features2,
                            kernel_size=3,
                            padding=1,
                            bias=False),
                            nn.BatchNorm2d(features2),
                            nn.SiLU(inplace = True),
                            nn.Conv2d(
                            in_channels=features2,
                            out_channels=features2,
                            kernel_size=3,
                            padding=1,
                            bias=False),
                            nn.BatchNorm2d(features2),
                            nn.SiLU(inplace = True)
                            )

        self.encoder3gg19 = nn.Sequential(
                            nn.Conv2d(
                            in_channels=features2,
                            out_channels=features3,
                            kernel_size=3,
                            padding=1,
                            bias=False),
                            nn.BatchNorm2d(features3),
                            nn.Conv2d(
                            in_channels=features3,
                            out_channels=features3,
                            kernel_size=3,
                            padding=1,
                            bias=False),
                            nn.BatchNorm2d(features3),
                            nn.SiLU(inplace = True)
                            )

        
        self.encoder4gg19 = nn.Sequential(
                            nn.Conv2d(
                            in_channels=features3,
                            out_channels=features4,
                            kernel_size=3,
                            padding=1,
                            bias=False),
                            nn.BatchNorm2d(features4),
                            nn.SiLU(inplace = True),
                            nn.Conv2d(
                            in_channels=features4,
                            out_channels=features4,
                            kernel_size=3,
                            padding=1,
                            bias=False),
                            nn.BatchNorm2d(features4),
                            nn.SiLU(inplace = True),
                            )

        # self.encoder1 = UUNet._blockRELU(in_channels, features, name="enc1")
        self.pool1gg19 = nn.MaxPool2d(kernel_size=2, stride=2)
        # self.encoder2 = UUNet._blockRELU(features, features * 2, name="enc2")
        self.pool2gg19 = nn.MaxPool2d(kernel_size=2, stride=2)
        # self.encoder3 = UUNet._blockRELU(features * 2, features * 4, name="enc3")
        self.pool3gg19 = nn.MaxPool2d(kernel_size=2, stride=2)
        # self.encoder4 = UUNet._blockRELU(features * 4, features * 8, name="enc4")
        self.pool4gg19 = nn.MaxPool2d(kernel_size=2, stride=2)

        #self.bottleneck = UNet._blockRELU(features * 8, features * 8, name="bottleneck")
        self.bottleneckgg19 = SimpleASPP(
            2, features4, features4//4, kernel_sizes=[1, 3, 3, 3], dilations=[1, 2, 4, 6]
        )

        self.upconv4gg19 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        
        self.dec4gg19 = nn.Sequential(
                            nn.Conv2d(
                            in_channels=features4 + features4,
                            out_channels=features4,
                            kernel_size=3,
                            padding=1,
                            bias=False),
                            nn.BatchNorm2d(features4),
                            nn.SiLU(inplace = True),
                            nn.Conv2d(
                            in_channels=features4,
                            out_channels=features4,
                            kernel_size=3,
                            padding=1,
                            bias=False),
                            nn.BatchNorm2d(features4),
                            nn.SiLU(inplace = True),
                            ChannelSELayer(
                            spatial_dims = 2,
                            in_channels = features4,
                            r=8)
                            )

        self.upconv3gg19 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.dec3gg19 = nn.Sequential(
                        nn.Conv2d(
                        in_channels=features4 + features3,
                        out_channels=features3,
                        kernel_size=3,
                        padding=1,
                        bias=False),
                        nn.BatchNorm2d(features3),
                        nn.SiLU(inplace = True),
                        nn.Conv2d(
                        in_channels=features3,
                        out_channels=features3,
                        kernel_size=3,
                        padding=1,
                        bias=False),
                        nn.BatchNorm2d(features3),
                        nn.SiLU(inplace = True),
                        ChannelSELayer(
                        spatial_dims = 2,
                        in_channels = features3,
                        r=8)
                        )

        self.upconv2gg19 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.dec2gg19 = nn.Sequential(
                        nn.Conv2d(
                        in_channels=features3 + features2,
                        out_channels=features2,
                        kernel_size=3,
                        padding=1,
                        bias=False),
                        nn.BatchNorm2d(features2),
                        nn.SiLU(inplace = True),
                        nn.Conv2d(
                        in_channels=features2,
                        out_channels=features2,
                        kernel_size=3,
                        padding=1,
                        bias=False),
                        nn.BatchNorm2d(features2),
                        nn.SiLU(inplace = True),
                        ChannelSELayer(
                        spatial_dims = 2,
                        in_channels = features2,
                        r=8)
                        )

        self.upconv1gg19 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)


        self.dec1gg19 = nn.Sequential(
                        nn.Conv2d(
                        in_channels=features2 + features,
                        out_channels=features,
                        kernel_size=3,
                        padding=1,
                        bias=False),
                        nn.BatchNorm2d(features),
                        nn.SiLU(inplace = True),
                        nn.Conv2d(
                        in_channels=features,
                        out_channels=features,
                        kernel_size=3,
                        padding=1,
                        bias=False),
                        nn.BatchNorm2d(features),
                        nn.SiLU(inplace = True),
                        ChannelSELayer(
                        spatial_dims = 2,
                        in_channels = features,
                        r=8)
                        )


        self.convgg19 = nn.Conv2d(
            in_channels=features, out_channels=in_channels, kernel_size=1
        )



        # UNET
        self.encoder1U= nn.Sequential(
                        nn.Conv2d(
                        in_channels=in_channels,
                        out_channels=features,
                        kernel_size=3,
                        padding=1,
                        bias=False),
                        nn.BatchNorm2d(features),
                        nn.SiLU(inplace = True),
                        nn.Conv2d(
                        in_channels=features,
                        out_channels=features,
                        kernel_size=3,
                        padding=1,
                        bias=False),
                        nn.BatchNorm2d(features),
                        nn.SiLU(inplace = True),
                        ChannelSELayer(
                        spatial_dims = 2,
                        in_channels = features,
                        r=8)
                        )
        
        self.encoder2U = nn.Sequential(
                        nn.Conv2d(
                        in_channels=features,
                        out_channels=features2,
                        kernel_size=3,
                        padding=1,
                        bias=False),
                        nn.BatchNorm2d(features2),
                        nn.SiLU(inplace = True),
                        nn.Conv2d(
                        in_channels=features2,
                        out_channels=features2,
                        kernel_size=3,
                        padding=1,
                        bias=False),
                        nn.BatchNorm2d(features2),
                        nn.SiLU(inplace = True),
                        ChannelSELayer(
                        spatial_dims = 2,
                        in_channels = features2,
                        r=8)
                        )

        self.encoder3U = nn.Sequential(
                        nn.Conv2d(
                        in_channels=features2,
                        out_channels=features3,
                        kernel_size=3,
                        padding=1,
                        bias=False),
                        nn.BatchNorm2d(features3),
                        nn.SiLU(inplace = True),
                        nn.Conv2d(
                        in_channels=features3,
                        out_channels=features3,
                        kernel_size=3,
                        padding=1,
                        bias=False),
                        nn.BatchNorm2d(features3),
                        nn.SiLU(inplace = True),
                        ChannelSELayer(
                        spatial_dims = 2,
                        in_channels = features3,
                        r=8)
                        )

        
        self.encoder4U = nn.Sequential(
                        nn.Conv2d(
                        in_channels=features3,
                        out_channels=features4,
                        kernel_size=3,
                        padding=1,
                        bias=False),
                        nn.BatchNorm2d(features4),
                        nn.SiLU(inplace = True),
                        nn.Conv2d(
                        in_channels=features4,
                        out_channels=features4,
                        kernel_size=3,
                        padding=1,
                        bias=False),
                        nn.BatchNorm2d(features4),
                        nn.SiLU(inplace = True),
                        ChannelSELayer(
                        spatial_dims = 2,
                        in_channels = features4,
                        r=8)
                        )

        # self.encoder1 = UUNet._blockRELU(in_channels, features, name="enc1")
        self.pool1U = nn.MaxPool2d(kernel_size=2, stride=2)
        # self.encoder2 = UUNet._blockRELU(features, features * 2, name="enc2")
        self.pool2U = nn.MaxPool2d(kernel_size=2, stride=2)
        # self.encoder3 = UUNet._blockRELU(features * 2, features * 4, name="enc3")
        self.pool3U = nn.MaxPool2d(kernel_size=2, stride=2)
        # self.encoder4 = UUNet._blockRELU(features * 4, features * 8, name="enc4")
        self.pool4U = nn.MaxPool2d(kernel_size=2, stride=2)

        #self.bottleneck = UNet._blockRELU(features * 8, features * 8, name="bottleneck")
        self.bottleneckU = SimpleASPP(
            2, features4, features4//4, kernel_sizes=[1, 3, 3, 3], dilations=[1, 2, 4, 6]
        )

        self.upconv4U = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.dec4U = nn.Sequential(
                    nn.Conv2d(
                    in_channels=features4 + features4*2 ,
                    out_channels=features4 ,
                    kernel_size=3,
                    padding=1,
                    bias=False),
                    nn.BatchNorm2d(features4),
                    nn.SiLU(inplace = True),
                    nn.Conv2d(
                    in_channels=features4,
                    out_channels=features4,
                    kernel_size=3,
                    padding=1,
                    bias=False),
                    nn.BatchNorm2d(features4),
                    nn.SiLU(inplace = True),
                    ChannelSELayer(
                    spatial_dims = 2,
                    in_channels = features4,
                    r=8)
                    )

        self.upconv3U = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.dec3U = nn.Sequential(
                    nn.Conv2d(
                    in_channels=features4 + features3*2 ,
                    out_channels=features3,
                    kernel_size=3,
                    padding=1,
                    bias=False),
                    nn.BatchNorm2d(features3),
                    nn.SiLU(inplace = True),
                    nn.Conv2d(
                    in_channels=features3,
                    out_channels=features3,
                    kernel_size=3,
                    padding=1,
                    bias=False),
                    nn.BatchNorm2d(features3),
                    nn.SiLU(inplace = True),
                    ChannelSELayer(
                    spatial_dims = 2,
                    in_channels = features3,
                    r=8)
                    )

        self.upconv2U = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.dec2U = nn.Sequential(
                    nn.Conv2d(
                    in_channels=features3 + features2*2 ,
                    out_channels=features2 ,
                    kernel_size=3,
                    padding=1,
                    bias=False),
                    nn.BatchNorm2d(features2),
                    nn.SiLU(inplace = True),
                    nn.Conv2d(
                    in_channels=features2,
                    out_channels=features2,
                    kernel_size=3,
                    padding=1,
                    bias=False),
                    nn.BatchNorm2d(features2),
                    nn.SiLU(inplace = True),
                    ChannelSELayer(
                    spatial_dims = 2,
                    in_channels = features2,
                    r=8)
                    )

        self.upconv1U = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.dec1U = nn.Sequential(
                    nn.Conv2d(
                    in_channels=features2 + features*2 ,
                    out_channels=features,
                    kernel_size=3,
                    padding=1,
                    bias=False),
                    nn.BatchNorm2d(features),
                    nn.SiLU(inplace = True),
                    nn.Conv2d(
                    in_channels=features,
                    out_channels=features,
                    kernel_size=3,
                    padding=1,
                    bias=False),
                    nn.BatchNorm2d(features),
                    nn.SiLU(inplace = True),
                    ChannelSELayer(
                    spatial_dims = 2,
                    in_channels = features,
                    r=8)
                    )

        self.convU = nn.Conv2d(
            in_channels=features, out_channels=in_channels, kernel_size=1
        )

        self.conv = nn.Conv2d(
            in_channels=in_channels * 2, out_channels=out_channels, kernel_size=1
        )


    def forward(self, x):
        enc1 = self.encoder1gg19(x)
        enc2 = self.encoder2gg19(self.pool1gg19(enc1))
        enc3 = self.encoder3gg19(self.pool2gg19(enc2))
        enc4 = self.encoder4gg19(self.pool3gg19(enc3))
        
        bottleneck = self.bottleneckgg19(self.pool4gg19(enc4))

        dec4 = self.upconv4gg19(bottleneck)
        dec4 = torch.cat((dec4, enc4), dim=1)
        dec4 = self.dec4gg19(dec4)
        dec3 = self.upconv3gg19(dec4)
        dec3 = torch.cat((dec3, enc3), dim=1)
        dec3 = self.dec3gg19(dec3)
        dec2 = self.upconv2gg19(dec3)
        dec2 = torch.cat((dec2, enc2), dim=1)
        dec2 = self.dec2gg19(dec2)
        dec1 = self.upconv1gg19(dec2)
        dec1 = torch.cat((dec1, enc1), dim=1)

        dec1 = self.dec1gg19(dec1)
        output1 = self.convgg19(dec1)
        multiply = output1 * x

        enc1U = self.encoder1U(multiply)
        enc2U = self.encoder2U(self.pool1U(enc1U))
        enc3U = self.encoder3U(self.pool2U(enc2U))
        enc4U = self.encoder4U(self.pool3U(enc3U))
        
        bottleneckU = self.bottleneckU(self.pool4U(enc4U))

        dec4U = self.upconv4U(bottleneckU)
        dec4U = torch.cat((dec4U, enc4U, enc4), dim=1)
        dec4U = self.dec4U(dec4U)
        dec3U = self.upconv3U(dec4U)
        dec3U = torch.cat((dec3U, enc3U, enc3), dim=1)
        dec3U = self.dec3U(dec3U)
        dec2U = self.upconv2U(dec3U)
        dec2U = torch.cat((dec2U, enc2U, enc2), dim=1)
        dec2U = self.dec2U(dec2U)
        dec1U = self.upconv1U(dec2U)
        dec1U = torch.cat((dec1U, enc1U, enc1), dim=1)
        dec1U = self.dec1U(dec1U)

        output2 = self.convU(dec1U)

        concateoutput = torch.cat((output2, output1), dim=1)

        output = self.conv(concateoutput)

        return output


    @staticmethod
    def _blockRELU(in_channels, features, name):
        
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
                    (name + "relu1", nn.SiLU(inplace = True)),
                    #(name + "relu1", LayerInfo(name + "relu1")),
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
                    (name + "relu2", nn.SiLU(inplace = True)),


                    #(name + "relu2", LayerInfo(name + "relu2")),
                ]
            )
        )


