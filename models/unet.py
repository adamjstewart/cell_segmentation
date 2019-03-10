import torch
import torch.nn as nn


__all__ = ['UNet', 'unet23']


def double_conv(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3),
        nn.ReLU(),
        nn.Conv2d(out_channels, out_channels, 3),
        nn.ReLU()
    )


def center_crop(img, output_size):
    _, _, h, w = img.size()
    _, _, th, tw = output_size
    i = (h - th) // 2
    j = (w - tw) // 2
    return img[:, :, i:i + th, j:j + tw]


class Contract(nn.Module):

    def __init__(self, in_channels, out_channels, dropout=False):
        super(Contract, self).__init__()
        assert in_channels < out_channels

        self.pool = nn.MaxPool2d(2)
        self.conv = double_conv(in_channels, out_channels)
        self.drop = None

        if dropout:
            self.drop = nn.Dropout2d()

    def forward(self, x):
        x = self.pool(x)
        x = self.conv(x)

        if self.drop is not None:
            x = self.drop(x)

        return x


class Expand(nn.Module):

    def __init__(self, in_channels, out_channels):
        super(Expand, self).__init__()
        assert in_channels > out_channels

        self.upconv = nn.ConvTranspose2d(in_channels, out_channels, 2)
        self.relu = nn.ReLU()
        self.conv = double_conv(in_channels, out_channels)

    def forward(self, x, out):
        x = self.upconv(x)
        x = self.relu(x)

        out = center_crop(out, x.size())
        x = torch.cat([out, x], 1)

        x = self.conv(x)

        return x


class UNet(nn.Module):

    def __init__(self, in_channels=1):
        super(UNet, self).__init__()

        # Contraction
        self.conv1_2 = double_conv(in_channels, 2 ** 6)
        self.conv3_4 = Contract(2 ** 6, 2 ** 7)
        self.conv5_6 = Contract(2 ** 7, 2 ** 8)
        self.conv7_8 = Contract(2 ** 8, 2 ** 9, dropout=True)
        self.conv9_10 = Contract(2 ** 9, 2 ** 10, dropout=True)

        # Expansion
        self.conv11_13 = Expand(2 ** 10, 2 ** 9)
        self.conv14_16 = Expand(2 ** 9, 2 ** 8)
        self.conv17_19 = Expand(2 ** 8, 2 ** 7)
        self.conv20_22 = Expand(2 ** 7, 2 ** 6)
        self.conv23 = nn.Conv2d(2 ** 6, 1, 1)

        # Initialize weights
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # Contraction
        out1 = self.conv1_2(x)
        out2 = self.conv3_4(out1)
        out3 = self.conv5_6(out2)
        out4 = self.conv7_8(out3)
        x = self.conv9_10(out4)

        # Expansion
        x = self.conv11_13(x, out4)
        x = self.conv14_16(x, out3)
        x = self.conv17_19(x, out2)
        x = self.conv20_22(x, out1)
        x = self.conv23(x)


def unet23():
    return UNet()
