import torch
import torch.nn as nn


def center_crop(img, output_size):
    _, _, h, w = img.size()
    _, _, th, tw = output_size
    i = (h - th) // 2
    j = (w - tw) // 2
    return img[:, :, i:i + th, j:j + tw]


class UNet(nn.Module):

    def __init__(self):
        super(UNet, self).__init__()

        self.relu = nn.Relu()
        self.pool = nn.MaxPool2d(2)
        self.drop = nn.Dropout2d()

        # Contraction
        self.conv1 = nn.Conv2d(1, 64, 3)
        self.conv2 = nn.Conv2d(64, 64, 3)

        self.conv3 = nn.Conv2d(64, 128, 3)
        self.conv4 = nn.Conv2d(128, 128, 3)

        self.conv5 = nn.Conv2d(128, 256, 3)
        self.conv6 = nn.Conv2d(256, 256, 3)

        self.conv7 = nn.Conv2d(256, 512, 3)
        self.conv8 = nn.Conv2d(512, 512, 3)

        self.conv9 = nn.Conv2d(512, 1024, 3)
        self.conv10 = nn.Conv2d(1024, 1024, 3)

        # Expansion
        self.conv11 = nn.ConvTranspose2d(1024, 512, 2)
        self.conv12 = nn.Conv2d(1024, 512, 3)
        self.conv13 = nn.Conv2d(512, 512, 3)

        self.conv14 = nn.ConvTranspose2d(512, 256, 2)
        self.conv15 = nn.Conv2d(512, 256, 3)
        self.conv16 = nn.Conv2d(256, 256, 3)

        self.conv17 = nn.ConvTranspose2d(256, 128, 2)
        self.conv18 = nn.Conv2d(256, 128, 3)
        self.conv19 = nn.Conv2d(128, 128, 3)

        self.conv20 = nn.ConvTranspose2d(128, 64, 2)
        self.conv21 = nn.Conv2d(128, 64, 3)
        self.conv22 = nn.Conv2d(64, 64, 3)
        self.conv23 = nn.Conv2d(64, 1, 1)

        # Initialize weights
        for m in self.modules():
            nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
            nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # Contraction
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        out1 = self.relu(x)

        x = self.pool(out1)
        x = self.conv3(x)
        x = self.relu(x)
        x = self.conv4(x)
        out2 = self.relu(x)

        x = self.pool(out2)
        x = self.conv5(x)
        x = self.relu(x)
        x = self.conv6(x)
        out3 = self.relu(x)

        x = self.pool(out3)
        x = self.conv7(x)
        x = self.relu(x)
        x = self.conv8(x)
        x = self.relu(x)
        out4 = self.drop(x)

        x = self.pool(out4)
        x = self.conv9(x)
        x = self.relu(x)
        x = self.conv10(x)
        x = self.relu(x)
        x = self.drop(x)

        # Expansion
        x = self.conv11(x)
        x = self.relu(x)
        out4 = center_crop(out4, x.size())
        x = torch.cat([out4, x], 1)
        x = self.conv12(x)
        x = self.relu(x)
        x = self.conv13(x)
        x = self.relu(x)

        x = self.conv14(x)
        x = self.relu(x)
        out3 = center_crop(out3, x.size())
        x = torch.cat([out3, x], 1)
        x = self.conv15(x)
        x = self.relu(x)
        x = self.conv16(x)
        x = self.relu(x)

        x = self.conv17(x)
        x = self.relu(x)
        out2 = center_crop(out2, x.size())
        x = torch.cat([out2, x], 1)
        x = self.conv18(x)
        x = self.relu(x)
        x = self.conv19(x)
        x = self.relu(x)

        x = self.conv20(x)
        x = self.relu(x)
        out1 = center_crop(out1, x.size())
        x = torch.cat([out1, x], 1)
        x = self.conv21(x)
        x = self.relu(x)
        x = self.conv22(x)
        x = self.relu(x)

        x = self.conv23(x)


def unet23():
    pass
