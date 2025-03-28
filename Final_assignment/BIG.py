import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self, in_channels=3, n_classes=19):
        
        super(Model, self).__init__()

        self.dconv11 = (DoubleConv(in_channels, 64))
        self.dconv12 = (DoubleConv(64, 64))
        self.down112 = (Down(64, 128))
        self.dconv21 = (DoubleConv(128, 128))
        self.up221 = (Up(2)) ## till here it works!
        self.down212 = (Down(64, 128))
        self.down223 = (Down(128, 256))
        self.down213a = (Down(64, 128))
        self.down213b = (Down(128, 256))
        self.dconv13 = (DoubleConv(192, 64))
        self.dconv23 = (DoubleConv(256, 128))
        self.dconv33 = (DoubleConv(512, 256))
        self.up321 = (Up(2))
        self.up331 = (Up(4))
        self.up332 = (Up(2))
        self.down312 = (Down(64, 128))
        self.down313a = (Down(64, 128))
        self.down313b = (Down(128, 256))
        self.down314a = (Down(64, 128))
        self.down314b = (Down(128, 256))
        self.down314c = (Down(256, 512))
        self.down323 = (Down(128, 256))
        self.down324a = (Down(128, 256))
        self.down324b = (Down(256, 512))
        self.down334 = (Down(256, 512))
        self.dconv14 = (DoubleConv(448, 64))
        self.dconv24 = (DoubleConv(512, 128))
        self.dconv34 = (DoubleConv(768, 256))
        self.dconv44 = (DoubleConv(1536, 512))
        self.up421 = (Up(2))
        self.up431 = (Up(4))
        self.up441 = (Up(8))
        self.up432 = (Up(2))
        self.up442 = (Up(4))
        self.up443 = (Up(2))
        self.down412 = (Down(64, 128))
        self.down413a = (Down(64, 128))
        self.down413b = (Down(128, 256))
        self.down414a = (Down(64, 128))
        self.down414b = (Down(128, 256))
        self.down414c = (Down(256, 512))
        self.down423 = (Down(128, 256))
        self.down424a = (Down(128, 256))
        self.down424b = (Down(256, 512))
        self.down434 = (Down(256, 512))
        self.dconv15 = (DoubleConv(960, 64))
        self.dconv25 = (DoubleConv(1024, 128))
        self.dconv35 = (DoubleConv(1280, 256))
        self.dconv45 = (DoubleConv(2048, 512))
        self.up521 = (Up(2))
        self.up531 = (Up(4))
        self.up541 = (Up(8))

        self.convlast = (nn.Conv2d(960, 64, kernel_size=3, padding=1))
        self.outc = (OutConv(64, n_classes))

    def forward(self, x):
        d11 = self.dconv11(x)
        d12 = self.dconv12(d11)
        d22 = self.down112(d12)
        up221 = self.up221(d22)
        d13 = torch.cat([up221, d12], dim=1)
        d13 = self.dconv13(d13)
        down212 = self.down212(d12)
        d23 = torch.cat([d22, down212], dim=1)
        d23 = self.dconv23(d23)
        down223 = self.down223(d22)
        down213a = self.down213a(d12)
        down213 = self.down213b(down213a)
        d33 = torch.cat([down213, down223], dim=1)
        d33 = self.dconv33(d33)
        up331 = self.up331(d33)
        up321 = self.up321(d23)
        up332 = self.up332(d33)
        d14 = torch.cat([up321, up331, d13], dim=1)
        d14 = self.dconv14(d14)
        down312 = self.down312(d13)
        down313a = self.down313a(d13)
        down313 = self.down313b(down313a)
        down314a = self.down314a(d13)
        down314b = self.down314b(down314a)
        down314 = self.down314c(down314b)
        d24 = torch.cat([d23, down312, up332], dim=1)
        d24 = self.dconv24(d24)
        down323 = self.down323(d23)
        down324a = self.down324a(d23)
        down324 = self.down324b(down324a)
        d34 = torch.cat([d33, down313, down323], dim=1)
        d34 = self.dconv34(d34)
        down334 = self.down334(d33)
        d44 = torch.cat([down314, down324, down334], dim=1)
        d44 = self.dconv44(d44)
        up421 = self.up421(d24)
        up431 = self.up431(d34)
        up441 = self.up441(d44)
        up432 = self.up432(d34)
        up442 = self.up442(d44)
        up443 = self.up443(d44)
        d15 = torch.cat([up421, up431, up441, d14], dim=1)
        d15 = self.dconv15(d15)
        down412 = self.down412(d14)
        down413a = self.down413a(d14)
        down413 = self.down413b(down413a)
        down414a = self.down414a(d14)
        down414b = self.down414b(down414a)
        down414 = self.down414c(down414b)
        d25 = torch.cat([down412, up432, up442, d22], dim=1)
        d25 = self.dconv25(d25)
        down423 = self.down423(d24)
        down424a = self.down424a(d24)
        down424 = self.down424b(down424a)
        d35 = torch.cat([down413, down423, up443, d34], dim=1)
        d35 = self.dconv35(d35)
        down434 = self.down434(d34)
        d45 = torch.cat([down414, down424, down434, d44], dim=1)
        d45 = self.dconv45(d45)
        up521 = self.up521(d25)
        up531 = self.up531(d35)
        up541 = self.up541(d45)
        x = torch.cat([up521, up531, up541, d15], dim=1)
        x = self.convlast(x)
        logits = self.outc(x)

        return logits
        

class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then conv"""

    def __init__(self, scale_factor, bilinear=True):
        super().__init__()
        self.up = nn.Upsample(scale_factor=scale_factor, mode='bilinear', align_corners=True)
        
    def forward(self, x1):
        return self.up(x1)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)