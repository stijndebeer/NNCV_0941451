import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self, in_channels=3, n_classes=19):
        
        super(Model, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)           
        self.dconv11 = Tripleres(64, 64)
        self.dconv12 = Doubleres(64, 64)
        self.down112 = Down(64, 128)
        self.dconv22 = Doubleres(128, 128)
        self.up221 = Up(2)
        self.down212 = Down(64, 128)
        self.down223 = Down(128, 256)
        self.down213a = Down(64, 128)
        self.down213b = Down(128, 256)
        self.dconv13 = Doubleres(192, 64)
        self.dconv23 = Doubleres(256, 128)
        self.dconv33 = Doubleres(512, 256)
        self.up321 = Up(2)
        self.up331 = Up(4)
        self.up332 = Up(2)
        self.down312 = Down(64, 128)
        self.down313a = Down(64, 128)
        self.down313b = Down(128, 256)
        self.down314a = Down(64, 128)
        self.down314b = Down(128, 256)
        self.down314c = Down(256, 512)
        self.down323 = Down(128, 256)
        self.down324a = Down(128, 256)
        self.down324b = Down(256, 512)
        self.down334 = Down(256, 512)
        self.dconv14 = Doubleres(448, 64)
        self.dconv24 = Doubleres(512, 128)
        self.dconv34 = Doubleres(768, 256)
        self.dconv44 = Doubleres(1536, 512)
        self.up421 = Up(2)
        self.up431 = Up(4)
        self.up441 = Up(8)
        self.outc = OutConv(960, n_classes)
        # self.ocr = OCRBlock(in_channels=960, mid_channels=512, out_channels=n_classes, num_classes=n_classes)
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        d11 = self.dconv11(x)
        d12 = self.dconv12(d11)
        d112 = self.down112(d12)
        d22 = self.dconv22(d112)
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
        x = torch.cat([up421, up431, up441, d14], dim=1)
        # logits, aux_logits = self.ocr(x)
        logits = self.outc(x)
        return logits#, aux_logits
        

class Doubleres(nn.Module): #basically a resnet block
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, downsample=None):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels)
        )

    def forward(self, x):
        residual = x
        if self.downsample is not None:
            residual = self.downsample(x)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = x + residual
        x = self.relu(x)
        return x
    
class Tripleres(nn.Module): #basically a bottleneck resnet block
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, downsample=None):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0, bias=False) #kernel1 enorm_dice_final
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv3 = nn.Conv2d(out_channels, out_channels, kernel_size=1, padding=0, bias=False) #kernel1 enorm_dice_final
        self.bn3 = nn.BatchNorm2d(out_channels)
        self.downsample = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels)
        )

    def forward(self, x):
        residual = x
        if self.downsample is not None:
            residual = self.downsample(x)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.conv3(x)
        x = self.bn3(x)
        x = x + residual
        x = self.relu(x)
        return x 

class Down(nn.Module):
    """Downscaling with maxpool then conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1, bias=False)
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
        self.merge_conv = nn.Sequential(
        nn.Conv2d(in_channels, in_channels, kernel_size=1),
        nn.BatchNorm2d(in_channels),
        nn.ReLU(inplace=True),
        nn.Conv2d(in_channels, out_channels, kernel_size=1),
        )

    def forward(self, x):
        return self.merge_conv(x)

class OCRBlock(nn.Module):
    def __init__(self, in_channels, mid_channels, out_channels, num_classes):
        super(OCRBlock, self).__init__()
        
        self.contextual_rep = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True)
        )
        
        self.object_context_block = nn.Sequential(
            nn.Conv2d(mid_channels, mid_channels, kernel_size=1, bias=False),
        #    nn.BatchNorm2d(mid_channels), #it crashes because of this? wtf?
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, mid_channels, kernel_size=1, bias=False)
        )
        
        self.cls_head = nn.Sequential(
            nn.Conv2d(mid_channels, out_channels, kernel_size=1)
        )
        
        self.aux_head = nn.Sequential(
            nn.Conv2d(in_channels, num_classes, kernel_size=1)
        )

    def forward(self, x):
        context = self.contextual_rep(x)
        object_context = self.object_context_block(context)
        output = self.cls_head(object_context)        
        aux_out = self.aux_head(x)
        
        return output, aux_out