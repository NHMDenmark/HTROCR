import torch
from torch import nn

class FusedInvertedBottleneck(nn.Module):
    def __init__(self, in_channels, out_channels, expansion_factor):
        '''
        Parameters
        ----------
        in_channels : integer
            Input channels to the FIB block.
        out_channels : integer
            Out channels after fused inverted bottleneck layers.
        expansion_factor : integer
            Expansion factor for depthwise separable convolution in fused inverted bottleneck layers.
        '''
        super().__init__()
        self.depthwise_conv = nn.Conv2d(in_channels, in_channels * expansion_factor, kernel_size=3, padding=1, groups=in_channels)
        self.pointwise_conv = nn.Conv2d(in_channels * expansion_factor, out_channels, kernel_size=1)
        self.bn1 = nn.BatchNorm2d(in_channels * expansion_factor)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        residual = x
        out = self.depthwise_conv(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.pointwise_conv(out)
        out = self.bn2(out)
        out += residual
        out = self.relu(out)
        return out
    
class ReduceBlock(nn.Module):
    def __init__(self, in_channels, f1, f2, f3, f4):
        '''
        Parameters
        ----------
        in_channels : integer
            Input channels to the reduce block.
        f1 : integer
            Height reduction convolution with 'valid' padding kernel size
        f2 : integer
            Width feature extraction convolution with 'same' padding kernel size
        f3 : integer
            Height feature extraction convolution with 'same' padding kernel size
        '''
        super().__init__()
        self.conv0 = nn.Conv2d(in_channels, in_channels, kernel_size=(1,1), stride=(1,1), padding='same')
        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size=(f1,1), stride=(1,1), padding='valid')
        self.conv2 = nn.Conv2d(in_channels, in_channels, kernel_size=(1,f2), stride=(1,1), padding='same')
        self.conv3 = nn.Conv2d(in_channels, in_channels*2, kernel_size=(f3,1), stride=(1,1), padding='same')
        self.residual = nn.Conv2d(in_channels, in_channels*2, kernel_size=(f4,1), stride=(1,1), padding='valid')
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.bn2 = nn.BatchNorm2d(in_channels * 2)
        self.relu = nn.ReLU6(inplace=True)

    def forward(self, x):
        out = self.conv0(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv1(out)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv3(out)
        out = self.bn2(out)
        out = self.relu(out)
        residual = self.residual(x)
        residual = self.bn2(residual)
        residual = self.relu(residual)
        # residual = torch.mean(residual, dim=2, keepdim=True)
        out = torch.cat((out,residual), dim=1)
        return out

class Backbone(nn.Module):
    '''
    CNN backbone with Fused Inverted Bottleneck layers
    adjusted according to:
    'Rethinking Text Line Recognition Models' by Diaz et al. 2021
    '''
    def __init__(self, in_channels, out_channels, expansion_factor):
        '''
        Parameters
        ----------
        in_channels : integer
            Image color channels
        out_channels : integer
            Out channels after fused inverted bottleneck layers.
        expansion_factor : integer
            Expansion factor for depthwise separable convolution in fused inverted bottleneck layers.
        '''
        super().__init__()
        self.space_to_depth = nn.PixelUnshuffle(4)
        self.depthwise_conv = nn.Conv2d(16* in_channels, 16 * in_channels * expansion_factor, kernel_size=3, padding=1,  groups=16*in_channels)
        self.bn1 = nn.BatchNorm2d(16 * in_channels * expansion_factor)
        self.pointwise_conv = nn.Conv2d(16 * expansion_factor, out_channels, kernel_size=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.fib_layers = nn.ModuleList([FusedInvertedBottleneck(out_channels, out_channels, expansion_factor) for _ in range(10)])
#        self.reduce_block = ReduceBlock(out_channels, 10, 3, 6, 10)
        self.reduce_block = ReduceBlock(out_channels, 8, 3, 6, 8)

    def forward(self, x):
        out = self.space_to_depth(x)
        out = self.depthwise_conv(out)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.pointwise_conv(out)
        out = self.bn2(out)
        out = self.relu(out)
        for layer in self.fib_layers:
            out = layer(out)
        out = self.reduce_block(out)
        out = out.squeeze(2)
        out = out.permute(0, 2, 1)
        return out