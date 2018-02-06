import numpy as np
from torch import nn
from torch.nn import functional as F


class SRResNet(nn.Module):
    def __init__(self, args, shape):
        super(SRResNet, self).__init__()
        BN = args.batchnorm
        self._name = 'SRResNet'
        self.dim = args.dim
        self.n_resblocks = 16
        self.shape = shape
        self.factor = args.downsample

        convblock_init = nn.Sequential(
                nn.Conv2d(3, 64, 9, stride=1, padding=4),
                nn.PReLU()
                )

        convbn = nn.Sequential(
                nn.Conv2d(64, 64, 3, stride=1, padding=1),
                nn.BatchNorm2d(64)
                )
               
        upsample_block = nn.Sequential(
                nn.Conv2d(64, 256, 3, stride=1, padding=1),
                nn.PixelShuffle(2),
                nn.PReLU(),
                nn.Conv2d(64, 256, 3, stride=1, padding=1),
                nn.PixelShuffle(2),
                nn.PReLU(),
                nn.Conv2d(64, 3, 9, stride=1, padding=4)
                )

        upsample = nn.Sequential(
                nn.Conv2d(64, 256, 3, stride=1, padding=1),
                nn.PixelShuffle(2),
                nn.PReLU(),
                )


        resblock = nn.Sequential(
                nn.Conv2d(64, 64, 3, stride=1, padding=1),
                nn.BatchNorm2d(64),
                nn.PReLU(),
                nn.Conv2d(64, 64, 3, stride=1, padding=1),
                nn.BatchNorm2d(64)
                )
        
        self.convbn = convbn
        self.convblock = convblock_init
        self.upsample = upsample
        self.upsample_block = upsample_block
        self.resblock = resblock

    def forward(self, x):
        x = self.convblock(x)
        xr = x.clone()

        for _ in range(self.n_resblocks):
            xr = self.resblock(xr) + xr
        
        x = self.convbn(xr) + x
        
        output = self.upsample_block(x)
        
        out_dim = self.factor * self.shape[1]
        output = output.view(-1, 3, 96, 96)
        
        return output
