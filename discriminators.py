from torch import nn


class SRdiscriminator(nn.Module):
    def __init__(self, args):
        super(SRdiscriminator, self).__init__()
        self.BN = args.batchnorm
        self.name = 'srD'
        self.shape = (3, 32, 32)
        self.dim = args.dim
        
        conv_init = nn.Sequential(nn.Conv2d(3, 64, 3, 1), nn.LeakyReLU())
        conv2 = nn.Sequential(nn.Conv2d(64, 64, 3, 2), nn.LeakyReLU())
        conv3 = nn.Sequential(nn.Conv2d(64, 128, 3, 1), nn.LeakyReLU())
        conv4 = nn.Sequential(nn.Conv2d(128, 128, 3, 2), nn.LeakyReLU())
        conv5 = nn.Sequential(nn.Conv2d(128, 256, 3, 1), nn.LeakyReLU())
        conv6 = nn.Sequential(nn.Conv2d(256, 256, 3, 2), nn.LeakyReLU())
        conv7 = nn.Sequential(nn.Conv2d(256, 512, 3, 1), nn.LeakyReLU())
        conv8 = nn.Sequential(nn.Conv2d(512, 512, 3, 2), nn.LeakyReLU())

        out_block = nn.Sequential(
                nn.Linear(512 * 6 * 6, 1024),
                nn.LeakyReLU(),
                nn.Linear(1024, 1),
                nn.Sigmoid()
                )

        self.conv_init = conv_init
        self.conv2 = conv2
        self.conv3 = conv3
        self.conv4 = conv4
        self.conv5 = conv5
        self.conv6 = conv6
        self.conv7 = conv7
        self.conv8 = conv8
        self.out_block = out_block

    def forward(self, input):
        block1 = self.conv_init(input)
        block2 = self.conv2(block1)
        block3 = self.conv3(block2)
        block4 = self.conv4(block3)
        block5 = self.conv5(block4)
        block6 = self.conv6(block5)
        block7 = self.conv7(block6)
        block8 = self.conv8(block7)
        print (block8.size())
        out = block8.view(block7.size(0), -1)
        print (out.size())
        out = self.out_block(block8)
        return out
