from torch import nn


class SRdiscriminator(nn.Module):
    def __init__(self, args, shape):
        super(SRdiscriminator, self).__init__()
        self.name = 'srganD'
        self.shape = shape

        conv_init = nn.Sequential(nn.Conv2d(3, 64, 3, 1), nn.LeakyReLU())
        conv2 = nn.Sequential(nn.Conv2d(64, 64, 3, 2), 
                nn.LeakyReLU(),
                nn.BatchNorm2d(64))
        conv3 = nn.Sequential(nn.Conv2d(64, 128, 3, 1),
                nn.LeakyReLU(),
                nn.BatchNorm2d(128))
        conv4 = nn.Sequential(nn.Conv2d(128, 128, 3, 2),
                nn.LeakyReLU(),
                nn.BatchNorm2d(128))
        conv5 = nn.Sequential(nn.Conv2d(128, 256, 3, 1),
                nn.LeakyReLU(),
                nn.BatchNorm2d(256))
        conv6 = nn.Sequential(nn.Conv2d(256, 256, 3, 2),
                nn.LeakyReLU(),
                nn.BatchNorm2d(256))
        conv7 = nn.Sequential(nn.Conv2d(256, 512, 3, 1),
                nn.LeakyReLU(),
                nn.BatchNorm2d(512))
        conv8 = nn.Sequential(nn.Conv2d(512, 512, 3, 2),
                nn.LeakyReLU(),
                nn.BatchNorm2d(512))

        out_block = nn.Sequential(
                nn.Linear(512 * 3 * 3, 1024),
                nn.LeakyReLU(),
                nn.Linear(1024, 1),
                nn.Sigmoid())

        self.conv_init = conv_init
        self.conv2 = conv2
        self.conv3 = conv3
        self.conv4 = conv4
        self.conv5 = conv5
        self.conv6 = conv6
        self.conv7 = conv7
        self.conv8 = conv8
        self.out_block = out_block

    def forward(self, x):
        x = self.conv_init(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)
        x = self.conv7(x)
        x = self.conv8(x)
        out = x.view(x.size()[0], -1)
        out = self.out_block(out)
        return out
