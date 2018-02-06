from torch import nn

class CIFARdiscriminator(nn.Module):
    def __init__(self, args):
        super(CIFARdiscriminator, self).__init__()
        self._name = 'cifarD'
        self.shape = (32, 32, 3)
        self.dim = args.dim
        convblock = nn.Sequential(
                nn.Conv2d(3, self.dim, 3, 2, padding=1),
                nn.Dropout(p=0.3),
                nn.LeakyReLU(),
                nn.Conv2d(self.dim, 2 * self.dim, 3, 2, padding=1),
                nn.Dropout(p=0.3),
                nn.LeakyReLU(),
                nn.Conv2d(2 * self.dim, 4 * self.dim, 3, 2, padding=1),
                nn.Dropout(p=0.3),
                nn.LeakyReLU()
                )
        self.main = convblock
        self.linear = nn.Linear(4*4*4*self.dim, 1)

    def forward(self, input):
        output = self.main(input)
        print ("cifar out shape: ", output.size())
        output = output.view(-1, 4*4*4*self.dim)
        output = self.linear(output)
        return output


class MNISTdiscriminator(nn.Module):
    def __init__(self, args):
        super(MNISTdiscriminator, self).__init__()
        self._name = 'mnistD'
        self.shape = (1, 28, 28)
        self.dim = args.dim
        convblock = nn.Sequential(
                nn.Conv2d(1, self.dim, 5, stride=2, padding=2),
                nn.Dropout(p=0.3),
                nn.ReLU(True),
                nn.Conv2d(self.dim, 2*self.dim, 5, stride=2, padding=2),
                nn.Dropout(p=0.3),
                nn.ReLU(True),
                nn.Conv2d(2*self.dim, 4*self.dim, 5, stride=2, padding=2),
                nn.Dropout(p=0.3),
                nn.ReLU(True)
                )
        self.main = convblock
        self.output = nn.Linear(4*4*4*self.dim, 1)

    def forward(self, input):
        input = input.view(-1, 1, 28, 28)
        out = self.main(input)
        out = out.view(-1, 4*4*4*self.dim)
        out = self.output(out)
        return out.view(-1)


class SRdiscriminator(nn.Module):
    def __init__(self, args):
        super(SRdiscriminatorCIFAR, self).__init__()
        self.BN = args.batchnorm
        self.batch_size = args.batch_size
        self.name = 'SRcifarD'
        self.shape = (32, 32, 3)
        self.dim = args.dim

        conv_init = nn.Sequential(nn.Conv2d(3, 32, 3, 1), nn.LeakyReLU())
        conv2 = nn.Sequential(nn.Conv2d(32, 32, 3, 2), nn.LeakyReLU())
        conv3 = nn.Sequential(nn.Conv2d(32, 64, 3, 1), nn.LeakyReLU())
        conv4 = nn.Sequential(nn.Conv2d(64, 64, 3, 2), nn.LeakyReLU())
        conv5 = nn.Sequential(nn.Conv2d(64, 128, 3, 1), nn.LeakyReLU())
        conv6 = nn.Sequential(nn.Conv2d(128, 128, 3, 2), nn.LeakyReLU())

        out_block = nn.Sequential(
                nn.Linear(1152, 1024),
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
        self.out_block = out_block

    def forward(self, input):
        #print ("input size: ", input.size())
        block1 = self.conv_init(input)
        #print ('block1: ', block1.size())
        block2 = self.conv2(block1)
        #print ('block2: ', block2.size())
        block3 = self.conv3(block2)
        #print ('block3: ', block3.size())
        block4 = self.conv4(block3)
        #print ('block4: ', block4.size())
        block5 = self.conv5(block4)
        #print ('block5: ', block5.size())
        out = block5.view(self.batch_size, -1)
        out = self.out_block(out)
        return out


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
