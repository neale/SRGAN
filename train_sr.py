import os
import sys
import time
import argparse
import numpy as np
from scipy.misc import imshow

import torch
import torchvision
import torchvision.transforms as transforms
import  torchvision.models as models
from torch import nn
from torch import autograd
from torch import optim
from torch.nn import functional as F

import ops
import plot
import utils
import encoders
import generators
import discriminators
from vgg import vgg19, vgg19_bn, VGGextraction

#TODO
""" 
Still need to get vgg19 perceptural loss working, 
Have vgg19_bn, and vgg19, but I need to grab the features from the last layer
vgg - conv5/4
"""
def load_args():

    parser = argparse.ArgumentParser(description='recover-gan')
    parser.add_argument('-d', '--downsample', default=4, type=int, help='')
    parser.add_argument('-l', '--gp', default=10, type=int, help='grad penalty')
    parser.add_argument('-b', '--batch_size', default=50, type=int)
    parser.add_argument('-e', '--epochs', default=200000, type=int)
    parser.add_argument('-o', '--output_dim', default=784, type=int)
    parser.add_argument('-t', '--task', default='AE', type=str)
    parser.add_argument('--dim', default=64)
    parser.add_argument('--dataset', default='mnist')
    parser.add_argument('--batchnorm', default=True)
    parser.add_argument('--resume', default=None)
    args = parser.parse_args()
    return args


def load_models(args):
    if args.task == 'AE':
        if args.dataset == 'mnist':
            netG = generators.MNISTgenerator(args).cuda()
            netD = discriminators.MNISTdiscriminator(args).cuda()
            netE = encoders.MNISTencoder(args).cuda()

        elif args.dataset == 'cifar10':
            netG = generators.CIFARgenerator(args).cuda()
            netD = discriminators.CIFARdiscriminator(args).cuda()
            netE = encoders.CIFARencoder(args).cuda()

    if args.task == 'sr':
        if args.dataset == 'cifar10':
            netG = generators.genResNet(args).cuda()
            netD = discriminators.SRdiscriminatorCIFAR(args).cuda()
            vgg = vgg19_bn(pretrained=True).cuda()
            netE = VGGextraction(vgg)

        elif args.dataset == 'imagenet':
            downsize = 224 // args.downsample
            netG = generators.SRResNet(args, (3, 96, 96)).cuda(0)
            netD = discriminators.SRdiscriminator(args).cuda(0)
            netD = None
            vgg = vgg19_bn(pretrained=True).cuda(1)
            netE = VGGextraction(vgg).cuda(1)
        
        elif args.dataset == 'raise':
            netG = generators.SRResNet(args, (3, 96, 96)).cuda(0)
            netD = discriminators.SRdiscriminator(args).cuda(0)
            netD = None
            vgg = vgg19_bn(pretrained=True).cuda(1)
            netE = VGGextraction(vgg).cuda(1)

    print (netG, netD, netE)
    return (netG, netD, netE)


def train():
    args = load_args()
    train_gen, dev_gen, test_gen = utils.dataset_iterator(args)
    torch.manual_seed(1)
    netG, netD, netE = load_models(args)

    # optimizerD = optim.Adam(netD.parameters(), lr=1e-4, betas=(0.5, 0.9))
    optimizerG = optim.Adam(netG.parameters(), lr=0.0001, betas=(0.5, 0.9))
    vgg_scale = 0.0784 # 1/12.75
    mse_criterion = nn.MSELoss()
    one = torch.FloatTensor([1]).cuda(0)
    mone = (one * -1).cuda(0)

    gen = utils.inf_train_gen(train_gen)

    """ attempt to resume generator from checkpoint """
    if args.resume is not None:
        print ("loading ", args.resume)
        state = torch.load(args.resume)
        netG.load_state_dict(state)

    """ train SRResNet with MSE only """
    for iteration in range(1, 200001):
        start_time = time.time()
        
        _data_hr = next(gen)
        real_data_lr, real_data_hr = utils.scale_data(args, _data_hr)
        real_data_hr_v = autograd.Variable(real_data_hr)
        real_data_lr_v = autograd.Variable(real_data_lr)
        
        fake_hr = netG(real_data_lr_v)

        netG.zero_grad()
        content_loss = mse_criterion(fake_hr, real_data_hr_v)
        psnr = ops.psnr(args, content_loss)
        content_loss.backward()
        optimizerG.step()

        save_dir = './plots/'+args.dataset
        plot.plot(save_dir, '/content_loss_(mse)', content_loss.data.cpu().numpy())
        plot.plot(save_dir, '/psnr', np.array(psnr))
        data = (real_data_lr, real_data_hr, fake_hr)
        if iteration % 20 == 19:
            utils.generate_sr_image(iteration, netG, save_dir, args, data)
        if (iteration < 5) or (iteration % 20 == 19):
            plot.flush()
        plot.tick()
        if iteration % 5000 == 0:
            torch.save(netG.state_dict(), './SRResNet_PL.pt')

    """ train SRGAN """
    for iteration in range(args.epochs):
        start_time = time.time()
        """ Update AutoEncoder """

        for p in netD.parameters():
            p.requires_grad = False
        netG.zero_grad()
        netE.zero_grad()
        _data = next(gen)
        real_data = utils.stack_data(args, _data)
        real_data_v = autograd.Variable(real_data)
        encoding = netE(real_data_v)
        fake = netG(encoding)
        ae_loss = ae_criterion(fake, real_data_v)
        ae_loss.backward(one)
        optimizerE.step()
        optimizerG.step()

        """ Update D network """

        for p in netD.parameters():  # reset requires_grad
            p.requires_grad = True  # they are set to False below in netG update
        for i in range(5):
            _data = next(gen)
            real_data = utils.stack_data(args, _data)
            real_data_v = autograd.Variable(real_data)
            # train with real data
            netD.zero_grad()
            D_real = netD(real_data_v)
            D_real = D_real.mean()
            D_real.backward(mone)
            # train with fake data
            noise = torch.randn(args.batch_size, args.dim).cuda()
            noisev = autograd.Variable(noise, volatile=True)  # totally freeze netG
            # instead of noise, use image
            fake = autograd.Variable(netG(real_data_v).data)
            inputv = fake
            D_fake = netD(inputv)
            D_fake = D_fake.mean()
            D_fake.backward(one)

            # train with gradient penalty 
            gradient_penalty = ops.calc_gradient_penalty(args,
                    netD, real_data_v.data, fake.data)
            gradient_penalty.backward()

            D_cost = D_fake - D_real + gradient_penalty
            Wasserstein_D = D_real - D_fake
            optimizerD.step()

        # Update generator network (GAN)
        # noise = torch.randn(args.batch_size, args.dim).cuda()
        # noisev = autograd.Variable(noise)
        _data = next(gen)
        real_data = utils.stack_data(args, _data)
        real_data_v = autograd.Variable(real_data)
        # again use real data instead of noise
        fake = netG(real_data_v)
        #Perceptual loss
        vgg_data_v = autograd.Variable(vgg_data)
        vgg_features_real = netE(vgg_data_v)
        fake = netG(real_data_v)
        vgg_features_fake = netE(fake)
        diff = vgg_features_fake - vgg_features_real.cuda(0)
        perceptual_loss = vgg_scale * ((diff.pow(2)).sum(3).mean())  # mean(sum(square(diff)))
        perceptual_loss.backward(one)
        G = netD(fake)
        G = G.mean()
        G.backward(mone)
        G_cost = -G
        optimizerG.step()

        # Write logs and save samples 

        save_dir = './plots/'+args.dataset
        plot.plot(save_dir, '/disc cost', np.round(D_cost.cpu().data.numpy(), 4))
        plot.plot(save_dir, '/gen cost', np.round(G_cost.cpu().data.numpy(), 4))
        plot.plot(save_dir, '/w1 distance', np.round(Wasserstein_D.cpu().data.numpy(), 4))
        # plot.plot(save_dir, '/ae cost', np.round(ae_loss.data.cpu().numpy(), 4))

        # Calculate dev loss and generate samples every 100 iters
        if iteration % 100 == 99:
            dev_disc_costs = []
            for images, _ in dev_gen():
                imgs = utils.stack_data(args, images) 
                imgs_v = autograd.Variable(imgs, volatile=True)
                D = netD(imgs_v)
                _dev_disc_cost = -D.mean().cpu().data.numpy()
                dev_disc_costs.append(_dev_disc_cost)
            plot.plot(save_dir ,'/dev disc cost', np.round(np.mean(dev_disc_costs), 4))

            # utils.generate_image(iteration, netG, save_dir, args)
            # utils.generate_ae_image(iteration, netE, netG, save_dir, args, real_data_v)
            utils.generate_sr_image(iteration, netG, save_dir, args, real_data_v)
        # Save logs every 100 iters 
        if (iteration < 5) or (iteration % 100 == 99):
            plot.flush()
        plot.tick()

if __name__ == '__main__':
    train()
