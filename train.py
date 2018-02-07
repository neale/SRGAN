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
    parser.add_argument('-b', '--batch_size', default=4, type=int)
    parser.add_argument('-e', '--epochs', default=200000, type=int)
    parser.add_argument('-t', '--task', default='SRGAN', type=str)
    parser.add_argument('-c', '--crop_size', default=96, type=int)
    parser.add_argument('--dim', default=64)
    parser.add_argument('--dataset', default='mnist')
    parser.add_argument('--batchnorm', default=True)
    parser.add_argument('--gen_resume', default=None)
    args = parser.parse_args()
    return args


def load_models(args):

    crop = args.crop_size
    downsize = crop // args.downsample
    shape = (3, crop, crop)
    if args.dataset == 'cifar10':
        netG = generators.genResNet(args).cuda()
        netD = discriminators.SRdiscriminatorCIFAR(args).cuda()
        vgg = vgg19_bn(pretrained=True).cuda()
        netL = VGGextraction(vgg)

    elif args.dataset == 'imagenet':
        netG = generators.SRResNet(args, shape).cuda(0)
        netD = discriminators.SRdiscriminator(args, shape).cuda(0)
        netD = None
        vgg = vgg19_bn(pretrained=True).cuda(1)
        netL = VGGextraction(vgg).cuda(1)
    
    elif args.dataset == 'raise':
        netG = generators.SRResNet(args, shape).cuda(0)
        netD = discriminators.SRdiscriminator(args, shape).cuda(0)
        vgg = vgg19_bn(pretrained=True).cuda(1)
        netL = VGGextraction(vgg).cuda(1)

    print (netG, netD, netL)
    return (netG, netD, netL)


def train():
    args = load_args()
    train_gen, dev_gen, test_gen = utils.dataset_iterator(args)
    torch.manual_seed(1)
    netG, netD, netL = load_models(args)

    optimizerG = optim.Adam(netG.parameters(), lr=0.0001, betas=(0.5, 0.9))
    optimizerD = optim.Adam(netD.parameters(), lr=1e-4, betas=(0.5, 0.9))
    vgg_scale = 0.006  # scales perceptual loss to be on order of MSE loss
    loss_ratio = 0.001 # balancing ratio on content vs GAN
    mse_criterion = nn.MSELoss()
    one = torch.FloatTensor([1]).cuda(0)
    mone = (one * -1).cuda(0)

    gen = utils.inf_train_gen(train_gen)

    """ Attempt to resume generator from checkpoint """
    if args.gen_resume is not None:
        print ("loading ", args.gen_resume)
        state = torch.load(args.gen_resume)
        netG.load_state_dict(state)

    """ Pretrain SRResNet with MSE only """
    if args.task == 'SRResNet':
        for iteration in range(1, args.epochs):
            start_time = time.time()
            
            _data_hr = next(gen)
            real_data_lr, real_data_hr = utils.scale_data(args, _data_hr)
            real_data_lr = real_data_lr.cuda(0)
            real_data_hr = real_data_hr.cuda(0)
            real_data_hr_v = autograd.Variable(real_data_hr)
            real_data_lr_v = autograd.Variable(real_data_lr)
            
            fake_hr = netG(real_data_lr_v)

            netG.zero_grad()
            content_loss = mse_criterion(fake_hr, real_data_hr_v)
            psnr = ops.psnr(args, content_loss)
            content_loss.backward()
            optimizerG.step()

            save_dir = './plots/'+args.dataset
            plot.plot('content_loss_(mse)', content_loss.data.cpu().numpy())
            plot.plot('psnr', np.array(psnr))
            data = (real_data_lr, real_data_hr, fake_hr)
            if iteration % 20 == 19:
                utils.generate_sr_image(iteration, netG, save_dir, args, data)
            if (iteration < 5) or (iteration % 20 == 19):
                plot.flush()
            plot.tick()
            if iteration % 5000 == 0:
                torch.save(netG.state_dict(), './SRResNet_PL.pt')

    elif args.task == 'SRGAN':
        """ reduce generator learning rate """
        for param in optimizerG.param_groups:
            param['lr'] *= 0.10

        """ train SRGAN """
        for iteration in range(1, args.epochs):
            start_time = time.time()
            """ Update D network """
            for p in netD.parameters():  # reset requires_grad
                p.requires_grad = True  # they are set to False below in netG update
            
            for i in range(1):
                _data_hr = next(gen)
                real_data_lr, real_data_hr = utils.scale_data(args, _data_hr)
                real_data_lr = real_data_lr.cuda(0)
                real_data_hr = real_data_hr.cuda(0)
                real_data_hr_v = autograd.Variable(real_data_hr)
                real_data_lr_v = autograd.Variable(real_data_lr)
                fake_hr = netG(real_data_lr_v)
                """ trying DCGAN first """
                netD.zero_grad()
                D_real = netD(real_data_hr_v)
                D_fake = netD(fake_hr)
                D_fake_loss = torch.log(1. - D_fake + 1e-6)
                D_real_loss = torch.log(D_real + 1e-6)
                D_cost = (-(D_fake_loss + D_real_loss)).mean()
                D_cost.backward(retain_graph=True)
                """
                D_real = D_real.mean()
                D_real.backward(mone)
                D_fake = D_fake.mean()
                D_fake.backward(one)
                # train with gradient penalty 
                gradient_penalty = ops.calc_gradient_penalty(args,
                        netD, real_data_hr_v.data, fake_hr.data)
                gradient_penalty.backward()
                D_cost = D_fake - D_real + gradient_penalty
                Wasserstein_D = D_real - D_fake
                """
                optimizerD.step()

            """ Train Generator """
            for p in netD.parameters():  # reset requires_grad
                p.requires_grad = False  # they are set to False below in netG update
            netG.zero_grad()
            """
            _data_hr = next(gen)
            real_data_lr, real_data_hr = utils.scale_data(args, _data_hr)
            real_data_lr = real_data_lr.cuda(0)
            real_data_hr = real_data_hr.cuda(0)
            real_data_hr_v = autograd.Variable(real_data_hr)
            real_data_lr_v = autograd.Variable(real_data_lr)
            fake_hr = netG(real_data_lr_v)
            """
            vgg_real = netL(real_data_hr_v)
            vgg_fake = netL(fake_hr)
            p1_loss = mse_criterion(fake_hr, real_data_hr_v)
            p2_loss = vgg_scale * mse_criterion(vgg_fake, vgg_real)
            perceptual_loss =  p1_loss + p2_loss
            # perceptual_loss.backward(one)
            """ Try DCGAN first """
            adv_loss = (-torch.log(netD(fake_hr) + 1e-6)).mean()
            #G = netD(fake_hr)
            #G = G.mean()
            # G.backward(mone)
            #adv_loss = -G
            #perceptual_loss = vgg_scale * ((diff.pow(2)).sum(3).mean())  
            total_loss = perceptual_loss + (loss_ratio * adv_loss)
            total_loss.backward()
            optimizerG.step()

            psnr = ops.psnr(args, perceptual_loss)

            save_dir = './plots/'+args.dataset
            plot.plot('vgg loss', perceptual_loss.cpu().data.numpy())
            plot.plot('psnr', np.array(psnr))
            plot.plot('disc cost', D_cost.cpu().data.numpy())
            plot.plot('gen cost', adv_loss.cpu().data.numpy())
            #plot.plot('w1 distance', Wasserstein_D.cpu().data.numpy())

            # Calculate dev loss and generate samples every 100 iters
            if iteration % 20 == 19:
                dev_disc_costs = []
                for images, _ in dev_gen():
                    _, imgs = utils.scale_data(args, images) 
                    imgs = imgs.cuda(0)
                    imgs_v = autograd.Variable(imgs, volatile=True)
                    D = netD(imgs_v)
                    _dev_disc_cost = -D.mean().cpu().data.numpy()
                    dev_disc_costs.append(_dev_disc_cost)
                plot.plot('dev disc cost', np.mean(dev_disc_costs))
                data = (real_data_lr, real_data_hr, fake_hr)
                utils.generate_sr_image(iteration, netG, save_dir, args, data)
            # Save logs every 100 iters 
            if (iteration < 5) or (iteration % 20 == 19):
                plot.flush()
            plot.tick()

if __name__ == '__main__':
    train()
