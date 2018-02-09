import os
import sys
import time
import argparse
import numpy as np
from scipy.misc import imshow

import torch
import torchvision
import torchvision.transforms as transforms
import torchvision.models as models
from torch import nn
from torch import autograd
from torch import optim
from torch.optim.lr_scheduler import StepLR
from torch.nn import functional as F

import ops
import plot
import utils
import generators
import discriminators
from vgg import vgg19, vgg19_bn, VGGextraction


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
    parser.add_argument('--resumeG', default=False)
    parser.add_argument('--resumeD', default=False)
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
        vgg = models.vgg19(pretrained=True).cuda(1)
        netL = VGGextraction(vgg).cuda(1)

    print (netG, netD, netL)
    return (netG, netD, netL)


def train():
    args = load_args()
    np.set_printoptions(precision=4)
    train_gen, dev_gen, test_gen = utils.dataset_iterator(args)
    torch.manual_seed(1)
    netG, netD, netL = load_models(args)

    optimizerG = optim.Adam(netG.parameters(), lr=10e-5)
    optimizerD = optim.Adam(netD.parameters(), lr=10e-4)
    scheduler = StepLR(optimizerG, step_size=500000, gamma=0.1)
    vgg_scale = 0.006  # scales perceptual loss to be on order of MSE loss
    loss_ratio = 0.001 # balancing ratio on content vs GAN
    mse_criterion = nn.MSELoss()
    one = torch.FloatTensor([1]).cuda(0)
    mone = (one * -1).cuda(0)

    gen = utils.inf_train_gen(train_gen)

   
    """ Pretrain SRResNet with MSE only """
    if args.task == 'SRResNet':
        
        """ Attempt to resume generator from checkpoint """
        if args.resumeG is True:
            print ("loading generator from weights")
            state = torch.load('./SRResNet.pt')
            netG.load_state_dict(state)

        for iter in range(1, 1000000):
            start_time = time.time()
            scheduler.step()
            _data_hr = next(gen)
            real_data_lr, real_data_hr = utils.scale_data(args, _data_hr)
            real_data_lr = real_data_lr.cuda(0)
            real_data_hr = real_data_hr.cuda(0)
            real_data_hr_v = autograd.Variable(real_data_hr)
            real_data_lr_v = autograd.Variable(real_data_lr)
            
            fake_hr = netG(real_data_lr_v)

            netG.zero_grad()
            content_loss = mse_criterion(fake_hr, real_data_hr_v)
            """ 
            # something is weird here
            if (iter > 1000) and (content_loss.cpu().data.numpy()[0] > .4):
                print("\nweird thing happened, check for saved logs\n")
                np.save("weird_fake.npy", fake_hr.cpu().data.numpy())
                np.save("weird_real.npy", real_data_hr_v.cpu().data.numpy())
                continue
            """
            psnr = ops.psnr(args, content_loss)
            content_loss.backward()
            optimizerG.step()

            save_dir = './plots/'+args.dataset
            plot.plot('content_loss_(mse)', content_loss.data.cpu().numpy())
            plot.plot('psnr', np.array(psnr))
            data = (real_data_lr, real_data_hr, fake_hr)
            if iter % 100 == 99:
                utils.generate_sr_image(iter, netG, save_dir, args, data)
            if (iter < 5) or (iter % 100 == 99):
                plot.flush()
            plot.tick()
            if iter % 100000 == 0:
                torch.save(netG.state_dict(), './SRResNet_{}.pt'.format(iter))

    elif args.task == 'SRGAN':
        """ Attempt to resume generator from checkpoint """
        if args.resumeD:
            print ("loading generator")
            state = torch.load('./SRGAN_D.pt')
            netD.load_state_dict(state)
        if args.resumeG:
            print ("loading discriminator")
            state = torch.load('./SRResNet.pt')
            netG.load_state_dict(state)

        for p in netL.parameters():  # reset requires_grad
            p.requires_grad = False  # they are set to False below in netG update
        
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
            real_gpu1 = autograd.Variable(real_data_hr.cuda(1), requires_grad=False)
            fake_gpu1 = autograd.Variable(fake_hr.cpu().data.cuda(1), requires_grad=False)

            vgg_real = netL(real_gpu1)
            vgg_fake = netL(fake_gpu1)
            # p1_loss = mse_criterion(fake_gpu1, real_gpu1)
            p2_loss = vgg_scale * mse_criterion(vgg_fake, vgg_real)
            """ 
            test_loss = vgg_scale * ((vgg_fake - vgg_real).pow(2).sum(3).mean())
            if (iteration % 20) == 0:
                print ("test: ", test_loss.cpu().data.numpy()[0])
                print ("mse: ", p2_loss.cpu().data.numpy()[0])
            """
            #print ("mse loss: ", p1_loss.cpu().data.numpy())
            #print ("vgg loss: ", p2_loss.cpu().data.numpy())
            # perceptual_loss =  (p1_loss + p2_loss).cuda(0)
            perceptual_loss = p2_loss.cuda(0)
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
            if iteration % 100 == 0:
                save_dir = './plots/'+args.dataset
                content_loss = np.around(float(perceptual_loss.cpu().data.numpy()[0]), 5)
                adversarial_loss = np.around(float(adv_loss.cpu().data.numpy()[0]), 5)
                disc_cost = np.around(float(D_cost.cpu().data.numpy()[0]), 5)
                psnr = np.around(float(np.array(psnr)), 5)
                print("iter {}\tcontent loss: {}\tadv loss: {}\tdisc cost: {}\tpsnr: {}"
                        .format(iteration, content_loss, adversarial_loss, disc_cost, psnr)
                )
                data = (real_data_lr, real_data_hr, fake_hr)
                utils.generate_sr_image(iteration, netG, save_dir, args, data)
            """
            #plot.plot('w1 distance', Wasserstein_D.cpu().data.numpy())
            if (iteration < 5) or (iteration % 20 == 19):
                plot.flush()
            """
            plot.tick()
            if (iteration % 5000) == 0:
                torch.save(netD.state_dict(), 'SRGAN_D.pt')
                torch.save(netG.state_dict(), 'SRGAN_G.pt')

if __name__ == '__main__':
    train()
