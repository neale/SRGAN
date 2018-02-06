import os
import sys
import time
import argparse
import numpy as np
from scipy.misc import imshow

import torch
import torchvision
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
from data import mnist
from data import cifar10

def load_args():

    parser = argparse.ArgumentParser(description='recover-gan')
    parser.add_argument('-d', '--dim', default=64, type=int, help='latent space')
    parser.add_argument('-l', '--gp', default=10, type=int, help='grad penalty')
    parser.add_argument('-b', '--batch_size', default=50, type=int)
    parser.add_argument('-e', '--epochs', default=200000, type=int)
    parser.add_argument('-o', '--output_dim', default=784, type=int)
    parser.add_argument('--dataset', default='mnist')
    args = parser.parse_args()
    return args

def load_models(args):
    if args.dataset == 'mnist':
        netG = generators.MNISTgenerator(args).cuda()
        netD = discriminators.MNISTdiscriminator(args).cuda()
        netE = encoders.MNISTencoder(args).cuda()

    if args.dataset == 'cifar10':
        netG = generators.CIFARgenerator(args).cuda()
        netD = discriminators.CIFARdiscriminator(args).cuda()
        netE = encoders.CIFARencoder(args).cuda()
	
    print (netG, netD, netE)
    return (netG, netD, netE)


def stack_data(args, _data):
    if args.dataset == 'cifar10':
        preprocess = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        datashape = (3, 32, 32)
        _data = _data.reshape(args.batch_size, *datashape).transpose(0, 2, 3, 1)
        real_data = torch.stack([preprocess(item) for item in _data]).cuda()
    elif args.dataset == 'mnist':
        real_data = torch.Tensor(_data).cuda()

    return real_data


def train():
    args = load_args()
    train_gen, dev_gen, test_gen = utils.dataset_iterator(args)
    torch.manual_seed(1)
    np.set_printoptions(precision=4) 
    netG, netD, netE = load_models(args)

    optimizerD = optim.Adam(netD.parameters(), lr=1e-4, betas=(0.5, 0.9))
    optimizerG = optim.Adam(netG.parameters(), lr=1e-4, betas=(0.5, 0.9))
    optimizerE = optim.Adam(netE.parameters(), lr=1e-4, betas=(0.5, 0.9))
    ae_criterion = nn.MSELoss()
    one = torch.FloatTensor([1]).cuda()
    mone = (one * -1).cuda()

    gen = utils.inf_train_gen(train_gen)

    preprocess = torchvision.transforms.Compose([
	torchvision.transforms.ToTensor(),
	torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    for iteration in range(args.epochs):
        start_time = time.time()
        """ Update AutoEncoder """
        for p in netD.parameters():
            p.requires_grad = False
        netG.zero_grad()
        netE.zero_grad()
        _data = next(gen)
        real_data = stack_data(args, _data)
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
            real_data = stack_data(args, _data)
            real_data_v = autograd.Variable(real_data)
            # train with real data
            netD.zero_grad()
            D_real = netD(real_data_v)
            D_real = D_real.mean()
            D_real.backward(mone)
            # train with fake data
            noise = torch.randn(args.batch_size, args.dim).cuda()
            noisev = autograd.Variable(noise, volatile=True)  # totally freeze netG
            fake = autograd.Variable(netG(noisev).data)
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
        noise = torch.randn(args.batch_size, args.dim).cuda()
        noisev = autograd.Variable(noise)
        fake = netG(noisev)
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
        plot.plot(save_dir, '/ae cost', np.round(ae_loss.data.cpu().numpy(), 4))
        
        # Calculate dev loss and generate samples every 100 iters
        if iteration % 100 == 99:
            dev_disc_costs = []
            for images, _ in dev_gen():
                imgs = stack_data(args, images) 
                imgs_v = autograd.Variable(imgs, volatile=True)
                D = netD(imgs_v)
                _dev_disc_cost = -D.mean().cpu().data.numpy()
                dev_disc_costs.append(_dev_disc_cost)
            plot.plot(save_dir ,'/dev disc cost', np.round(np.mean(dev_disc_costs), 4))
            
            # utils.generate_image(iteration, netG, save_dir, args)
            utils.generate_ae_image(iteration, netE, netG, save_dir, args, real_data_v)
        # Save logs every 100 iters 
        if (iteration < 5) or (iteration % 100 == 99):
            plot.flush()
        plot.tick()
        
if __name__ == '__main__':
    train()
