import torch
import numpy as np
from math import log10
import torch.autograd as autograd
import torch.nn as nn


def calc_gradient_penalty(args, model, real_data, gen_data):
    batch_size = args.batch_size
    datashape = model.shape
    alpha = torch.rand(batch_size, 1)
    alpha = alpha.expand(batch_size, int(real_data.nelement()/batch_size))
    alpha = alpha.contiguous().view(batch_size, *datashape).cuda()
    interpolates = alpha * real_data + ((1 - alpha) * gen_data)
    interpolates = interpolates.cuda()
    interpolates = autograd.Variable(interpolates, requires_grad=True)
    disc_interpolates = model(interpolates)
    gradients = autograd.grad(outputs=disc_interpolates, 
            inputs=interpolates,
            grad_outputs=torch.ones(disc_interpolates.size()).cuda(),      
            create_graph=True, 
            retain_graph=True, 
            only_inputs=True)[0]

    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * args.gp
    return gradient_penalty


def psnr(args, mse):
    psnr = 10. * log10(1. / mse.data[0]+0.00001)
    return psnr
