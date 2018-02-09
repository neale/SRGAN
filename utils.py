import torch
import torch.autograd as autograd
import torchvision
import torchvision.transforms as transforms
import numpy as np
from data import Mnist
from data import Cifar10
from data import Imagenet
from data import Raise
from scipy.misc import imsave
import matplotlib.pyplot as plt

DATA_DIR = '../adversarial-toolbox/images/'
def dataset_iterator(args):
    if args.dataset == 'mnist':
        train_gen, dev_gen, test_gen = Mnist.load(args.batch_size, args.batch_size)
    if args.dataset == 'cifar10':
        data_dir = DATA_DIR+'cifar-10-batches-py/'
        train_gen, dev_gen = Cifar10.load(args.batch_size, data_dir)
        test_gen = None
    if args.dataset == 'imagenet':
        data_dir = DATA_DIR+'imagenet12/imagenet_val_png/'
        train_gen, dev_gen = Imagenet.load(args.batch_size, data_dir)
        test_gen = None
    if args.dataset == 'raise':
        data_dir = DATA_DIR+'raise/'
        train_gen = Raise.load(args.batch_size, data_dir)
        dev_gen = train_gen
        test_gen = None
    else:
        raise ValueError

    return (train_gen, dev_gen, test_gen)


def inf_train_gen(train_gen):
    while True:
        for images, _ in train_gen():
            yield images


def scale_data(args, data):

    transform = transforms.Compose([transforms.ToPILImage(),
        transforms.RandomCrop(24*4),
        transforms.ToTensor()])

    normalize = transforms.Normalize(mean = [.5, .5, .5],
            std = [.5, .5, .5])

    scale = transforms.Compose([transforms.ToPILImage(),
        transforms.Resize(24),
        transforms.ToTensor(),
        transforms.Normalize(mean = [.5, .5, .5],
            std = [.5, .5, .5])
        ]) 
    low_res = torch.FloatTensor(args.batch_size, 3, 24, 24)
    high_res = torch.FloatTensor(args.batch_size, 3, 96, 96)

    for i in range(len(data)):
        x = transform(data[i])
        low_res[i] = scale(x)
        high_res[i] = normalize(x)
    
    return low_res, high_res

 
def generate_sr_image(iter, netG, save_path, args, data):
    lr, hr, sr = data
    batch_size = args.batch_size
    if netG._name == 'mnistG':
        samples = samples.view(batch_size, 28, 28)
    lr = lr.cpu().numpy().transpose(0, 2, 3, 1)[..., ::-1]
    hr = hr.cpu().numpy().transpose(0, 2, 3, 1)[..., ::-1]
    sr = sr.cpu().data.numpy().transpose(0, 2, 3, 1)[..., ::-1]
    lr = (lr - np.min(lr))/(np.max(lr) - np.min(lr))
    hr = (hr - np.min(hr))/(np.max(hr) - np.min(hr))
    sr = ((sr+2)/2*127.5).astype(np.uint8)
    #print ("SR: ", np.max(sr), np.min(sr), sr.shape)
    save_name = save_path+'/SRGAN_iter_{}'.format(iter)
    show_sr(lr[0], hr[0], sr[0], save_name)


def generate_image(iter, model, save_path, args):
    batch_size = args.batch_size
    datashape = model.shape
    if model._name == 'mnistG':
        fixed_noise_128 = torch.randn(batch_size, args.dim).cuda()
    else:
        fixed_noise_128 = torch.randn(128, args.dim).cuda()
    noisev = autograd.Variable(fixed_noise_128, volatile=True)
    samples = model(noisev)
    if model._name == 'mnistG':
        samples = samples.view(batch_size, 28, 28)
    else:
        samples = samples.view(-1, *(datashape[::-1]))
        samples = samples.mul(0.5).add(0.5)
    samples = samples.cpu().data.numpy()
    save_images(samples, save_path+'/samples_{}.jpg'.format(iter))


def show_sr(lr, hr, sr, name):
    #lr = (lr - np.min(lr))/(np.max(lr) - np.min(lr))
    #hr = (hr - np.min(hr))/(np.max(hr) - np.min(hr))
    #sr = ((sr+2)/2*127.5).astype(np.uint8)
    #print ("SR: ", np.max(sr), np.min(sr), sr.shape)
    #print ("LR: ", np.max(lr), np.min(lr), lr.shape)
    #print ("HR: ", np.max(hr), np.min(hr), hr.shape)
    plt.ion()
    plt.suptitle("LR, HR, SR")
    plt.subplot(1, 3, 1)
    plt.imshow((lr+1))
    plt.axis('off')
    plt.subplot(1, 3, 2)
    plt.imshow((hr+1))
    plt.axis('off')
    plt.subplot(1, 3, 3)
    plt.imshow(sr)
    plt.axis('off')
    plt.draw()
    plt.pause(0.001)
    plt.savefig(name, format='png')


def save_images(X, save_path, use_np=False):
    # [0, 1] -> [0,255]
    plt.ion()
    if not use_np:
        if isinstance(X.flatten()[0], np.floating):
            X = (255.99*X).astype('uint8')
    n_samples = X.shape[0]
    rows = int(np.sqrt(n_samples))
    while n_samples % rows != 0:
        rows -= 1
    nh, nw = rows, int(n_samples/rows)
    if X.ndim == 2:
        s = int(np.sqrt(X.shape[1]))
        X = np.reshape(X, (X.shape[0], s, s))
    if X.ndim == 4:
        X = X.transpose(0,2,3,1)
        h, w = X[0].shape[:2]
        img = np.zeros((h*nh, w*nw, 3))
    elif X.ndim == 3:
        h, w = X[0].shape[:2]
        img = np.zeros((h*nh, w*nw))
    for n, x in enumerate(X):
        j = int(n/nw)
        i = int(n%nw)
        img[j*h:j*h+h, i*w:i*w+w] = x

    #plt.imshow(img, cmap='gray')
    #plt.draw()
    #plt.pause(0.001)

    if use_np:
        np.save(save_path, img)
    else:
        imsave(save_path, img)


