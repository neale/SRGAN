import os
import glob
import gzip
import urllib
import numpy as np
from scipy.misc import imread, imresize
import _pickle as pickle

data_dir = '../../../images/raise/'


def raise_generator(filenames, batch_size, data_dir):
    all_data = []
    all_labels = []
    for filename in filenames:
        im = imresize(imread(filename), (299, 299))
        if im.ndim != 3:
            continue
        all_data.append(im)
        # all_labels.append(labels)

    images = np.array(all_data)
    # labels = np.concatenate(all_labels, axis=0)

    def get_epoch():
        rng_state = np.random.get_state()
        np.random.shuffle(images)
        # np.random.set_state(rng_state)
        # np.random.shuffle(labels)

        for i in range(int(len(images) / batch_size)):
            yield (np.copy(images[i*batch_size:(i+1)*batch_size]), None)
                   # labels[i*batch_size:(i+1)*batch_size])

    return get_epoch


def load(batch_size, data_dir=data_dir):
    dir_hr = data_dir+'RAISE_HR/'
    dir_lr = data_dir+'RAISE_HR/'
    filelist_hr = glob.glob(dir_hr+'*.png')[:500]
    filelist_lr = glob.glob(dir_lr+'*.png')[:500]
    print (len(filelist_hr), len(filelist_lr))

    return (
        raise_generator(filelist_hr, batch_size, data_dir), 
        raise_generator(filelist_lr, batch_size, data_dir)
    )
