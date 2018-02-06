import os
import glob
import gzip
import urllib
import numpy as np
from scipy.misc import imread
import _pickle as pickle

data_dir = '../../../images/imagenet12/imagenet_val/'


def imagenet_generator(filenames, batch_size, data_dir):
    all_data = []
    all_labels = []
    for filename in filenames:
        im = imread(filename)
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
    filelist = glob.glob(data_dir+'*.png')[:30000]
    trainlist = filelist[:int(0.8*len(filelist))]
    testlist = filelist[int(0.8*len(filelist)):]

    return (
        imagenet_generator(trainlist, batch_size, data_dir), 
        imagenet_generator(testlist, batch_size, data_dir)
    )
