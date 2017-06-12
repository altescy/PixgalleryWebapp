#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun  4 13:19:28 2017

@author: altescy
"""

import numpy as np
from scipy.misc import imread, imresize
import pickle


SHAPE = (224, 224)


def resize(x):
    fixed_h, fixed_w = SHAPE
    h, w, _  = x.shape
    shape = (fixed_h, fixed_w * w // h) if w > h else (fixed_h * h // w, fixed_w)
    
    left   = (shape[1] - fixed_w) // 2
    top    = (shape[0] - fixed_h) // 2
    right  = left + fixed_w
    bottom = top + fixed_h
    
    x = imresize(x, shape)
    return x[top:bottom, left:right, :]


def subtract_mean(x):
    x = x.astype(np.float32)
    mean = np.zeros(x.shape, dtype=np.float32)
    
    mean[:, :, 0] = 104
    mean[:, :, 1] = 117
    mean[:, :, 2] = 124
    
    return x - mean


def load_and_preprocess(fp):
    x = imread(fp, mode='RGB')
    x = resize(x)
    x = subtract_mean(x)
    x = x[:, :, ::-1].transpose((2, 0, 1))
    return x


class VGG19:
    def __init__(self, fp):
        print ("load model: {}".format(fp))
        with open(fp, 'rb') as f:
            self.model = pickle.load(f)

    
    def __call__(self, x):
        return self.model(inputs={'data': x}, outputs=['fc7'], train=False)



if __name__ =='__main__':
    vgg = VGG19('VGG/VGG_ILSVRC_19_layers.chainermodel.pkl')
    x = load_and_preprocess('images/acoustic-1851248__340.jpg')
    x = np.expand_dims(x, 0)
    y,  = vgg(x)
    print(y.shape)
