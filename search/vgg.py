#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun 10 15:22:51 2017

@author: altescy
"""

import os
import pickle
import numpy as np
from scipy.misc import imread, imresize

SHAPE = (224, 224)

BASE_DIR = os.path.dirname(__file__)
VGG_MODEL_FILE = os.path.join(BASE_DIR,
                              './model/VGG_ILSVRC_19_layers.chainermodel.pkl')


def clip(x):
    fixed_h, fixed_w = SHAPE
    h, w, _  = x.shape
    shape = (fixed_h, fixed_w * w // h) if w > h else (fixed_h * h // w, fixed_w)
    
    left   = (shape[1] - fixed_w) // 2
    top    = (shape[0] - fixed_h) // 2
    right  = left + fixed_w
    bottom = top + fixed_h
    
    x = imresize(x, shape)
    return x[top:bottom, left:right, :]


def resize(x):
    h, w, ch = x.shape
    if h < w:
        padlen = (w - h) // 2
        pad = np.zeros((padlen, w, ch), dtype=np.uint8)
        x = np.concatenate((pad, x, pad), axis=0)
    elif h > w:
        padlen = (h - w) // 2
        pad = np.zeros((h, padlen, ch), dtype=np.uint8)
        x = np.concatenate((pad, x, pad), axis=1)
    
    return imresize(x, SHAPE)


def subtract_mean(x):
    x = x.astype(np.float32)
    mean = np.zeros(x.shape, dtype=np.float32)
    
    mean[:, :, 0] = 104
    mean[:, :, 1] = 117
    mean[:, :, 2] = 124
    
    return x - mean


def load_image(fp):
    x = imread(fp, mode='RGB')
    x = resize(x)
    x = subtract_mean(x)
    x = x[:, :, ::-1].transpose((2, 0, 1))
    return x


class VGG19:
    def __init__(self, fp=VGG_MODEL_FILE):
        print ("load model: {}".format(fp))
        with open(fp, 'rb') as f:
            self.model = pickle.load(f)

    
    def __call__(self, x):
        return self.model(inputs={'data': x}, outputs=['fc7'], train=False)

if __name__ == '__main__':
    pass