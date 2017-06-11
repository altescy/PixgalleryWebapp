#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun 10 15:46:03 2017

@author: altescy
"""

import os
import glob
import pickle
import numpy as np

from search.vgg import load_image, VGG19


BASE_DIR = os.path.dirname(__file__)

FEATS_DIR = os.path.join(BASE_DIR, '..//library/feats')
FEATS_FP = os.path.join(FEATS_DIR, 'feats.pkl')
FEATS_DIMS = 4096


def normalize_norm(x):
    return x / np.linalg.norm(x, axis=1).reshape(-1, 1)


def add_image(fp):
    if os.path.exists(FEATS_FP):
        with open(FEATS_FP, 'rb') as f:
            feats = pickle.load(f)
    else:
        feats = {}
    
    x = np.expand_dims(load_image(fp), axis=0).astype(np.float32)
    
    vgg = VGG19()
    ft, = vgg(x)
    feats[fp] = normalize_norm(ft.data)[0]
    
    with open(FEATS_FP, 'wb') as f:
        pickle.dump(feats, f)
    del(vgg, feats, ft)


def make_datas(files):
    feats = {}
    
    x = []
    for fp in files:
        x.append(load_image(fp))
    x = np.array(x, dtype=np.float32)
    
    vgg = VGG19()
    ft, = vgg(x)
    ft = normalize_norm(ft.data)
    
    for fp, feat in zip(files, ft):
        feats[fp] = feat
    
    with open(FEATS_FP, 'wb') as f:
        pickle.dump(feats, f)

    del(vgg, feats, ft)



if __name__ == '__main__':
    ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg', 'gif'])
    files = []
    for ext in ALLOWED_EXTENSIONS:
        files += glob.glob('../library/images/*.{}'.format(ext))
    
    feats = {}
    vgg = VGG19()
    
    for i, fp in enumerate(files):
        fp = os.path.relpath(fp)
    
        x = np.expand_dims(load_image(fp), axis=0).astype(np.float32)
        
        ft, = vgg(x)
        feats[os.path.basename(fp)] = normalize_norm(ft.data)[0]
        print(os.path.basename(fp))
        
    with open(FEATS_FP, 'wb') as f:
        pickle.dump(feats, f)
    del(vgg, feats, ft)
