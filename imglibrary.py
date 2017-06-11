# -*- coding: utf-8 -*-

import os
import base64
import imghdr
import pickle
import numpy as np
from PIL import Image

from search.search import search_images
from search.search import FEATS_FP

ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg', 'gif'])
IMG_LIBRARY_DIR = os.path.abspath('./library/images')



def img2base64(fp):
    # convert image into base64
    with open(fp, 'rb') as f:
        img = base64.b64encode(f.read()).decode('utf-8')
    imgtype = imghdr.what(fp)
    return 'data:image/{};base64, {}'.format(imgtype, img)


def get_imagelist(imgdir):
    if os.path.exists(FEATS_FP):
        with open(FEATS_FP, 'rb') as f:
            files = list(pickle.load(f).keys())
    else:
        files = []
    
    images = []
    for fn in files:
        fp = os.path.join(IMG_LIBRARY_DIR, fn)
        w, h = Image.open(fp).size
        src  = img2base64(fp)
        images.append({"w": w, "h": h, "src": src, 'fn': fn})
    return images


def list2str(lst):
    ret = ''
    for i in lst:
        ret += str(i) + ','
    return ret

def str2intlist(s):
    ret = []
    for d in s.split(','):
        if d:
            ret += [int(d)]
    return ret


class ImgLibrary:
    def __init__(self, libdir=IMG_LIBRARY_DIR):
        self.libdir  = libdir
        self.images  = []
        self.reload()

    def __call__(self):
        return self.images

    def reload(self):
        self.images = get_imagelist(self.libdir)

    def argsort_by_alphabet(self):
        # sort images into alphabet order
        if self.images:
            return list2str(np.argsort([img['fn'] for img in self.images]))
        return []
    
    def argsort_by_alphabet_alt(self):
        return self.inverse(self.argsort_by_alphabet())
    
    def argsearch(self, sentence):
        files = [img['fn'] for img in self.images]
        return list2str(search_images(sentence, files))

    def sort(self, indexs):
        indexs = str2intlist(indexs)
        return [self.images[i] for i in indexs]
    
    def inverse(self, s):
        return list2str(str2intlist(s)[::-1])
            