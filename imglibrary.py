# -*- coding: utf-8 -*-

import os
import base64
import imghdr
import pickle
from PIL import Image

from search.search import search_images
from search.make_feats import FEATS_FP

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


class ImgLibrary:
    def __init__(self, libdir=IMG_LIBRARY_DIR):
        self.libdir  = libdir
        self.images  = None
        self.subset  = None
        self.inverse = False
        self.sentence = ''
        self.reload()

    def __call__(self):
        return self.invout(self.subset)

    def load_all(self):
        self.subset = self.images.copy()

    def invout(self, s):
        if self.inverse:
            return s[::-1]
        return s

    def reload(self):
        self.images = get_imagelist(self.libdir)

    def sort_by_alphabet(self):
        # sort images into alphabet order
        if self.subset:
            self.subset = sorted(self.subset, key=lambda img:img['fn'])
    
    def sort_by_alphabet_alt(self):
        self.sort_by_alphabet()
        self.subset = self.subset[::-1]
    
    def search(self, sentence):
        self.sentence = sentence
        files = [img['fn'] for img in self.images]
        result = search_images(sentence, files)
        self.subset = [self.images[i] for i in result]
