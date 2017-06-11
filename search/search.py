#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun 10 17:18:37 2017

@author: altescy
"""
import os
import pickle
import numpy as np
from chainer import serializers

from search.embedder import LSTMEmbed, sentence2ids, N_VOCAB
from search.make_feats import normalize_norm, FEATS_FP, FEATS_DIMS





def search_images(sentence, files):
    lstmembed = LSTMEmbed(N_VOCAB, FEATS_DIMS, train=False)
    serializers.load_npz('./search/model/lstmembed_4096model_160epoch.npz', \
                     lstmembed)
    lstmembed.reset_state()
    x = sentence2ids(sentence)
    y = normalize_norm(lstmembed(x).data)[0]
    
    with open(FEATS_FP, 'rb') as f:
        feats = pickle.load(f)
    
    ft = [feats[fp] for fp in files]
    
    cossim = np.dot(ft, y)
    print('result: [ {} ]'.format(sentence))
    for i in np.argsort(-cossim):
        print('{}: {}'.format(os.path.basename(files[i]), cossim[i]))
    
    return np.argsort(-cossim)



if __name__ == '__main__':
    import glob
    
    files = [os.path.abspath(fp) for fp in glob.glob('../library/images/*.jpg')]
    result = search_images("There's no such thing as a free lunch.", files)
    print(result)