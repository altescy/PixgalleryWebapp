# -*- coding: utf-8 -*-

import os
import pickle
import numpy as np
from chainer import links as L, functions as F, Chain


WD2ID_FP = os.path.abspath('./search/data/wd2id.pkl')
with open(WD2ID_FP, 'rb') as f:
    WD2ID = pickle.load(f)
N_VOCAB = len(WD2ID)


class LSTMEmbed(Chain):
    def __init__(self, in_size, out_size, train=True):
        super().__init__(
            embed=L.EmbedID(in_size, 512),
            lstm=L.LSTM(512, 512),
            out=L.Linear(512, out_size)
        )
        self.train = train

    def __call__(self, x):
        x = F.transpose_sequence(x)
        for x_ in x:
            self.lstm(F.dropout(self.embed(x_), train=self.train))
        h = self.out(F.dropout(self.lstm.h, train=self.train))
        return h

    def reset_state(self):
        self.lstm.reset_state()


def clean_sentence(sentence):
    sentence = sentence.lower()
    ret = ''
    for c in sentence:
        if c == ' ' or c.isalpha():
            ret += c
    return ret


def sentence2ids(sentence=None):
    """
    convert sentence into ids
    """
    sentence = clean_sentence(sentence)
    
    x = [WD2ID['<bos>']]
    for w in sentence.split(' '):
        if w not in WD2ID:
            w = '<unk>'
        x.append(WD2ID[w])
    x.append(WD2ID['<eos>'])
    x = np.array(x, dtype=np.int32)
    return np.expand_dims(x, axis=0)