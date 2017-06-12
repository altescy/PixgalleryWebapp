#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun  4 15:46:07 2017

@author: altescy
"""

import numpy as np
import pickle
import json
import glob


if __name__ == '__main__':
    file_dir = './images'
    files = []
    for ext in ['png', 'jpg', 'jpeg', 'gif', 'bmp'] :
        files += glob.glob(file_dir + '/*.' + ext)
    
    feats = 
    