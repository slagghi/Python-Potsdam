#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 13 12:50:56 2018

@author: jacopo
"""

import scipy.io as sio
import numpy as np

def save_np(filename):
    indir="matfiles/"+filename+".mat"
    temp=sio.loadmat(indir)
    temp=temp[filename]
    outdir="mat2numpy/"+filename+".npy"
    np.save(outdir,temp)
    
names=['l11b',  'l16b',  'l1b',   'l21m',  'l25m',  'l29m',  'l32b',  'l7m',
       'l11f',  'l16f',  'l1f',   'l21x',  'l25x',  'l29x',  'l32f',  'l7x',
       'l12b',  'l17b',  'l20b',  'l24b',  'l28b',  'l2b',   'l6b',
       'l12m',  'l17m',  'l20f',  'l24f',  'l28f',  'l2m',   'l6f',
       'l12x',  'l17x',  'l21b',  'l25b',  'l29b',  'l2x',   'l7b']

for st in names:
    save_np(st)
    