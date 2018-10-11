#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 11 12:46:33 2018

@author: sofian
"""

from keras.datasets import mnist
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation,Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.optimizers import SGD
from keras.utils import np_utils
import matplotlib
import matplotlib.pyplot as plt
import scipy
import scipy.ndimage
import numpy as np
from keras import backend as K
import keras
import pandas as pd


import os
liste_image=os.listdir("//Users//sofian//Documents//data_airbus_defi//train//other//")

import matplotlib.image as mpimg
import numpy as np

tab_image = pd.DataFrame(columns =['nom_img', 'eolien'])

for image in liste_image:
    img = mpimg.imread('/Users/sofian/Documents/data_airbus_defi/train/other/'+image)
    tab_image.loc[len(tab_image)] = [image, 0]

liste_image2=os.listdir("//Users//sofian//Documents//data_airbus_defi//train//target//")

for image2 in liste_image2:
    img = mpimg.imread('/Users/sofian/Documents/data_airbus_defi/train/target/'+image2)
    tab_image.loc[len(tab_image)] = [image2, 1]


