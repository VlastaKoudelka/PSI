# -*- coding: utf-8 -*-
"""
Created on Wed Apr 05 13:17:27 2017

@author: vavra
"""

import numpy as np
import matplotlib.colors as clr
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.cm as cm
import scipy.io as sio
from sklearn.metrics import silhouette_score as crit
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA as pca
import tsne
import csv

filename = 'selectElec.csv'
path = '\\SRC\\'

A = np.loadtxt([path + filename],dtype = 'S')
#text_file = open([path + filename], "r")
#lines = text_file.readlines()
#reader = csv.reader(filename,delimiter='\t')
#reader.