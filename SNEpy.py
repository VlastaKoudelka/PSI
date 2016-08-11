# -*- coding: utf-8 -*-
"""
Created on Wed Aug 10 13:59:44 2016

@author: vavra
"""
from scipy.cluster.vq import kmeans2
import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
import tsne
plt.close('all')

perplexity = 5
init_dims = 18
no_dims = 2

#Import matlab file
drug_mat = sio.loadmat('D:\Filip_PSI_mysi\Coherence_StatisticFinalTables\psilocin.mat')    #open the .mat
drug_str = drug_mat['psilocin']            #opent the structure
drug_str = drug_str[0]                #first structure
data = drug_str['data']
data = data[0]
names = drug_str['namesOfElecs']
average = np.mean(data,0).T

coh = np.zeros([6,4,66])    #band,time,pair

#select the bands
for i in np.arange(0,6,1):
    coh[i,:,:] = average[:,i:56:14].T

#Diferentiate the data in time
diff_coh = np.diff(coh,1,1)
diff_coh_res = np.reshape(diff_coh,[18,66],'F').T        

#t-SNE on dataset
mapped = tsne.tsne(diff_coh_res, no_dims, init_dims, perplexity)

labels = kmeans2(mapped,3,10)

plt.scatter(mapped[:,0],mapped[:,1],c = labels[1], s = 150,alpha = 0.5)
plt.show