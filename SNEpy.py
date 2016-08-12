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

#Import matlab files
drug_mat = sio.loadmat('D:\Filip_PSI_mysi\Coherence_StatisticFinalTables\MDL.mat')    #open the .mat
coord_mat = sio.loadmat('pair_coord_source.mat')


x = np.array(coord_mat['x_paircoord'])
y = np.array(coord_mat['y_paircoord'])


drug_str = np.array(drug_mat['MDL'][0])            #opent the structure

data = np.array(drug_str['data'][0])

names = drug_str['namesOfElecs']
average = np.mean(data,0).T

coh = np.zeros([4,6,66])    #band,time,pair

#select the bands
for i,val in enumerate(np.arange(0,56,14)):
    coh[i,:,:] = average[:,val:val+6].T

#Diferentiate the data in time
diff_coh = np.diff(coh,1,0)
diff_coh_res = np.reshape(diff_coh,[18,66],'F').T        
coh_res = np.reshape(coh,[24,66],'F').T


#t-SNE on dataset
mapped = tsne.tsne(diff_coh_res, no_dims, init_dims, perplexity)

labels = kmeans2(mapped,3,10)

#Show the coherence clusters
plt.figure(0)
plt.scatter(mapped[:,0],mapped[:,1],c = labels[1], s = 150,alpha = 0.5)

for i,coord in enumerate(mapped):
    plt.text(coord[0],coord[1],str(names[0][i][0][0]))
plt.show


plt.figure(4)
plt.matshow(diff_coh_res)


plt.figure(5)
plt.matshow(coh_res)
