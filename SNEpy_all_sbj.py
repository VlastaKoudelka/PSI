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

perplexity = 10     
init_dims = 18
no_dims = 2

#Import matlab file
drug_mat = sio.loadmat('D:\Filip_PSI_mysi\Coherence_StatisticFinalTables\psilocin.mat')    #open the .mat
drug_str = drug_mat['psilocin']            #opent the structure
drug_str = drug_str[0]                #first structure
data = drug_str['data']
data = data[0]
names = drug_str['namesOfElecs']
no_sbj = data.shape[0]
no_sbj = 3
#data[0] = data[3]

coh = np.zeros([4,6,66,no_sbj])    #band,time,pair
 

#take apart the bands and time intervals
for i,val in enumerate(np.arange(0,56,14)):
    for j in np.arange(0,no_sbj,1):        
        coh[i,:,:,j] = data[j,val:val+6,:]

#Diferentiate the data in time
diff_coh = np.diff(coh,1,0)
diff_coh_res = np.reshape(diff_coh,[18,66*no_sbj],'F').T 

coh_res = np.reshape(coh,[24,66*no_sbj]).T 
     
'''       
#t-SNE on dataset
mapped = tsne.tsne(diff_coh_res, no_dims, init_dims, perplexity)

labels = kmeans2(mapped,4,10)
    
plt.scatter(mapped[:,0],mapped[:,1],c = labels[1], s = 150,alpha = 0.5)
plt.show
'''

for i in np.arange(0,no_sbj,1):
    plt.figure(i)
    plt.matshow(diff_coh_res[i*66:i*66 + 66,:])
    plt.show
 