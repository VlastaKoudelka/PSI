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
coord_mat = sio.loadmat('Locations.mat')
index = np.array(coord_mat['index'][0])     #indices of used electrode pairs
drug_str = drug_str[0]                #first structure
data = drug_str['data']
data = data[0]
names = drug_str['namesOfElecs']
no_sbj = data.shape[0]
no_sbj = 1
data = data[:,:,index]                      #select the used pairs
data[0] = data[3]

coh = np.zeros([4,6,36,no_sbj])    #band,time,pair
 

#take apart the bands and time intervals
for i,val in enumerate(np.arange(0,56,14)):
    for j in np.arange(0,no_sbj,1):        
        coh[i,:,:,j] = data[j,val:val+6,:]

#Diferentiate the data in time
diff_coh = np.diff(coh,1,0)
diff_coh_res = np.reshape(diff_coh,[18,36*no_sbj],'F').T 

coh_res = np.reshape(coh,[24,36*no_sbj]).T 
     
     
#t-SNE on dataset
mapped = tsne.tsne(diff_coh_res, no_dims, init_dims, perplexity)

labels = kmeans2(mapped,4,10)

plt.figure(0)    
plt.scatter(mapped[:,0],mapped[:,1],c = labels[1], s = 150,alpha = 0.5)
plt.show


for i in np.arange(no_sbj):
    plt.figure(i)
    plt.matshow(diff_coh_res[i*36:i*36 + 36,:])    
    plt.colorbar()
    plt.text(0.5,36.5,r'$\delta$',fontsize=25)
    plt.text(3.5,36.5,r'$\theta$',fontsize=25)
    plt.text(6.5,36.5,r'$\alpha$',fontsize=25)
    plt.text(9.5,36.5,r'$\beta_1$',fontsize=25)
    plt.text(12.5,36.5,r'$\beta_2$',fontsize=25)
    plt.text(15.5,36.5,r'$\gamma$',fontsize=25)
    plt.show