# -*- coding: utf-8 -*-
"""
Created on Wed Aug 10 13:59:44 2016

@author: vavra
"""
from scipy.cluster.vq import kmeans2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.cm as cm
import scipy.io as sio
import sklearn.metrics as sk
import tsne
plt.close('all')

perplexity = 5
init_dims = 18
no_dims = 2
no_clstr = 3
no_pairs = 36


#Import matlab files
drug_mat = sio.loadmat('D:\Filip_PSI_mysi\Coherence_StatisticFinalTables\SB.mat')    #open the .mat
coord_mat = sio.loadmat('Locations.mat')
brain = mpimg.imread('brain2.png')


x = np.array(coord_mat['x_pair_sel'])       #x_pair_all for all pairs
y = np.array(coord_mat['y_pair_sel'])
index = np.array(coord_mat['index'][0])     #indices of used electrode pairs

drug_str = np.array(drug_mat['SB'][0])            #opent the structure
data = np.array(drug_str['data'][0])
names = drug_str['namesOfElecs']

data = data[:,:,index]                      #select the used electrodes
average = np.mean(data,0).T

coh = np.zeros([4,6,no_pairs])    #band,time,pair

#select the bands
for i,val in enumerate(np.arange(0,56,14)):
    coh[i,:,:] = average[:,val:val+6].T

#Diferentiate the data in time
diff_coh = np.diff(coh,1,0)
diff_coh_res = np.reshape(diff_coh,[18,no_pairs],'F').T        
coh_res = np.reshape(coh,[24,no_pairs],'F').T


#t-SNE on dataset
mapped = tsne.tsne(diff_coh_res, no_dims, init_dims, perplexity)

#Clusterring
labels = kmeans2(mapped,no_clstr,10)[1]

#Silhoulette criterion
clust_crit = np.zeros([len(labels),1])
for i in np.arange(2,len(labels)):
    labels2 = kmeans2(mapped,i,iter = 10)[1]
    clust_crit[i] = sk.silhouette_score(mapped,labels2)


#---Show the coherence clusters
colors = cm.jet(np.linspace(0, 1, no_clstr))

plt.figure(0)
plt.scatter(mapped[:,0],mapped[:,1],c = colors[labels], s = 100,alpha = 0.5)

#---Show electrode numbers
for i,coord in enumerate(mapped):
    plt.text(coord[0],coord[1],str(names[0][index[i]][0][0]))
    plt.show

#---Show brain
plt.figure(1)
plt.imshow(brain)

for i in np.arange(len(x)):
    plt.plot(x[i],y[i], c = colors[labels[i]],linewidth = 2)  
    plt.show
        
#---Show difference matrix
plt.figure(2)
plt.matshow(diff_coh_res)

#---Show coherence matrix
plt.figure(3)
plt.matshow(coh_res)

#---Show Silhoulette
plt.figure(5)
plt.plot(clust_crit)
plt.show
