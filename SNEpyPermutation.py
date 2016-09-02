# -*- coding: utf-8 -*-
"""
Created on Mon Aug 29 15:52:22 2016

@author: vavra
"""

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
import time
plt.close('all')

perplexity = 5
init_dims = 18
no_dims = 2
no_clstr = 4
no_pairs = 36
no_perm = 5
no_map = 10 #a number of t-SNE runs in one permutation
no_subs = 10
t = time.time()
#Import matlab files
drug_mat = sio.loadmat('D:\Filip_PSI_mysi\Coherence_StatisticFinalTables\MATSoubory\psilocin.mat')    #open the .mat
coord_mat = sio.loadmat('Locations.mat')
brain = mpimg.imread('brain2.png')

x_cord = np.array(coord_mat['x_pair_sel'])       #x_pair_all for all pairs
y_cord = np.array(coord_mat['y_pair_sel'])
index = np.array(coord_mat['index'][0])     #indices of used electrode pairs

drug_str = np.array(drug_mat['psilocin'][0])            #opent the structure
data = np.array(drug_str['data'][0])
names = drug_str['namesOfElecs']

data = data[:,:,index]                      #select the used pairs
#index = np.arange(0,no_pairs,1)            #uncomment for all pairs
average = np.mean(data,0).T

coh = np.zeros([4,6,no_pairs])    #band,time,pair

#select the bands
for i,val in enumerate(np.arange(0,56,14)):
    coh[i,:,:] = average[:,val:val+6].T

#Diferentiate the data in time
diff_coh = np.diff(coh,1,0)
diff_coh_res = np.reshape(diff_coh,[18,no_pairs],'F').T        
coh_res = np.reshape(coh,[24,no_pairs],'F').T

silhouette = np.zeros([no_perm,no_pairs])
for i in np.arange(no_perm):
    C_min = 1e10
    coh_perm = diff_coh_res[np.random.permutation(no_pairs)]
    #coh_perm = diff_coh_res
    for j in np.arange(no_map):
        [mapped,C] = tsne.tsne(coh_perm, no_dims, init_dims, perplexity)
        if C < C_min:
            win_map = mapped    #a map with the lowest error
            C_min = C
    for k in np.arange(2,no_pairs):
        labels = kmeans2(win_map,k,10)[1]
        silhouette[i,k] = sk.silhouette_score(win_map,labels)
 
plt.figure   
plt.plot(np.mean(silhouette.T,1),color = 'red')
plt.show

plt.figure
plt.plot(silhouette.T,color = 'red')
plt.show
np.save('perm_test.npy',silhouette)
print(time.time()-t)
#source = diff_coh_res[np.random.permutation(no_pairs)]
#for i in np.arange(no_subs - 1):
#    source = np.r_[source,diff_coh_res[np.random.permutation(no_pairs)]]
#
#[mapped,C] = tsne.tsne(source, no_dims, init_dims, perplexity)
#plt.scatter(mapped[:,0],mapped[:,1])

        
            
        
