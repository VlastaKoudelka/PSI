# -*- coding: utf-8 -*-
"""
Created on Wed Aug 31 12:00:42 2016

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

drug_name = 'PSI-WAY'
perplexity = 5
init_dims = 18
no_dims = 2
no_pairs = 36
no_inst = 5
no_kmeans = 10

#Import matlab files
drug_mat = sio.loadmat('D:\Filip_PSI_mysi\Coherence_StatisticFinalTables\MATSoubory\psilocinWAY.mat')    #open the .mat
sal_mat = sio.loadmat('D:\Filip_PSI_mysi\Coherence_StatisticFinalTables\MATSoubory\saline.mat')
coord_mat = sio.loadmat('Locations.mat')
index = np.array(coord_mat['index'][0])     #indices of used electrode pairs

drug_str = np.array(drug_mat['psilocinWAY'][0])            #opent the structure
sal_str = np.array(sal_mat['saline'][0])
data_drug = np.array(drug_str['data'][0])
data_sal = np.array(sal_str['data'][0])
names = drug_str['namesOfElecs']

data_drug = data_drug[:,:,index]                      #select the used pairs
data_sal = data_sal[:,:,index]
#index = np.arange(0,no_pairs,1)            #uncomment for all pairs
data_drug = np.mean(data_drug,0).T
data_sal = np.mean(data_sal,0).T
coh_drug = np.zeros([4,6,no_pairs])    #band,time,pair
coh_sal = np.zeros([4,6,no_pairs])

#select the bands
for i,val in enumerate(np.arange(0,56,14)):
    coh_drug[i,:,:] = data_drug[:,val:val+6].T
    coh_sal[i,:,:] = data_sal[:,val:val+6].T

#Diferentiate the data in time
diff_coh_drug = np.diff(coh_drug,1,0)
diff_coh_sal = np.diff(coh_sal,1,0)

diff_coh_res_drug = np.reshape(diff_coh_drug,[18,no_pairs],'F').T        
diff_coh_res_sal = np.reshape(diff_coh_sal,[18,no_pairs],'F').T  


#Silhoulette criterion for drug

labels2 = np.zeros([no_pairs,no_kmeans,no_pairs])
clust_crit = np.zeros([no_pairs,no_kmeans])
crit_drug = np.zeros([no_inst,no_pairs])

for k in np.arange(no_inst):
    diff_coh_res_drug = diff_coh_res_drug[np.random.permutation(no_pairs)]
    [mapped,C] = tsne.tsne(diff_coh_res_drug, no_dims, init_dims, perplexity)
    for i in np.arange(2,no_pairs):
        for j in np.arange(no_kmeans):        
            labels2[i,j,:] = kmeans2(mapped,i,10)[1]
            clust_crit[i,j] = sk.silhouette_score(mapped,labels2[i,j])
    crit_drug[k,:] = np.mean(clust_crit,1)
    
 #Silhoulette criterion for saline  
labels2 = np.zeros([no_pairs,no_kmeans,no_pairs])
clust_crit = np.zeros([no_pairs,no_kmeans])
crit_sal = np.zeros([no_inst,no_pairs])

for k in np.arange(no_inst):
    diff_coh_res_sal = diff_coh_res_sal[np.random.permutation(no_pairs)]
    [mapped,C] = tsne.tsne(diff_coh_res_sal, no_dims, init_dims, perplexity)
    for i in np.arange(2,no_pairs):
        for j in np.arange(no_kmeans):        
            labels2[i,j,:] = kmeans2(mapped,i)[1]
            clust_crit[i,j] = sk.silhouette_score(mapped,labels2[i,j])
    crit_sal[k,:] = np.mean(clust_crit,1)




plt.plot(crit_drug.T,color = 'red',linewidth = 2)
plt.plot(crit_sal.T,color = 'blue',linewidth = 1)
plt.xlabel('Number of clusters',fontsize = 18)
plt.ylabel('Criterion value',fontsize = 18)
plt.title('Silhouette criterion '+ drug_name + ' vs. saline',fontsize = 20)
plt.tick_params(labelsize = 16)
plt.show

f_name = drug_name + '_crit.jpeg'
plt.savefig(f_name, dpi=300, facecolor='w', edgecolor='w',
orientation='portrait', papertype=None, format=None,
transparent=False, bbox_inches=None, pad_inches=0.1,
frameon=None)
    





