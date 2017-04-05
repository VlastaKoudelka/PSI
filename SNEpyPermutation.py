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
from sklearn.metrics import silhouette_score as crit
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA as pca
import tsne
import time


 
#plt.close('all')

perplexity = 5
init_dims = 18
no_dims = 2
no_pairs = 36
no_perm = 50
no_map = 1      #a number of t-SNE runs in one permutation
t = time.time()

drug_name = 'psilocin'

#path = ('/home/vlastimilo/NUDZ_Data/Filip_PSI/MATSoubory/')

path = ('D:\Filip_PSI_mysi\Coherence_StatisticFinalTables\MATSoubory\\') 

#Import matlab files
drug_mat = sio.loadmat(path + drug_name + '.mat')   #open the .mat
                        
                        
coord_mat = sio.loadmat('Locations.mat')
brain = mpimg.imread('SRC/brain.png')

x_cord = np.array(coord_mat['x_pair_sel'])       #x_pair_all for all pairs
y_cord = np.array(coord_mat['y_pair_sel'])
index = np.array(coord_mat['index'][0])     #indices of used electrode pairs

drug_str = np.array(drug_mat[drug_name][0])            #opent the structure
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
    print('Permutation number:' + str(i))
    C_min = 1e10
    dif_perm = diff_coh_res[np.random.permutation(no_pairs)]
    coh_perm = diff_coh_res
#    for j in np.arange(no_map):
#        [mapped,C] = tsne.tsne(dif_perm, no_dims, init_dims, perplexity)
#        if C < C_min:
#            win_map = mapped    #a map with the lowest error
#            C_min = C

    #PCA - uncomment for use instead of t-SNE
    pca_obj = pca(n_components=2)
    win_map = pca_obj.fit_transform(diff_coh_res)

    for k in np.arange(2,no_pairs):
        kmeans_obj = KMeans(k)
        kmeans_obj.fit(win_map)
        labels = kmeans_obj.labels_        
        silhouette[i,k] = crit(win_map,labels)
 
#plt.figure   
#plt.plot(np.mean(silhouette.T,1),color = 'red')
#plt.show

plt.figure
plt.plot(silhouette.T,color = 'red')
plt.title('saline vs. psilocin silhouetts over ' + str(no_perm) + ' permutations PCA')
plt.xlabel('Number of clusters')
plt.ylabel('Criterion value')
plt.show

f_name = 'EXPORT\\' + drug_name + '_perm_test.jpeg'
plt.savefig(f_name, dpi=300, facecolor='w', edgecolor='w',
orientation='portrait', papertype=None, format=None,
transparent=False, bbox_inches=None, pad_inches=0.1,
frameon=None)
np.save(drug_name + '_perm_test.npy',silhouette)
print(time.time()-t)


    



        
            
        
