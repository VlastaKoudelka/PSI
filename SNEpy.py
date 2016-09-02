# -*- coding: utf-8 -*-
"""
Created on Wed Aug 10 13:59:44 2016

@author: vavra
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.cm as cm
import scipy.io as sio
import sklearn.metrics as sk
from sklearn.cluster import KMeans
import tsne
plt.close('all')

perplexity = 5
init_dims = 18
no_dims = 2
no_clstr = 4
no_pairs = 36

drug_name = 'PSI'

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

#randomly permute the pair coherences
r_perm = np.random.permutation(no_pairs)
diff_coh_res = diff_coh_res[r_perm]
index = index[r_perm]
x_cord = x_cord[r_perm]
y_cord = y_cord[r_perm]

#t-SNE on dataset
[mapped,C] = tsne.tsne(diff_coh_res, no_dims, init_dims, perplexity)

#Clusterring
k_means_obj = KMeans(no_clstr)
k_means_obj.fit(mapped)
labels = k_means_obj.labels_

#Silhoulette criterion
no_kmeans = 10
labels2 = np.zeros([no_pairs,no_kmeans,no_pairs])
clust_crit = np.zeros([no_pairs,no_kmeans])
for i in np.arange(2,len(labels)):
    for j in np.arange(no_kmeans):        
        k_means_obj = KMeans(i)
        k_means_obj.fit(mapped)        
        labels2[i,j,:] = k_means_obj.labels_
        clust_crit[i,j] = sk.silhouette_score(mapped,labels2[i,j])

best_kmeans = labels2[no_clstr,np.argmax(clust_crit[no_clstr,:]),:]
best_kmeans = best_kmeans.astype(int)           #takes the best kmeans run

clust_crit = np.mean(clust_crit,1)

#re-order difference matrix

for i in np.arange(no_clstr):
    cluster = diff_coh_res[best_kmeans == i,:]
    if (i == 0):
        diff_mat_sort = cluster
    else:        
        diff_mat_sort = np.r_[diff_mat_sort,cluster]
        
     
#---Show the coherence clusters
colors = cm.jet(np.linspace(0, 1, no_clstr))

plt.figure(0)
plt.scatter(mapped[:,0],mapped[:,1],c = colors[best_kmeans,:], s = 150,alpha = 0.7)
plt.tick_params(labeltop=False, labelbottom=False, bottom=True, top=True
                    ,left=True,right=True, labelright=False,labelleft=False)
plt.title(drug_name + ' coherence clusters after t-SNE', fontsize = 20)
plt.ylabel('Feature 2',fontsize = 18)
plt.xlabel('Feature 1',fontsize = 18)  

f_name = drug_name + '_clusters.jpeg'
plt.savefig(f_name, dpi=300, facecolor='w', edgecolor='w',
orientation='portrait', papertype=None, format=None,
transparent=False, bbox_inches=None, pad_inches=0.1,
frameon=None)              

#---Show electrode numbers
for i,coord in enumerate(mapped):
    plt.text(coord[0],coord[1],str(names[0][index[i]][0][0]),fontsize=12)
    plt.show
    
f_name = drug_name + '_clusters.jpeg'
plt.savefig(f_name, dpi=300, facecolor='w', edgecolor='w',
orientation='portrait', papertype=None, format=None,
transparent=False, bbox_inches=None, pad_inches=0.1,
frameon=None)  

#---Show brain
plt.figure(1)
plt.imshow(brain)
plt.title(drug_name + ' clusters in topographic view', fontsize = 20)    

for i in np.arange(len(x_cord)):
    plt.plot(x_cord[i],y_cord[i], c = colors[best_kmeans[i]],linewidth = 3,alpha = 0.65,solid_capstyle='round') 
    plt.tick_params(labeltop=False, labelbottom=False, bottom=False, top=False
                    ,left=False,right=False, labelright=False,labelleft=False)
    plt.show
    
f_name = drug_name + '_topo.jpeg'
plt.savefig(f_name, dpi=300, facecolor='w', edgecolor='w',
orientation='portrait', papertype=None, format=None,
transparent=False, bbox_inches=None, pad_inches=0.1,
frameon=None)

#---Show Silhoulette
plt.figure(2)
plt.plot(clust_crit,lw = 1,color = 'blue')
plt.ylabel('Silhouette criterion',fontsize = 18)
plt.xlabel('Number of clusters',fontsize = 18)
plt.title('Silhouette criterion ' + drug_name + ' vs. saline',fontsize = 20)
plt.tick_params(labelsize = 16)
plt.show


#---Show difference matrix
plt.figure(3)
plt.title(drug_name + ' difference matrix sorted',fontsize = 20)
plt.imshow(diff_mat_sort)
plt.colorbar()
plt.tick_params(labelsize = 16)
plt.xlabel('Time differences within bands',fontsize = 18)
plt.ylabel('Electrode pairs',fontsize = 18)

x = -2
y = 0
for i in np.arange(no_clstr):
    y = y + len(best_kmeans[best_kmeans == i])/2
    plt.scatter(x,y,c = colors[i], s = 150)
    y = y + len(best_kmeans[best_kmeans == i])/2
        
#plt.text(0.5,36.5,r'$\delta$',fontsize=18)
#plt.text(3.5,36.5,r'$\theta$',fontsize=18)
#plt.text(6.5,36.5,r'$\alpha$',fontsize=18)
#plt.text(9.5,36.5,r'$\beta_1$',fontsize=18)
#plt.text(12.5,36.5,r'$\beta_2$',fontsize=18)
#plt.text(15.5,36.5,r'$\gamma$',fontsize=18)

plt.xticks(np.linspace(1.5,15.5,6),(r'$\delta$',r'$\theta$',r'$\alpha$',r'$\beta_{Lo}$',r'$\beta_{Hi}$',r'$\gamma$') )

f_name = drug_name + '_difmat.jpeg'
plt.savefig(f_name, dpi=300, facecolor='w', edgecolor='w',
orientation='portrait', papertype=None, format=None,
transparent=False, bbox_inches=None, pad_inches=0.1,
frameon=None)

identify = np.c_[best_kmeans,index]
np.save(drug_name+'_clust_ident.npy',identify)

#plt.show

#---Show coherence matrix
#plt.figure(4)
#plt.title('coherence matrix')
#plt.imshow(coh_res)
#plt.colorbar()
#plt.text(0.5,36.5,r'$\delta$',fontsize=25)
#plt.text(4.5,36.5,r'$\theta$',fontsize=25)
#plt.text(8.5,36.5,r'$\alpha$',fontsize=25)
#plt.text(12.5,36.5,r'$\beta_1$',fontsize=25)
#plt.text(16.5,36.5,r'$\beta_2$',fontsize=25)
#plt.text(20.5,36.5,r'$\gamma$',fontsize=25)
#plt.tick_params(labeltop=False, labelbottom=False, bottom=False, top=False
#                    ,left=True,right=True, labelright=False,labelleft=True)