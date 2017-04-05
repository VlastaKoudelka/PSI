# -*- coding: utf-8 -*-
"""
Created on Wed Aug 10 13:59:44 2016

@author: vavra
"""
from scipy.cluster.vq import kmeans2
import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
import matplotlib.cm as cm
import tsne
plt.close('all')

perplexity =  50
init_dims = 18
no_dims = 2
no_clstr = 4
no_el = 36
subjects = [0,1,2,3,4,5,6,7]

drug_name = 'psilocin'
#path = ('/home/vlastimilo/NUDZ_Data/Filip_PSI/MATSoubory/')

path = ('D:\Filip_PSI_mysi\Coherence_StatisticFinalTables\MATSoubory\\') 

#Import matlab file
identify = np.load(drug_name + '_clust_ident.npy')   #load cluster identification 
drug_mat = sio.loadmat(path + drug_name + '.mat')    #open the .mat
drug_str = drug_mat[drug_name]            #opent the structure
coord_mat = sio.loadmat('Locations.mat')
index = np.array(coord_mat['index'][0])     #indices of used electrode pairs
drug_str = drug_str[0]                #first structure
data = drug_str['data']
data = data[0]
names = drug_str['namesOfElecs']
no_sbj = data.shape[0]

data = data[:,:,identify[:,1]]                      #select the used pairs
data = data[subjects]
no_sbj = data.shape[0]
ident = identify

for i in np.arange(no_sbj - 1):
    ident = np.r_[ident,identify]


coh = np.zeros([4,6,36,no_sbj])    #band,time,pair
 

#take apart the bands and time intervals
for i,val in enumerate(np.arange(0,56,14)):
    for j in np.arange(0,no_sbj,1):        
        coh[i,:,:,j] = data[j,val:val+6,:]

#Diferentiate the data in time
diff_coh = np.diff(coh,1,0)
diff_coh_res = np.reshape(diff_coh,[18,36*no_sbj],'F').T 
diff_coh_res = diff_coh_res/np.amax(diff_coh_res)

coh_res = np.reshape(coh,[24,36*no_sbj]).T 
     
     
#t-SNE on dataset
[mapped,C] = tsne.tsne(diff_coh_res, no_dims, init_dims, perplexity)

#labels = kmeans2(mapped,no_clust,10)
colors = cm.jet(np.linspace(0, 1, no_clstr))

#create subject labels

sub_labels = np.ones((no_sbj,no_el))
for i in np.arange(no_sbj):
    sub_labels[i] = sub_labels[i]*i
    
sub_labels = np.reshape(sub_labels,[no_el*no_sbj,1],'C').astype(int)    


plt.figure(0)    
plt.scatter(mapped[:,0],mapped[:,1],c = colors[ident[:,0],:], s = 150,alpha = 0.7)
plt.title(drug_name + ' coherence clusters for all subjects', fontsize = 20)
plt.ylabel('Feature 2',fontsize = 18)
plt.xlabel('Feature 1',fontsize = 18)
plt.tick_params(labeltop=False, labelbottom=False, bottom=True, top=True
                    ,left=True,right=True, labelright=False,labelleft=False)
for i,coord in enumerate(mapped):
    plt.text(coord[0],coord[1],sub_labels[i][0],fontsize=12)
                
plt.show

f_name = 'EXPORT\\' + drug_name + '_subject_clusters.jpeg'
plt.savefig(f_name, dpi=300, facecolor='w', edgecolor='w',
orientation='portrait', papertype=None, format=None,
transparent=False, bbox_inches=None, pad_inches=0.1,
frameon=None)
#np.save(drug_name + '_perm_test.npy',silhouette)


#for i in np.arange(no_sbj):
#    plt.figure(i)
#    plt.matshow(diff_coh_res[i*36:i*36 + 36,:])    
#    plt.colorbar()
#    plt.text(0.5,36.5,r'$\delta$',fontsize=25)
#    plt.text(3.5,36.5,r'$\theta$',fontsize=25)
#    plt.text(6.5,36.5,r'$\alpha$',fontsize=25)
#    plt.text(9.5,36.5,r'$\beta_1$',fontsize=25)
#    plt.text(12.5,36.5,r'$\beta_2$',fontsize=25)
#    plt.text(15.5,36.5,r'$\gamma$',fontsize=25)
#    plt.show