## PSI - data analyzing tools
Matlab/Python files for rat PSI data analysis
### List of files:
* **_SNEpy.py_**, **_graph_dynamics.m_**
    * Python and MATLAB codes calculating coherence dynamics and show clusters in topo-space
* **_SNEpy_all_sbj.py_**
    * Calculates t-SNE distibution of all subject coherences together
    * Takes labeled coherence pairs from **'_clust_ident.npy'** file from _SNEpy.py_
    * Shows clusters in corresponding collors\
* **_SNEpyPermutation.py_**
    * Performs permutation test of the mapping procedure based on t-SNE 
* **_topoview.py_**
    *  Short script filtering p-values and relative changes of coherences
    *  Shows significant coherence changes which relative valueven relative change treshhold
* **_combined_graph_dynamics.m_**
    * An experimental file mixing the effects of two drugs