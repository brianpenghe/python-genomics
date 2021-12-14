# python-genomics
A set of files to do genomics analysis in python

To use any of these script collections, just run these two lines in your python kernel / Jupyter notebook:
```
sys.path.append('/home/ubuntu/tools/python-genomics')
import Scanpyplus
```

## DeepTree feature selection
Among the functions in Scanpyplus, there's also a function to do feature gene selection (DeepTree algorithm). It removes garbage among highly variable genes, mitigate batch effect if you remove garbage batch by batch, and increases signal-to-noise ratio of the top PCs to promote rare cell type discovery.

[Here](https://nbviewer.jupyter.org/github/brianpenghe/python-genomics/blob/master/DeepTree_algorithm_demo.ipynb) is a [notebook](https://github.com/brianpenghe/python-genomics/blob/master/DeepTree_algorithm_demo.ipynb) to use DeepTree algorithm to "de-noise" highly-variable genes and improve initial clustering. 

A MATLAB implementation can be found [here](https://github.com/brianpenghe/Matlab-genomics).

This algorithm can be potentially used to reduce batch effect when fearing overcorrection, especially comparing conditions or time points. Two notebooks are provided showing "soft integration" of [fetal limb](https://nbviewer.jupyter.org/github/brianpenghe/python-genomics/blob/master/Soft_integration_limb.ipynb) and [pancreas](https://nbviewer.jupyter.org/github/brianpenghe/python-genomics/blob/master/Soft_integration_pancreas.ipynb) data.

## Doublet Cluster Labeling (DouCLing)
There are 4 types of doublets:

![image](https://user-images.githubusercontent.com/4110443/146040113-1c1b27e6-453e-48fa-a4e8-786ff8c759ec.png)

Cross-sample doublets can usually be identified by hastags or genetic backgrounds. Theoretically, (n-1)/n of intersample doublets can be identified when n samples with different hashtags/genetics are pooled equally.

Heterotypic doublets can sometimes trick data scientists into thinking they are a new type of dual-feature cell type like NKT cells etc. 
Heterotypic doublets are usually identified by matching individual cells to synthetic doublets regardless of manually curated clusters. 
To leverage the input from biologists' manual parsing and the increased sensitivity of cluster-average signatures, I introduce here an alternative approach to facilitate heterotypic doublet cluster identification. This approach scans through individual tiny clusters and look for its "Parent 2" that gives it a unique feature that's different from its sibling subclusters sharing the same "Parent 1". 
[A notebook using published PBMC data](https://nbviewer.jupyter.org/github/brianpenghe/python-genomics/blob/master/DOUblet_Cluster_Labeling.ipynb) is provided.

