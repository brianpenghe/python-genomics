# python-genomics
A set of files to do genomics analysis in python

To use any of these script collections, just run these two lines in your python kernel / Jupyter notebook:
```
sys.path.append('/home/ubuntu/tools/python-genomics')
import Scanpyplus
```

Among the functions in Scanpyplus, there's also a function to do feature gene selection (DeepTree algorithm). It removes garbage among highly variable genes, mitigate batch effect if you remove garbage batch by batch, and increases signal-to-noise ratio of the top PCs to promote rare cell type discovery.

[Here](https://github.com/brianpenghe/python-genomics/blob/master/DeepTree_algorithm_demo.ipynb) is a notebook to use DeepTree algorithm to "de-noise" highly-variable genes. 

A MATLAB implementation can be found [here](https://github.com/brianpenghe/Matlab-genomics).
