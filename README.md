# python-genomics
A set of files to do genomics analysis in python

To use any of these script collections, just run these two lines in your python kernel / Jupyter notebook:
```
sys.path.append('/home/ubuntu/tools/python-genomics')
import Scanpyplus
```

Among the functions in Scanpyplus, there's also a function to do feature gene selection (DeepTree algorithm). It removes garbage among highly variable genes, mitigate batch effect if you remove garbage batch by batch, and increases signal-to-noise ratio of the top PCs to promote rare cell type discovery.

[Here](https://github.com/brianpenghe/python-genomics/blob/master/DeepTree%20algorithm%20demo.ipynb) is a notebook to use DeepTree algorithm to "de-noise" highly-variable genes. 

If you use DeepTree please cite this: [He et al. 2020 Nature](https://www.nature.com/articles/s41586-020-2536-x) 

![image](https://user-images.githubusercontent.com/4110443/90067837-9b57cc80-dce7-11ea-9d31-8f5e49d07964.png)

A MATLAB implementation can be found [here](https://github.com/brianpenghe/Matlab-genomics)
