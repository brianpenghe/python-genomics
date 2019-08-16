import gc
import scrublet as scr
import scipy.io
import scvelo as scv
import matplotlib.pyplot as plt
from matplotlib import rcParams
import seaborn as sns
import numpy as np
import random
import sys
import jp_single_raw as jp
from sklearn.manifold import TSNE
from sklearn.preprocessing import scale
from sklearn.decomposition import TruncatedSVD
from sklearn.cluster import SpectralClustering
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.externals import joblib
from pysankey.sankey import sankey
#not that it used to be called pySankey with uppercase "s"

import numpy as np
import pandas as pd
import scanpy.api as sc
import anndata
import bbknn
import os
from scipy import sparse

def Bertie(adata,Resln=1,batch_key='batch'):
    scorenames = ['scrublet_score','scrublet_cluster_score','bh_pval']
    def bh(pvalues):
        '''
        Computes the Benjamini-Hochberg FDR correction.

        Input:
            * pvals - vector of p-values to correct
        '''
        n = int(pvalues.shape[0])
        new_pvalues = np.empty(n)
        values = [ (pvalue, i) for i, pvalue in enumerate(pvalues) ]
        values.sort()
        values.reverse()
        new_values = []
        for i, vals in enumerate(values):
            rank = n - i
            pvalue, index = vals
            new_values.append((n/rank) * pvalue)
        for i in range(0, int(n)-1):
            if new_values[i] < new_values[i+1]:
                new_values[i+1] = new_values[i]
        for i, vals in enumerate(values):
            pvalue, index = vals
            new_pvalues[index] = new_values[i]
        return new_pvalues

    for i in np.unique(adata.obs[batch_key]):
        adata_sample = adata[adata.obs[batch_key]==i,:]
        scrub = scr.Scrublet(adata_sample.X)
        doublet_scores, predicted_doublets = scrub.scrub_doublets(verbose=False)
        adata_sample.obs['scrublet_score'] = doublet_scores

        sc.pp.filter_genes(adata_sample, min_cells=3)
        sc.pp.normalize_per_cell(adata_sample, counts_per_cell_after=1e4)
        sc.pp.log1p(adata_sample)
        sc.pp.highly_variable_genes(adata_sample, min_mean=0.0125, max_mean=3, min_disp=0.5)
        adata_sample = adata_sample[:, adata_sample.var['highly_variable']]
        sc.pp.scale(adata_sample, max_value=10)
        sc.tl.pca(adata_sample, svd_solver='arpack')
        adata_sample = adata_sample.copy()
        # del adata_sample.obsm['X_diffmap']
        sc.pp.neighbors(adata_sample)
        #eoverclustering proper - do basic clustering first, then cluster each cluster
        sc.tl.louvain(adata_sample)
        for clus in np.unique(adata_sample.obs['louvain']):
            sc.tl.louvain(adata_sample, restrict_to=('louvain',[clus]),resolution=Resln)
            adata_sample.obs['louvain'] = adata_sample.obs['louvain_R']
        #compute the cluster scores - the median of Scrublet scores per overclustered cluster
        for clus in np.unique(adata_sample.obs['louvain']):
            adata_sample.obs.loc[adata_sample.obs['louvain']==clus, 'scrublet_cluster_score'] = \
                np.median(adata_sample.obs.loc[adata_sample.obs['louvain']==clus, 'scrublet_score'])
        #now compute doublet p-values. figure out the median and mad (from above-median values) for the distribution
        med = np.median(adata_sample.obs['scrublet_cluster_score'])
        mask = adata_sample.obs['scrublet_cluster_score']>med
        mad = np.median(adata_sample.obs['scrublet_cluster_score'][mask]-med)
        #let's do a one-sided test. the Bertie write-up does not address this but it makes sense
        pvals = 1-scipy.stats.norm.cdf(adata_sample.obs['scrublet_cluster_score'], loc=med, scale=1.4826*mad)
        adata_sample.obs['bh_pval'] = bh(pvals)
        #create results data frame for single sample and copy stuff over from the adata object
        scrublet_sample = pd.DataFrame(0, index=adata_sample.obs_names, columns=scorenames)
        for meta in scorenames:
            scrublet_sample[meta] = adata_sample.obs[meta]
        #write out complete sample scores
        #scrublet_sample.to_csv('scrublet-scores/'+i+'.csv')

        scrub.plot_histogram();
        #plt.savefig('limb/sample_'+i+'_doulet_histogram.pdf')
        adata.obs.loc[adata.obs[batch_key]==i,'doublet_scores']=doublet_scores
        adata.obs.loc[adata.obs[batch_key]==i,'bh_pval'] = bh(pvals)
    return adata
