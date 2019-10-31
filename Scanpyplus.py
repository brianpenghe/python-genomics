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
from scipy import cluster

def Bertie(adata,Resln=1,batch_key='batch'):
    scorenames = ['scrublet_score','scrublet_cluster_score','bh_pval']
    adata.obs['doublet_scores']=0
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


def snsCluster(MouseC1data,MouseC1ColorDict,MouseC1ColorDict2,cell_type='louvain',gene_type='highly_variable',\
            cellnames=['default'],genenames=['default'],figsize=(10,7),row_cluster=False,col_cluster=False,\
            robust=True,xticklabels=False,method='complete',metric='correlation'):
    if 'default' in cellnames:
        cellnames = MouseC1data.obs_names
    if 'default' in genenames:
        genenames = MouseC1data.var_names

    MouseC1data_df = MouseC1data[MouseC1data.obs_names,:].to_df()
    MouseC1data_df['louvain'] = MouseC1data[MouseC1data.obs_names,:].obs[cell_type]
    MouseC1data_df = MouseC1data_df.sort_values(by='louvain')

    MouseC1data_df3 = MouseC1data_df.loc[pd.Series(cellnames,index=cellnames).index,:]
    cluster_names=MouseC1data_df3.pop('louvain')
    louvain_col_colors=cluster_names.map(MouseC1ColorDict)
    adata_for_plotting = MouseC1data_df.loc[cellnames,MouseC1data_df.columns.isin(genenames)]
    adata_for_plotting = adata_for_plotting.reindex(columns=genenames)
    if gene_type == 'null':
        cg1_0point2=sns.clustermap(adata_for_plotting.transpose(),metric="correlation",cmap='RdYlBu_r',\
                 figsize=figsize,row_cluster=row_cluster,col_cluster=col_cluster,robust=robust,xticklabels=xticklabels,\
                 z_score=0,vmin=-2.5,vmax=2.5,col_colors=louvain_col_colors,method=method)
    else:
        genegroup_names=MouseC1data[:,MouseC1data.var_names].var[gene_type]
        celltype_row_colors=genegroup_names.map(MouseC1ColorDict2)
        cg1_0point2=sns.clustermap(adata_for_plotting.transpose(),metric="correlation",cmap='RdYlBu_r',\
                 figsize=figsize,row_cluster=row_cluster,col_cluster=col_cluster,robust=robust,xticklabels=xticklabels,\
                 z_score=0,vmin=-2.5,vmax=2.5,col_colors=louvain_col_colors,row_colors=celltype_row_colors,method=method)

    return cg1_0point2

def PseudoBulk(MouseC1data,genenames=['default'],cell_type='louvain',filterout=float):
    if 'default' in genenames:
        genenames = MouseC1data.var_names
    Main_cell_types = MouseC1data.obs[cell_type].unique()
    Main_cell_types = np.delete(Main_cell_types,\
            [ i for i in range(len(Main_cell_types)) if isinstance(Main_cell_types[i], float) ])
    MousePseudoBulk = pd.DataFrame(columns=(Main_cell_types,index=genenames)
    print(Main_cell_types)
    for key in Main_cell_types:
        temp=MouseC1data[MouseC1data.obs[cell_type]==key,:].to_df()
        temp[cell_type]=key
        temp2 = temp.groupby(by=cell_type).mean()
        del temp
        MousePseudoBulk.loc[:,key]=temp2.loc[key,:].transpose()
        del temp2
    return MousePseudoBulk

def DeepTree(adata,MouseC1ColorDict,MouseC1ColorDict2,cell_type='louvain',gene_type='highly_variable',\
            cellnames=['default'],genenames=['default'],figsize=(10,7),row_cluster=False,col_cluster=False,\
            method='complete',metric='correlation',Cutoff=0.8,CladeSize=2):
    test=snsCluster(adata,MouseC1ColorDict=MouseC1ColorDict,\
                           MouseC1ColorDict2=MouseC1ColorDict2,\
                           genenames=genenames, cellnames=cellnames,\
                           gene_type=gene_type, cell_type=cell_type,method=method,\
                           figsize=figsize,row_cluster=row_cluster,col_cluster=col_cluster)
    cutree = cluster.hierarchy.cut_tree(test.dendrogram_row.linkage,height=Cutoff)
    TreeDict=dict(zip(*np.unique(cutree, return_counts=True)))
    TreeDF=pd.DataFrame(TreeDict,index=[0])
    DeepIndex=[i in TreeDF.loc[:,TreeDF.iloc[0,:] > CladeSize].columns.values for i in cutree]
    bdata=adata[:,genenames]
    bdata.var['Deep']=DeepIndex
    test=snsCluster(bdata,MouseC1ColorDict=MouseC1ColorDict,\
                           MouseC1ColorDict2=MouseC1ColorDict2,\
                           cellnames=cellnames,gene_type='Deep',cell_type=cell_type,method=method,\
                           figsize=figsize,row_cluster=True,col_cluster=True)
    test=snsCluster(bdata,MouseC1ColorDict=MouseC1ColorDict,\
                           MouseC1ColorDict2=MouseC1ColorDict2,\
                           cellnames=cellnames,gene_type='null',cell_type=cell_type,method=method,\
                           genenames=genenames[np.array(DeepIndex)],\
                           figsize=figsize,row_cluster=True,col_cluster=True)
    return bdata

def DeepTree2(adata,method='complete',metric='correlation',cellnames=['default'],genenames=['default'],\
               Cutoff=0.8,CladeSize=2):
    if 'default' in cellnames:
        cellnames = adata.obs_names
    if 'default' in genenames:
        genenames = adata.var_names
    adata_df=adata[cellnames,:][:,genenames].to_df()
    testscipy=scipy.cluster.hierarchy.fclusterdata(adata_df.transpose(),\
                    metric=metric,method=method,t=Cutoff,criterion="distance")
    TreeDict=dict(zip(*np.unique(testscipy, return_counts=True)))
    TreeDF=pd.DataFrame(TreeDict,index=[0])
    DeepIndex=[i in TreeDF.loc[:,TreeDF.iloc[0,:] > CladeSize].columns.values for i in testscipy]
    bdata=adata[cellnames,:][:,genenames]
    bdata.var['Deep']=DeepIndex
    return bdata


from datetime import date
def LogisticRegressionCellType(Reference, Query, Category = 'louvain', DoValidate = False):
    #This function doesn't do normalization or scaling
    #The logistic regression function returns the updated Query object with predicted info stored
    IntersectGenes = np.intersect1d(Reference.var_names,Query.var_names)
    Reference2 = Reference[:,IntersectGenes]
    Query2 = Query[:,IntersectGenes]
    X = Reference2.X
    y = Reference2.obs[Category]
    x = Query2.X
    cv = StratifiedKFold(n_splits=5, random_state=None, shuffle=False)
    logit = LogisticRegression(penalty='l2',
                           random_state=42,
                           C=0.2,
                           solver='sag',
                           multi_class='multinomial',
                           max_iter=200,
                           verbose=10
                          )
    result=logit.fit(X, y)
    y_predict=result.predict(x)
    today = date.today()
    if DoValidate is True:
        scores = cross_val_score(logit, X, y, cv=cv)

    _ = joblib.dump(result,str(today)+'Sklearn.result.joblib.pkl',compress=9)

    Query.obs['Predicted'] = y_predict
    return Query
