import gc
import scrublet as scr
import scipy.io
from scipy import sparse
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
import joblib
from pysankey.sankey import sankey
#not that it used to be called pySankey with uppercase "s"

import numpy as np
import pandas as pd
import scanpy as sc
import anndata
import bbknn
import os
from scipy import sparse
from scipy import cluster
from glob import iglob
from upsetplot import UpSet
from upsetplot import plot
import gzip

def ExtractColor(adata,obsKey='louvain',keytype=int):
#   labels=sorted(adata.obs[obsKey].unique().to_list(),key=keytype)
    labels=adata.obs[obsKey].cat.categories
    colors=adata.uns[obsKey+'_colors']
    return dict(zip(labels,colors))

def UpdateUnsColor(adata,ColorDict,obsKey='louvain'):
    #ColorDict is like {'Secretory 3 C1': '#c0c0c5','Secretory 4 C1': '#87ceee'}
    ColorUns=ExtractColor(adata,obsKey,keytype=str)
    ColorUns.update(ColorDict)
    adata.uns[obsKey+'_colors']=list(ColorUns.values())
    return adata

def GetRaw(adata_all):
    return anndata.AnnData(X=adata_all.raw.X,obs=adata_all.obs,var=adata_all.raw.var,\
obsm=adata_all.obsm)

def CalculateRaw(adata,scaling_factor=10000):
    #The object must contain a log-transformed matrix
    #This function returns an integer-count object
    #The normalization constant is assumed to be 10000
    return anndata.AnnData(X=sparse.csr_matrix(np.rint(np.array(np.expm1(adata.X).todense().transpose())*(adata.obs['n_counts'].values).transpose() / scaling_factor).transpose()),\
                  obs=adata.obs,var=adata.var,obsm=adata.obsm,varm=adata.varm)

def OrthoTranslate(adata,\
oTable='~/refseq/Mouse-Human_orthologs_only.csv'):
    adata.var_names_make_unique(join='-')
    OrthologTable = pd.read_csv(oTable).dropna()
    MouseGenes=OrthologTable.loc[:,'Gene name'].drop_duplicates(keep=False)
    HumanGenes=OrthologTable.loc[:,'Human gene name'].drop_duplicates(keep=False)
    FilteredTable=OrthologTable.loc[((OrthologTable.loc[:,'Gene name'].isin(MouseGenes)) &\
                            (OrthologTable.loc[:,'Human gene name'].isin(HumanGenes))),:]
    bdata=adata[:,adata.var_names.isin(FilteredTable.loc[:,'Gene name'])]
    FilteredTable.set_index('Gene name',inplace=True,drop=False)
    bdata.var_names=FilteredTable.loc[bdata.var_names,'Human gene name']
    return bdata

def remove_barcode_suffix(adata):
    bdata=adata.copy()
    bdata.obs_names=pd.Index([i[0] for i in bdata.obs_names.str.split('-',expand=True)])
    return bdata

def file2gz(file,delete_original=True):
    with open(file,'rb') as src, gzip.open(file+'.gz','wb') as dst:
        dst.writelines(src)
    if delete_original==True:
        os.remove(file)

def Scanpy2MM(adata,prefix='temp'):
    #Scanpy2MM(adata,"./")
    #please make sure the object contains raw counts (using our CalculateRaw function)
    adata.var['feature_types']='Gene Expression'
    scipy.io.mmwrite(prefix+'matrix.mtx',adata.X.transpose(),field='integer')
    if 'gene_ids' not in adata.var.columns.unique():
        adata.var['gene_ids']=adata.var_names
    adata.var[['gene_ids','feature_types']].reset_index().set_index(keys='gene_ids').to_csv(prefix+"features.tsv", \
            sep = "\t", index= True,header=False)
    adata.obs.to_csv(prefix+"barcodes.tsv", sep = "\t", columns=[],header= False)
    adata.obs.to_csv(prefix+"metadata.tsv", sep = "\t", index= True)
    file2gz(prefix+"matrix.mtx")
    file2gz(prefix+"barcodes.tsv")
    ##file2gz(prefix+"metadata.tsv")
    file2gz(prefix+"features.tsv")

def ShiftEmbedding(adata,domain_key='batch',embedding='X_umap',ncols=3,alpha=0.9):
    from sklearn import preprocessing
    scaler = preprocessing.MinMaxScaler()
    adata.obs[embedding+'0']=adata.obsm[embedding][:,0]
    adata.obs[embedding+'1']=adata.obsm[embedding][:,1]
    X=adata.obs[embedding+'0']
    Y=adata.obs[embedding+'1']
    batch_categories=adata.obs[domain_key].unique()
    for i in list(range(len(batch_categories))):
        temp=adata[adata.obs[domain_key]==batch_categories[i]].obsm[embedding]
        scaler.fit(temp)
        X.loc[adata.obs[domain_key]==batch_categories[i]]=(scaler.transform(temp)*alpha+ [int(i/ncols),i%ncols])[:,0]
        Y.loc[adata.obs[domain_key]==batch_categories[i]]=(scaler.transform(temp)*alpha+ [i/ncols,i%ncols])[:,1]
    adata.obsm[embedding]=np.vstack((X.values,Y.values)).T
    del adata.obs[embedding+'0']
    del adata.obs[embedding+'1']
    return adata

def CopyEmbedding(aFrom,aTo,embedding='X_umap'):
    aFrom.obs['temp0']=aFrom.obsm[embedding][:,0]
    aFrom.obs['temp1']=aFrom.obsm[embedding][:,1]
    aTo.obs['temp0']=''
    aTo.obs['temp1']=''
    aTo.obs.loc[aFrom.obs_names,'temp0']=aFrom.obs['temp0']
    aTo.obs.loc[aFrom.obs_names,'temp1']=aFrom.obs['temp1']
    aTo.obsm[embedding]=np.vstack((aTo.obs['temp0'],aTo.obs['temp1'])).T
    del aFrom.obs['temp0']
    del aFrom.obs['temp1']
    del aTo.obs['temp0']
    del aTo.obs['temp1']
    return aTo

def CopyMeta(aFrom,aTo,overwrite=False):
    if overwrite==True:
        obs_items=aFrom.obs.columns
        var_items=aFrom.var.columns
    else:
        obs_items=aFrom.obs.columns[~aFrom.obs.columns.isin(aTo.obs.columns)]
        var_items=aFrom.var.columns[~aFrom.var.columns.isin(aTo.var.columns)]
    aTo.obs[obs_items]=''
    aTo.var[var_items]=''
    aTo.obs.loc[aFrom.obs_names,obs_items]=aFrom.obs.loc[:,obs_items]
    aTo.var.loc[aFrom.var_names,var_items]=aFrom.var.loc[:,var_items]
    return aTo

def celltype_per_stage_plot(adata,celltypekey='louvain',stagekey='batch',plotlabel=True,\
    celltypelist=['default'],stagelist=['default'],celltypekeytype=int,stagekeytype=str,
    fontsize='x-small',yfontsize='x-small',legend_pos=(1,0.5),savefig=None):
    # this is a function for horizonal bar plots
    if 'default' in celltypelist:
        celltypelist = sorted(adata.obs[celltypekey].unique().tolist(),key=celltypekeytype)
    if 'default' in stagelist:
        stagelist = sorted(adata.obs[stagekey].unique().tolist(),key=stagekeytype)
    celltypelist=[i for i in celltypelist if i in adata.obs[celltypekey].unique()]
    stagelist=[i for i in stagelist if i in adata.obs[stagekey].unique()]
    colors=ExtractColor(adata,celltypekey,keytype=str)
    count_array=np.array(pd.crosstab(adata.obs[celltypekey],adata.obs[stagekey]).loc[celltypelist,stagelist])
    count_ratio_array=count_array / np.sum(count_array,axis=0)
    for i in range(len(celltypelist)):
        plt.barh(stagelist[::-1],count_ratio_array[i,::-1],
            left=np.sum(count_ratio_array[0:i,::-1],axis=0),color=colors[celltypelist[i]],label=celltypelist[i])
        plt.yticks(fontsize=yfontsize)
    plt.grid(b=False)
    if plotlabel:
        plt.legend(celltypelist,fontsize=fontsize,bbox_to_anchor=legend_pos)
    if savefig is not None:
        plt.savefig(savefig+'.pdf',bbox_inches='tight')

def stage_per_celltype_plot(adata,celltypekey='louvain',stagekey='batch',plotlabel=True,\
    # this is a function for vertical bar plots
    # please remember to run pl.umap to assign colors
    celltypelist=['default'],stagelist=['default'],celltypekeytype=int,stagekeytype=str,
    fontsize='x-small',xfontsize='x-small',legend_pos=(1,0.5),savefig=None):
    if 'default' in celltypelist:
        celltypelist = sorted(adata.obs[celltypekey].unique().tolist(),key=celltypekeytype)
    if 'default' in stagelist:
        stagelist = sorted(adata.obs[stagekey].unique().tolist(),key=stagekeytype)
    celltypelist=[i for i in celltypelist if i in adata.obs[celltypekey].unique()]
    stagelist=[i for i in stagelist if i in adata.obs[stagekey].unique()]
    colors=ExtractColor(adata,stagekey,keytype=str)
    count_array=np.array(pd.crosstab(adata.obs[celltypekey],adata.obs[stagekey]).loc[celltypelist,stagelist])
    count_ratio_array=count_array.transpose() / np.sum(count_array,axis=1)
    for i in range(len(stagelist)):
        plt.bar(celltypelist,count_ratio_array[i,:],
            bottom=1-np.sum(count_ratio_array[0:i+1,:],axis=0),
           color=colors[stagelist[i]],label=stagelist[i])
        plt.xticks(fontsize=xfontsize)
    plt.grid(b=False)
    plt.legend(stagelist,fontsize=fontsize,bbox_to_anchor=legend_pos)
    plt.xticks(rotation=90)
    if savefig is not None:
        plt.savefig(savefig+'.pdf',bbox_inches='tight')

def mtx2df(mtx,idx,col):
    #mtx is the name/location of the matrix.mtx file
    #idx is the index file (rownames)
    #col is the colnames file
    count = scipy.io.mmread(mtx)
    idxs = [i.strip() for i in open(idx)]
    cols = [i.strip() for i in open(col)]
    sc_count = pd.DataFrame(data=count.toarray(),
                        index=idxs,
                        columns=cols)
    return sc_count 

def DEmarkers(adata,celltype,reference,obs,max_out_group_fraction=0.25,\
use_raw=False,length=100,obslist=['percent_mito','n_counts','batch'],\
min_fold_change=2,min_in_group_fraction=0.25,log=True,method='wilcoxon'):
    celltype=celltype
    sc.tl.rank_genes_groups(adata, obs, groups=[celltype],
                        reference=reference,method=method,log=log)
    sc.tl.filter_rank_genes_groups(adata, groupby=obs,\
                    max_out_group_fraction=max_out_group_fraction,
                    min_fold_change=min_fold_change,use_raw=use_raw,
                    min_in_group_fraction=min_in_group_fraction,log=log)
    GeneList=pd.DataFrame(adata.uns['rank_genes_groups_filtered']['names']).loc[:,celltype].dropna().head(length).transpose().tolist()
    sc.pl.umap(adata,color=GeneList+obslist,
           color_map = 'jet',use_raw=use_raw)
    sc.pl.dotplot(adata,var_names=GeneList,
             groupby=obs,use_raw=use_raw,standard_scale='var')
    sc.pl.stacked_violin(adata[adata.obs[obs].isin([celltype,reference]),:],var_names=GeneList,groupby=obs,
           swap_axes=True)
    return GeneList

def GlobalMarkers(adata,obs,max_out_group_fraction=0.25,min_fold_change=2,\
min_in_group_fraction=0.25,use_raw=False,method='wilcoxon'):
    sc.tl.rank_genes_groups(adata,groupby=obs,n_genes=len(adata.var_names),
                  method=method)
    sc.tl.filter_rank_genes_groups(adata,groupby=obs,
                    max_out_group_fraction=max_out_group_fraction,
                    min_fold_change=min_fold_change,use_raw=use_raw,
                    min_in_group_fraction=min_in_group_fraction)
    Markers=pd.DataFrame(adata.uns['rank_genes_groups_filtered']['names'])
    return Markers.apply(lambda x: pd.Series(x.dropna().values))

def HVGbyBatch(adata,batch_key='batch',min_mean=0.0125, max_mean=3, min_disp=0.5,\
min_clustersize=100,genenames=['default']):
    if 'default' in genenames:
        genenames = adata.var_names
    sc.settings.verbosity=0
    batchlist=adata.obs[batch_key].value_counts()
    for key in batchlist[batchlist>min_clustersize].index:
        adata_sample = adata[adata.obs[batch_key]==key,:][:,genenames]
        print(key)
        sc.pp.highly_variable_genes(adata_sample, min_mean=min_mean, max_mean=max_mean, min_disp=min_disp)
        adata.var['highly_variable'+key]=pd.Series(adata.var_names,\
            index=adata.var_names).isin(adata_sample.var_names[adata_sample.var['highly_variable']])
    sc.settings.verbosity=3
    adata.var['highly_variable_n']=0
    temp=adata.var['highly_variable_n'].astype('int32')
    for key in batchlist[batchlist>min_clustersize].index:
        temp=temp+adata.var['highly_variable'+key].astype('int32')
    adata.var['highly_variable_n']=temp
    return adata

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
        adata_sample=adata_sample.copy()
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
        del adata_sample
    return adata

def Bertie_preclustered(adata,batch_key='batch',cluster_key='louvain'):
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
        adata_sample=adata_sample.copy()

        for clus in np.unique(adata_sample.obs[cluster_key]):
            adata_sample.obs.loc[adata_sample.obs[cluster_key]==clus, 'scrublet_cluster_score'] = \
                np.median(adata_sample.obs.loc[adata_sample.obs[cluster_key]==clus, 'scrublet_score'])

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
        del adata_sample
    return adata


def snsSplitViolin(adata,genelist,celltype='leiden',celltypelist=['0','1']):
    df=sc.get.obs_df(adata[adata.obs[celltype].isin(celltypelist)],genelist+[celltype])
    df = df.set_index(celltype).stack().reset_index()
    df.columns=[celltype,'gene','value']
    sns.violinplot(data=df, x='gene', y='value', hue=celltype,
                split=True, inner="quart", linewidth=1)

def DownSample(MouseC1data,cell_type='leiden',downsampleTo=10):
    NewIndex3=[]
    if ( downsampleTo > 0 ) & (isinstance(downsampleTo, int)):
        for i in MouseC1data.obs[cell_type].sort_values().unique():
            NewIndex3=NewIndex3+random.sample(\
                   population=MouseC1data[MouseC1data.obs[cell_type]==i].obs_names.tolist(),
            k=min(downsampleTo,len(MouseC1data[MouseC1data.obs[cell_type]==i\
                         ].obs_names.tolist())))
    return MouseC1data[NewIndex3]

def snsCluster(MouseC1data,MouseC1ColorDict2={False:'#000000',True:'#00FFFF'},cell_type='louvain',gene_type='highly_variable',\
            cellnames=['default'],genenames=['default'],figsize=(10,7),row_cluster=False,col_cluster=False,\
            robust=True,xticklabels=False,yticklabels=False,method='complete',metric='correlation',cmap='jet',\
            downsampleTo=0):
    if 'default' in cellnames:
        cellnames = MouseC1data.obs_names
        if ( downsampleTo > 0 ) & (isinstance(downsampleTo, int)):
            NewIndex3=[]
            for i in MouseC1data.obs[cell_type].sort_values().unique():
                NewIndex3=NewIndex3+random.sample(\
                          population=MouseC1data[MouseC1data.obs[cell_type]==i\
                                 ].obs_names.tolist(),
                k=min(downsampleTo,len(MouseC1data[MouseC1data.obs[cell_type]==i\
                                 ].obs_names.tolist())))
            cellnames=MouseC1data[NewIndex3].obs_names
    if 'default' in genenames:
        genenames = MouseC1data.var_names
    genenames = [i for i in genenames if i in MouseC1data.var_names]
    cellnames = [i for i in cellnames if i in MouseC1data.obs_names]
    cell_types=cell_type
    gene_types=gene_type
    if type(cell_type) == str:
        cell_types=[cell_type]
    if type(gene_type) == str:
        gene_types=[gene_type]
    louvain_col_colors=[]
    for key in cell_types: 
        MouseC1data_df = MouseC1data[MouseC1data.obs_names][:,genenames].to_df()
        MouseC1data_df[key] = MouseC1data[MouseC1data.obs_names].obs[key]
        MouseC1data_df = MouseC1data_df.sort_values(by=key)
        MouseC1data_df3 = MouseC1data_df.loc[pd.Series(cellnames,index=cellnames).index,:]
        cluster_names=MouseC1data_df3.pop(key)
        louvain_col_colors.append(cluster_names.map(ExtractColor(MouseC1data,obsKey=key,keytype=str)).astype(str))
    adata_for_plotting = MouseC1data_df.loc[cellnames,MouseC1data_df.columns.isin(genenames)]
    adata_for_plotting = adata_for_plotting.reindex(columns=genenames)
    if len(louvain_col_colors) > 1:
        louvain_col_colors=pd.concat(louvain_col_colors,axis=1)
    else:
        louvain_col_colors=louvain_col_colors[0]
    if 'null' in gene_types:
        cg1_0point2=sns.clustermap(adata_for_plotting.transpose(),metric=metric,cmap=cmap,\
                 figsize=figsize,row_cluster=row_cluster,col_cluster=col_cluster,robust=robust,xticklabels=xticklabels,\
                 yticklabels=yticklabels,z_score=0,vmin=-2.5,vmax=2.5,col_colors=louvain_col_colors,method=method)
    else:
        celltype_row_colors=[]
        for key in gene_types:
            genegroup_names=MouseC1data[:,genenames].var[key]
            celltype_row_colors.append(genegroup_names.map(MouseC1ColorDict2).astype(str))
        if len(celltype_row_colors) > 1:
            celltype_row_colors=pd.concat(celltype_row_colors,axis=1)
        else:
            celltype_row_colors=celltype_row_colors[0]
        cg1_0point2=sns.clustermap(adata_for_plotting.transpose(),metric=metric,cmap=cmap,\
                 figsize=figsize,row_cluster=row_cluster,col_cluster=col_cluster,robust=robust,xticklabels=xticklabels,\
                 yticklabels=yticklabels,z_score=0,vmin=-2.5,vmax=2.5,col_colors=louvain_col_colors,row_colors=celltype_row_colors,method=method)

    return cg1_0point2

def markSeaborn(snsObj,genes,clustermap=True):
    if clustermap == True:
        NewIndex=pd.DataFrame(np.asarray([snsObj.data.index[i] for i in snsObj.dendrogram_row.reordered_ind])).iloc[:,0]
        NewIndex2=NewIndex.isin(genes)
        snsObj.ax_heatmap.set_yticks(NewIndex[NewIndex2].index.values.tolist())
        snsObj.ax_heatmap.set_yticklabels(NewIndex[NewIndex2].values.tolist())
    else:
        NewIndex=pd.DataFrame(np.asarray(snsObj.data.index))
        NewIndex2=snsObj.data.index.isin(genes)
        snsObj.ax_heatmap.set_yticks(NewIndex[NewIndex2].index.values)
        snsObj.ax_heatmap.set_yticklabels(NewIndex[NewIndex2].values[:,0])
    #snsObj.fig
    return snsObj.fig

def PseudoBulk(MouseC1data,genenames=['default'],cell_type='louvain',filterout=float,metric='mean'):
    if 'default' in genenames:
        genenames = MouseC1data.var_names
    Main_cell_types = MouseC1data.obs[cell_type].unique()
    Main_cell_types = np.delete(Main_cell_types,\
            [ i for i in range(len(Main_cell_types)) if isinstance(Main_cell_types[i], filterout) ])
    MousePseudoBulk = pd.DataFrame(columns=Main_cell_types,index=genenames)
    print(Main_cell_types)
    for key in Main_cell_types:
        temp=MouseC1data[MouseC1data.obs[cell_type]==key].to_df()
        tempbool=temp.astype(bool)
        temp[cell_type]=key
        tempbool[cell_type]=key
        if metric=='mean':
            temp2 = temp.groupby(by=cell_type).mean()
        elif metric=='median':
            temp2 = temp.groupby(by=cell_type).median()
        elif metric=='fraction':
            temp2 = tempbool.groupby(by=cell_type).sum()/tempbool.groupby(by=cell_type).count()
        del temp
        MousePseudoBulk.loc[:,key]=temp2.loc[key,:].transpose()
        del temp2
    return MousePseudoBulk

def DeepTree(adata,MouseC1ColorDict2,cell_type='louvain',gene_type='highly_variable',\
            cellnames=['default'],genenames=['default'],figsize=(10,7),row_cluster=True,col_cluster=True,\
            method='complete',metric='correlation',Cutoff=0.8,CladeSize=2):
    if 'default' in cellnames:
        cellnames = adata.obs_names
    if 'default' in genenames:
        genenames = adata.var_names
    test=snsCluster(adata,\
                           MouseC1ColorDict2=MouseC1ColorDict2,\
                           genenames=genenames, cellnames=cellnames,\
                           gene_type=gene_type, cell_type=cell_type,method=method,\
                           figsize=figsize,row_cluster=row_cluster,col_cluster=col_cluster,metric=metric)
    cutree = cluster.hierarchy.cut_tree(test.dendrogram_row.linkage,height=Cutoff)
    TreeDict=dict(zip(*np.unique(cutree, return_counts=True)))
    TreeDF=pd.DataFrame(TreeDict,index=[0])
    DeepIndex=[i in TreeDF.loc[:,TreeDF.iloc[0,:] > CladeSize].columns.values for i in cutree]
    bdata=adata[:,test.data.index][cellnames]
    bdata.var['Deep']=DeepIndex
    test1=snsCluster(bdata,\
                           MouseC1ColorDict2=MouseC1ColorDict2,\
                           cellnames=cellnames,gene_type='Deep',cell_type=cell_type,method=method,\
                           figsize=figsize,row_cluster=True,col_cluster=True,metric=metric)
    test2=snsCluster(bdata[:,DeepIndex],\
                           MouseC1ColorDict2=MouseC1ColorDict2,\
                           cellnames=cellnames,gene_type='null',cell_type=cell_type,method=method,\
                           figsize=figsize,row_cluster=True,col_cluster=True,metric=metric)
    return [bdata,test,test1,test2]

def DeepTree_per_batch(adata,batch_key='batch',obslist=['batch'],min_clustersize=100):
    batchlist=adata.obs[batch_key].value_counts()
    for key in batchlist[batchlist>min_clustersize].index:
        print(key)
        bdata=adata[:,adata.var['highly_variable'+key]][adata.obs[batch_key]==key,:]
        sc.pp.filter_genes(bdata,min_cells=3)
        sc.pl.umap(bdata,color=obslist)
        [bdata,test, test1, test2]=DeepTree(bdata,
                        MouseC1ColorDict2={False:'#000000',True:'#00FFFF'},
                        cell_type=obslist,
                        cellnames=adata[adata.obs[batch_key]==key,:].obs_names.tolist(),
                        genenames=adata[:,adata.var['highly_variable'+key]].var_names.tolist(),
                         row_cluster=True,col_cluster=True)
        adata.var['Deep_'+key]=pd.Series(adata.var_names,index=adata.var_names).isin((bdata)[:,bdata.var['Deep']].var_names)
#        sc.pl.umap(adata,color=obslist)
    adata.var['Deep_n']=0
    temp=adata.var['Deep_n'].astype('int32')
    for key in batchlist[batchlist>min_clustersize].index:
        temp=temp+adata.var['Deep_'+key].astype('int32')
    adata.var['Deep_n']=temp
    return adata

def Venn_Upset(adata,genelists,size_height=3):
    #gene lists can be ['Deep_1','Deep_2']
    deepgenes=pd.DataFrame(adata.var[genelists+['highly_variable']])
    deepgenes=deepgenes.set_index(genelists)
    upset = UpSet(deepgenes, subset_size='count', intersection_plot_elements=size_height)
    upset.plot()
    return upset

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
def LogisticRegressionCellType(Reference, Query, Category = 'louvain', DoValidate = False,\
    multi_class='multinomial',n_jobs=-1,max_iter=1000,tol=1e-4,keyword=''):
    #This function doesn't do normalization or scaling
    #The logistic regression function returns the updated Query object with predicted info stored
    Reference.var_names_make_unique()
    Query.var_names_make_unique()
    IntersectGenes = np.intersect1d(Reference.var_names,Query.var_names)
    Reference2 = Reference[:,IntersectGenes]
    Query2 = Query[:,IntersectGenes]
    X = Reference2.X
    y = Reference2.obs[Category].replace(np.nan,'None',regex=True)
    x = Query2.X
    cv = StratifiedKFold(n_splits=5, random_state=None, shuffle=False)
    logit = LogisticRegression(penalty='l2',
                           random_state=42,
                           C=0.2,
                           tol=tol,
                           n_jobs=n_jobs,
                           solver='sag',
                           multi_class=multi_class,
                           max_iter=max_iter,
                           verbose=10
                          )
    result=logit.fit(X, y)
    y_predict=result.predict(x)
    today = date.today()
    if DoValidate is True:
        scores = cross_val_score(logit, X, y, cv=cv)
        print(scores)
    _ = joblib.dump(result,str(today)+'Sklearn.result.'+keyword+'.pkl',compress=9)
    np.savetxt(str(today)+'Sklearn.result.'+keyword+'.csv',IntersectGenes,fmt='%s',delimiter=',')
    Query.obs['Predicted'] = y_predict
    return Query

def LogisticPrediction(adata,model_pkl,genelistcsv):
    #This function imports saved logistic model and gene list csv to predict cell types for adata
    #adata has better been scaled if you trained a model using scaled AnnData
    CT_genes=pd.read_csv(genelistcsv,header=None)
    CT_genes['idx'] = CT_genes.index
    CT_genes.columns = ['symbol', 'idx']
    CT_genes = np.array(CT_genes['symbol'])

    lr = joblib.load(open(model_pkl,'rb'))
    lr.features = CT_genes
    features = adata.var_names
    k_x = features.isin(list(CT_genes))
    print(f'{k_x.sum()} features used for prediction', file=sys.stderr)
    k_x_idx = np.where(k_x)[0]
    temp=adata
    X = temp.X[:, k_x_idx]
    features = features[k_x]
    ad_ft = pd.DataFrame(features.values, columns=['ad_features']).reset_index().rename(columns={'index': 'ad_idx'})
    lr_ft = pd.DataFrame(lr.features, columns=['lr_features']).reset_index().rename(columns={'index': 'lr_idx'})
    lr_idx = lr_ft.merge(ad_ft, left_on='lr_features', right_on='ad_features').sort_values(by='ad_idx').lr_idx.values

    lr.n_features_in_ = lr_idx.size
    lr.features = lr.features[lr_idx]
    lr.coef_ = lr.coef_[:, lr_idx]
    predicted_hi = lr.predict(X)
    adata.obs['predicted_hi'] = predicted_hi

    return adata

def DouCLing(adata,hi_type,lo_type,rm_genes=[],print_marker_genes=False):
    DoubletScores=pd.DataFrame(0,index=adata.obs['new_celltype'].unique(),
                          columns=['Parent1','Parent2','Parent1_count','Parent2_count','All_count','p_value'])
    #Dou(blet)C(luster)L(abe)ling method
    #hi_type='leiden_R' is the key for high-resolution cell types that main include doublet clusters
    #lo_type='leiden' is the key for low-resolution cell types that represent compartments
    #this function aims to identify cross-compartment doublet clusters. Same-compartment doublet clusters 
    # look more like transitional cell types closer to homotypic doublets which are difficult to catch
    #rm_genes=['TYMS','MKI67'] is a list of genes you don't want to include, such as cell-cycle genes
    import time
    import scipy.stats as ss
    start_time = time.time()
    alpha=adata.obs[lo_type].value_counts()
    for j in adata.obs[lo_type].sort_values().unique():
        temp=adata[adata.obs[lo_type]==j][:,~adata.var_names.isin(rm_genes)]
        if len(temp.obs[hi_type].value_counts())==1:
            continue
        sc.tl.rank_genes_groups(temp,groupby=hi_type,n_genes=50)
        Markers=pd.DataFrame(temp.uns['rank_genes_groups']['names'])
        for i in pd.DataFrame(temp.uns['rank_genes_groups']['names']).columns:
            scoring_genes=Markers.loc[~(Markers.loc[:,i].isin(rm_genes)),i]
            if print_marker_genes:
                print(scoring_genes)
            if len(scoring_genes)<20:
                continue
            sc.tl.score_genes(adata,gene_list=scoring_genes,score_name=i+'_score')
            DoubletScores.loc[i,'Parent1']=adata[adata.obs[hi_type]==i\
                        ].obs[lo_type].value_counts().index[0]
            cutoff=adata[adata.obs.new_celltype==i].obs[i+'_score'].quantile(q=0.75)

            beta=adata.obs.loc[adata.obs.loc[:,i+'_score'\
                    ]>cutoff,lo_type].value_counts()
            DoubletScores.loc[i,'Parent2']=beta.index[0]
#        if DoubletScores.loc[i,'Parent1']==DoubletScores.loc[i,'Parent2']:
#            pass
#        else:
            DoubletScores.loc[i,'Parent2_count']=beta[0]
            DoubletScores.loc[i,'Parent1_count']=beta.loc[DoubletScores.loc[i,'Parent1']]
            DoubletScores.loc[i,'All_count']=beta.sum()
            hpd=ss.hypergeom(alpha.sum()-alpha.loc[DoubletScores.loc[i,'Parent1']],
                         alpha.loc[DoubletScores.loc[i,'Parent2']],
                        beta.sum()-beta.loc[DoubletScores.loc[i,'Parent1']])
            DoubletScores.loc[i,'p_value']=hpd.pmf(DoubletScores.loc[i,'Parent2_count'])
    print("--- %s seconds ---" % (time.time() - start_time))
    DoubletScores.loc[:,'Is_doublet_cluster']=(DoubletScores.loc[:,'Parent2_count'] / DoubletScores.loc[:,'All_count'] > 0.6) & \
~(DoubletScores.loc[:,'Parent1'] == DoubletScores.loc[:,'Parent2'])
    return DoubletScores
