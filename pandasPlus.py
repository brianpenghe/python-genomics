import gc
#import scrublet as scr
import scipy.io
#import scvelo as scv
import matplotlib.pyplot as plt
import networkx as nx
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
#not that it used to be called pySankey with uppercase "s"
import scipy.stats
import numpy as np
import pandas as pd
#import scanpy.api as sc
import anndata
#import bbknn
import os
from scipy import sparse
from scipy import cluster
from glob import iglob

def show_graph_with_labels(adjacency_matrix):
    rows, cols = np.where(adjacency_matrix.values >= 0.9)
    edges = zip(rows.tolist(), cols.tolist())
    gr = nx.Graph()
    gr.add_edges_from(edges)
    nx.draw(gr, node_size=100, 
            labels={i : adjacency_matrix.index.values.tolist()[i] for i in range(0, len(adjacency_matrix.index.values) ) }, 
            with_labels=True)
    plt.show()

def DF2Ann(DF):
    #This function converts a dataframe to AnnData
    #Make sure to transpose if needed
    return(anndata.AnnData(DF))

def UpSetFromLists(listOflist,labels,size_height=3,showplot=True):
    from upsetplot import UpSet
    listall=list(set([j for i in listOflist for j in i]))
    temp=pd.Series(listall,index=listall)
    temp2=pd.concat([temp.isin(i) for i in listOflist+[temp]],axis=1)
    temp2.columns=labels+['all']
    temp2=temp2.set_index(labels)
    upset = UpSet(temp2,subset_size='count', intersection_plot_elements=3)
    if showplot is True:
        upset.plot()
    return upset

def zscore(DF,dropna=True,axis=1):
    output=pd.DataFrame(scipy.stats.zscore(DF.values,axis=1),
             index=DF.index,
            columns=DF.columns)
    if dropna==True:
        return output.dropna(axis=1-axis)
    return output

def Ginni(beta):
    #beta is a DataFrame with columns of distributions
    beta.loc['Ginni']=[np.abs(np.subtract.outer(beta.loc[:,i].values,\
               beta.loc[:,i].values)).mean()/np.mean(beta.loc[:,i].values)*0.5 for i in beta.columns]
    return beta

def cellphonedb_p2adjMat(cpdb_p_loc='pvalues.txt',pval=0.05):
    #this function reads the p_value file from cellphonedb output and generates a matrix file 
    #with significant pairs only
    cpdb_p=pd.read_csv(cpdb_p_loc,sep='\t')
    cellpairs=pd.Series(cpdb_p.iloc[0,11:].index).str.split('|',expand=True)
    cell0=cellpairs.loc[:,0].unique()
    cell1=cellpairs.loc[:,1].unique()
    InterMat=pd.DataFrame('',index=cell0,columns=cell1)
    for i in cell0:
        for j in cell1:
            InterMat.loc[i,j]= '|'.join(cpdb_p.loc[cpdb_p.loc[:,i+'|'+j]<pval,
                                               'interacting_pair'].tolist())
    InterMat.to_csv(cpdb_p_loc+'.pairs'+str(pval)+'.csv')
    return InterMat

def cellphonedb_n_interaction_Mat(cpdb_p_loc='pvalues.txt',pval=0.05):
    #this function reads the p_value file from cellphonedb output and generates a matrix file 
    #with significant pairs only
    cpdb_p=pd.read_csv(cpdb_p_loc,sep='\t')
    cellpairs=pd.Series(cpdb_p.iloc[0,11:].index).str.split('|',expand=True)
    cell0=cellpairs.loc[:,0].unique()
    cell1=cellpairs.loc[:,1].unique()
    InterMat=pd.DataFrame(0,index=cell0,columns=cell1)
    for i in cell0:
        for j in cell1:
            InterMat.loc[i,j]= len(cpdb_p.loc[cpdb_p.loc[:,i+'|'+j]<pval,
                                               'interacting_pair'])
    InterMat.to_csv(cpdb_p_loc+'.n_pairs'+str(pval)+'.csv')
    return InterMat

def cellphonedb_mat_per_interaction(interacting_pair,cpdb_p_loc='pvalues.txt'):
    #this function returns a matrix of celltype-celltype interaction p-values for a specific interaction-pair
    cpdb_p=pd.read_csv(cpdb_p_loc,sep='\t')
    cellpairs=pd.Series(cpdb_p.iloc[0,11:].index).str.split('|',expand=True)
    cell0=cellpairs.loc[:,0].unique()
    cell1=cellpairs.loc[:,1].unique()
    InterMat=pd.DataFrame(0,index=cell0,columns=cell1)
    for i in cell0:
        for j in cell1:
            InterMat.loc[i,j]= cpdb_p.loc[cpdb_p.loc[:,'interacting_pair'].isin([interacting_pair]),i+'|'+j].values
    InterMat.to_csv(cpdb_p_loc+'.pairs'+interacting_pair+'.csv')
    return InterMat

def ListsOverlap(A,B):
    #This function is written by ChatGPT to explore the overlapping element counts between each list in A and each list in B
    #A and B are both lists of lists
    counts = []
    for a_list in A:
        row = []
        for b_list in B:
            shared = len(set(a_list).intersection(b_list))
            row.append(shared)
        counts.append(row)
    df = pd.DataFrame(counts, columns=["B" + str(i+1) for i in range(len(B))], index=["A" + str(i+1) for i in range(len(A))])
    return df
