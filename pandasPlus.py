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
from sklearn.externals import joblib
from pysankey.sankey import sankey
#not that it used to be called pySankey with uppercase "s"

import numpy as np
import pandas as pd
#import scanpy.api as sc
#import anndata
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
