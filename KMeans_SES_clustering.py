#!/usr/bin/env python
# coding: utf-8

import util
import copy

import pandas as pd
import numpy as np

from sklearn.decomposition import PCA
from sklearn.preprocessing import scale
from sklearn.preprocessing import minmax_scale
from sklearn.cluster import MiniBatchKMeans

from scipy.special import rel_entr


# Privacy requirement, min_population and min_tracts in a cluster
POPULATION = 20000
TRACTS = 2

# Selected features for clustering
features = ['fraction_assisted_income','fraction_high_school_edu',
                'median_income','fraction_no_health_ins',
                'fraction_poverty','fraction_vacant_housing']

# Populaiton and censut tract id column names
# These two columns are required for clustering
pop_col = 'POPULATION'
tract_col = 'census_tract_fips'

input_csv = 'data/Tract_clustering_dataset_2017.csv'
output_csv = 'data/2017_clustering_out.csv'


def run_kmeans(_input_csv=input_csv, _output_csv=output_csv, _features=features,_pop_col = pop_col,_tract_col=tract_col,
               n_clstrs=60000, max_iter=200, n_init=5, ratio=0.01, random_state=None):
    """
    Compute the lower bound of number of clusters
    
    Parameters
    ----------
    _input_csv : list
        Input census tract level SES data
    _output_csv : int
        Clustered SES data with selected features and deprivation index
    _features : list
        the list of feature columns selected for clustering
    _pop_col : string
        The colnum name of POPULATION column   
    _tract_col : string
        The colnum name of censut tract id column
    n_clstrs : int
        Number of clusters for K-Means, we suggest set it to 0.8 * number of tracts. This is NOT the final number of clusters.
        The final number of clusters 
    ... : 
        parameters for K-Means, see sklearn.cluster.MiniBatchKMeans for more details.
        
    Return
    ------
    kldivs : list
        The list of KL-divergence scores of each features between clustered data and raw tract-level data. 
        This is a measure of the data utility (similarity) with the following elements:
        [feature_1_div,...feature_n_div,dep_index_div,sum_div]
    """    
    
    df = pd.read_csv(_input_csv)
    X = scale(df[_features].values)
    clr = MiniBatchKMeans(n_clusters=n_clstrs,random_state=random_state,
                          n_init=n_init,max_iter=max_iter,reassignment_ratio=ratio)
    clr.fit(X)
    labels = clr.labels_
    labels = util.privacy_refinement(df,X,labels,POPULATION,TRACTS,n_clstrs,_pop_col,_tract_col,show_progress=False)
    util.save_result(df, labels, _features,_pop_col, _tract_col, _output_csv)
    kldivs = util.kl_divergence(_input_csv,_output_csv,_features,_tract_col)
    return kldivs

kl_divs = run_kmeans()