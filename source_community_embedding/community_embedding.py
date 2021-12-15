#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 13 11:50:27 2021

@author: camila
"""

import tensorflow as tf
#tf.enable_eager_execution()

#%% imports 

from pathlib import Path
import sys
path = str(Path(Path(__file__).parent.absolute()).parent.absolute())
sys.path.insert(0, path)
sys.path.insert(0, path + '/source_community_detection/')
sys.path.insert(0, path + '/source_community_embedding/')
sys.path.insert(0, path + '/source_graph_util/')
sys.path.insert(0, path + '/source_node_classification/')


import MOSpecG_partitioning_OV as SpecG
import SpecOV_partitioning_v2 as SpecOV
import spectral_functions as sf
import metrics as metrics

import sys
import numpy as np

import multiprocessing
import queue

import time

from sklearn.metrics.pairwise import cosine_similarity,cosine_distances
from sklearn.decomposition import NMF

import igraph

from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score

from sklearn.metrics import pairwise_distances
from scipy.spatial.distance import cosine

from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score

from numpy import linalg
    
import os

import pandas as pd

import shutil

import networkx as nx


import matplotlib.pyplot as plt
from math import isclose
from sklearn.decomposition import PCA
import os
import networkx as nx
import numpy as np
import pandas as pd
#from stellargraph import StellarGraph, datasets
#from stellargraph.data import EdgeSplitter
from collections import Counter
import multiprocessing
from IPython.display import display, HTML
from sklearn.model_selection import train_test_split

from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler


#import all the dependencies
from keras.layers import Dense,Conv2D,MaxPooling2D,UpSampling2D
from keras import Input, Model
from keras.datasets import mnist
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers, losses

import matplotlib.cm as cm
import subprocess

#very_low_num = 10E-15
very_low_num = 10E-10



    
#%% Classes
#def main_wang():

class SpecOV_params:  
    num_pareto=11
    
    bool_ov=True
    ov_thr=0.5
    
    p_leading=0.3   
    it_spec=2 #10
    

class SpecG_params:    
    p_leading=0.3    
    ensemble_thr=0.5
    #TODO:
    
    bool_verbose=False
    out_verbose=[]    
    
    MO_bool=True

    n_gen=1 #10
    n_pop=5  #3 #ERROR!!
    p_offs=0.6
  
#%% Scores

    
    
#%% Node classification

#%%% 
#glin = graph.g.linegraph()
 

#%% Embedding spectral pure

#%%% Spectral dec 

class Eig:
    val = None
    vec = None
    pos_ind = None
    neg_ind = None  

def get_embedding(embedding_vec, u):
    return embedding_vec[u]

#%% plot embedding

def plot_embedding_all(g, H, ri_vert_pos, Rp, T, dim_p, p_expected, k_expected,it_spec,it_emb, MO, IT, p_leading, file_graph, file_expected, results_folder, file_com_out, time_folder, pareto_file, num_pareto, ensemble_thr, n_gen, n_pop, p_offs, bool_ov, ov_thr=0,op_com=1,dim_emb=2, char_split=' ',name='', op_var='pdimemb',  lambdav = 1e9,  alpha=0.5, beta=5,bool_link_pred=False,bool_proposed=True,gamma1=0.4,gamma2=0.3,gamma3=0.3):
    H = H[:, np.where( np.all(H[..., :] == 0, axis=0) == False )[0]]
    p_best = sf.create_part_from_Sov(H).astype(int)
    
    file=results_folder+"plot_"
    plot_embedding(ri_vert_pos, Rp, p_best, file+"ri_vert_pos.png")
    #plot_embedding(Rp, file+"Rp")        
    
    if 1==0:
        adj = g.get_adjacency(attribute="weight")
        adj_mat = np.matrix(adj.data) 
        plot_embedding(adj_mat,None,p_best, file+"A")
        model = NMF(n_components=2, init='nndsvdar') #, init='random') #b, init='random' random_state=0,
        M = model.fit_transform(adj_mat) #M
        UA = np.transpose(model.components_) #U
        plot_embedding(UA,None,p_best, file+"UA")
        #adj_mat = adj_mat/ np.max(adj_mat)
     
        #plot_embedding(adj_mat,Rp,file+"S")
        
        #-- Cosine
        S1 = np.array(adj.data)    
        S2 = cosine_similarity(S1)
        S=S2
        plot_embedding(S,None,p_best, file+"S.png")
        plot_embedding(S2,None,p_best, file+"S2.png")
        
    
        
        model = NMF(n_components=2, init='nndsvdar') #, init='random') #b, init='random' random_state=0,
        M = model.fit_transform(S) #M
        US = np.transpose(model.components_) #U
        plot_embedding(US,None,p_best, file+"US.png")
        
        TS = np.dot(T, T.transpose()) 
        plot_embedding(TS,None,p_best, file+"TS.png")
        model = NMF(n_components=2, init='nndsvdar') #, init='random') #b, init='random' random_state=0,
        M = model.fit_transform(TS) #M
        UT = np.transpose(model.components_) #U
        plot_embedding(UT,None,p_best, file+"UT.png")

        gamma1=0.1
        gammaS=0.7
        gammaT=0.2
        
        AF = np.add( np.add( gamma1* adj_mat, gammaT*TS), gammaS*S2 )
        plot_embedding(AF,None,p_best, file+"AF_"+str(gamma1)+"_"+str(gammaS)+"_"+str(gammaT)+".png")
            
        #original rp
        gamma1=1
        gamma2=0
        gamma3=0
        q_best_orig, Sov_best_orig, g_orig, eig_val_orig, ri_vert_pos_orig, ri_vert_neg_orig, Rp_orig, Rn_orig, mod_mat_orig,A_orig,P_orig = SpecOV.spec_ov_main(file_graph, it_spec, p_leading, gamma1, gamma2, gamma3, ov_thr, bool_ov, file_com_out, bool_verbose, out_verbose,KNumber_pre=k_expected,mod_orig=False,T=T)       
        ri_vert_pos_orig = ri_vert_pos_orig[:,0:dim_p ]
        Rp_orig = Rp_orig[:,0:dim_p ]
        Sov_best_orig = Sov_best_orig[:, np.where( np.all(Sov_best_orig[..., :] == 0, axis=0) == False )[0]]
        p_best_orig = sf.create_part_from_Sov(Sov_best_orig)
     
        plot_embedding(ri_vert_pos_orig, Rp_orig, p_best_orig, file+"ri_vert_pos_orig.png")
        f1_macro_spec, f1_micro_spec = metrics(ri_vert_pos_orig, p_expected)
        #S 
        gamma1=0
        gamma2=1
        gamma3=0
        _, H_p, _, _, ri_vert_pos_p, _, Rp_p, _, _,_,_ = SpecOV.spec_ov_main(file_graph, it_spec, p_leading, gamma1, gamma2, gamma3,ov_thr, bool_ov, file_com_out, bool_verbose, out_verbose,KNumber_pre=k_expected,mod_orig=False,T=T)       
        H_p = H_p[:, np.where( np.all(H_p[..., :] == 0, axis=0) == False )[0]]
        p_p = sf.create_part_from_Sov(H_p)
        plot_embedding(ri_vert_pos_p,Rp_p,p_p, file+"S_ri.png")
        f1_macro_spec, f1_micro_spec =  metrics.node_classification(ri_vert_pos_p, p_expected)
        
        #T
        gamma1=0
        gamma2=0
        gamma3=1
        _, H_p, _, _, ri_vert_pos_p, _, Rp_p, _, _,_,_ = SpecOV.spec_ov_main(file_graph, it_spec, p_leading, gamma1, gamma2, gamma3,ov_thr, bool_ov, file_com_out, bool_verbose, out_verbose,KNumber_pre=k_expected,mod_orig=False,T=T)       
        H_p = H_p[:, np.where( np.all(H_p[..., :] == 0, axis=0) == False )[0]]
        p_p = sf.create_part_from_Sov(H_p)
        plot_embedding(ri_vert_pos_p,Rp_p,p_p, file+"T_ri.png")
        f1_macro_spec, f1_micro_spec = metrics.node_classification(ri_vert_pos_p, p_expected)
        
 
    
    
    
def plot_embedding(vec, Rp, p_best, file):
    print(file)
    #print(Rp)
    
    vec = vec/np.max(vec)
    if Rp is not None:
        Rp = Rp/np.max(Rp)
    fig=plt.figure(figsize=(8, 8))
    ax=fig.add_axes([0,0,1,1])
    
    colors = cm.rainbow(np.linspace(0, 1, len(np.unique(p_best))))
    
    for i in np.arange( p_best.shape[0] ):
        k=int(p_best[i])
        #print(k)
        ax.scatter(vec[i,0], vec[i,1], color=colors[k], marker='.',s=100)
        if Rp is not None:
            ax.scatter(Rp[k,0], Rp[k,1], color=colors[k], marker="s",s=200)
    #ax.scatter(vec[1,0], vec[1,1], color='r')
    #color according to community
    #plt.show()   
    if Rp is not None:
        ax.scatter(0,0, color='black', marker='D',s=100)
    
    ax=fig.add_axes([0,0,1,1])
    plt.xticks(fontsize = 15)
    plt.yticks(fontsize = 15)
        
    ax.grid()
    
    plt.savefig(file)
    plt.show()    
    
