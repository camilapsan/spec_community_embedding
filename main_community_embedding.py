#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 13 11:50:27 2021

@author: camila
"""

#import tensorflow as tf
#tf.enable_eager_execution()

#%% imports 

import subprocess
import matplotlib.cm as cm
import os
import sys
import numpy as np
from numpy import linalg
import multiprocessing
import queue
import time
import pandas as pd
import shutil
import igraph
import networkx as nx


from pathlib import Path
import sys
#path = str(Path(Path(__file__).parent.absolute()).parent.absolute())
#sys.path.insert(0, path)
#sys.path.insert(0, path + '/source_community_detection/')
#sys.path.insert(0, path + '/source_community_embedding/')
#sys.path.insert(0, path + '/source_graph_util/')

path = str(Path(Path(__file__).absolute()).absolute())
sys.path.insert(0, path)
sys.path.insert(0, 'source_community_detection/')
sys.path.insert(0, 'source_community_embedding/')
sys.path.insert(0, 'source_graph_util/')


import MOSpecG_partitioning_OV as SpecG
import SpecOV_partitioning_v2 as SpecOV
import spectral_functions as sf
import community_embedding as ComEmb
import spectral_based as Spectral
import SpecNMF as SpecNMF
import SpecRp as SpecRp


import matplotlib.pyplot as plt
from math import isclose
from sklearn.decomposition import PCA

from sklearn.metrics.pairwise import cosine_similarity,cosine_distances
from sklearn.decomposition import NMF

from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score

from sklearn.metrics import pairwise_distances
from scipy.spatial.distance import cosine

from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score


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

from keras.layers import Dense,Conv2D,MaxPooling2D,UpSampling2D
from keras import Input, Model
from keras.datasets import mnist
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers, losses


#very_low_num = 10E-15
very_low_num = 10E-10     
                  

# %%get params 

def get_params_from_op_alg(op_alg):
    op_var=''
    if op_alg=='autorp_SpecDecOV':
        TEST_DISJ_OV=False
        op_emb="AutoEnc"
        #op_var='pset'
        op_var='pdimemb'     
        op_com="SpecDecOV"            
    elif op_alg=='autorpNorm_SpecDecOV':
        TEST_DISJ_OV=False
        op_emb="AutoEncNorm"
        #op_var='pset'
        op_var='pdimemb'     
        op_com="SpecDecOV"     
    elif op_alg=='autorpNorm2_SpecDecOV':
        TEST_DISJ_OV=False
        op_emb="AutoEncNorm2"
        #op_var='pset'
        op_var='pdimemb'     
        op_com="SpecDecOV" 
    elif op_alg=='autorpNorm3_SpecDecOV':
        TEST_DISJ_OV=False
        op_emb="AutoEncNorm3"
        #op_var='pset'
        op_var='pdimemb'     
        op_com="SpecDecOV"  
    elif op_alg=='autorp_SpecG':
        TEST_DISJ_OV=False
        op_emb="AutoEnc"
        #op_var='pset'
        op_var='pdimemb'     
        op_com="SpecG"            
    #---- NMF
    elif op_alg == 'wang-SpecDecOV' :
        #TEST WANG
        TEST_DISJ_OV=True #uses overlapping for both tests
        op_emb="NMF"
        op_com="SpecDecOV"
        #op_var='pdimemb'       
        #op_var='pset'
        op_var='pdimemb'     
    elif op_alg == 'wangOv-SpecDecOV' :
        #TEST WANG
        TEST_DISJ_OV=True #uses overlapping for both tests
        op_emb="NMF"
        op_com="SpecDecOV"
        #op_var='pdimemb'       
        #op_var='pset'
        op_var='H-pdimemb'                     
    elif op_alg == 'wang-SpecG':
        #TEST WANG
        TEST_DISJ_OV=True #uses overlapping for both tests
        op_emb="NMF"
        op_com="SpecG"
        op_var='pdimemb'        
    elif op_alg == 'wangauto-SpecDecOV':
        TEST_DISJ_OV=True #uses overlapping for both tests, instead of H uses autoencoder of rp
        op_emb="NMF"
        op_com="SpecDecOV-Rp"
        op_var='pdimemb'
    elif op_alg == 'wangauto-SpecG':
        TEST_DISJ_OV=True #uses overlapping for both tests, instead of H uses autoencoder of rp
        op_emb="NMF"
        op_com="SpecG-Rp"
        op_var='pdimemb'

        
    #params: TODO: variar
    if 'abl' in op_var:
        print("TODO: params")
        lambdav_list=[0.1, 1, 100, 1000, 1e9]
        alpha_list = [0.1, 0.5, 1, 5, 10]         
        beta_list=np.arange(0,11)
    else: 
        lambdav_list = [1e9]        
        alpha_list=[0.5]
        beta_list=[5]
        #lambdav_list = [1000],    #alpha_list=[50],     #beta_list=[0.05]    
    
    print("op_com=", op_com)
    return TEST_DISJ_OV, op_emb, op_com, op_var, lambdav_list, alpha_list, beta_list

#%%% Call - Datasets overlapping

def call_community_embedding_overlapping(op_emb,dim_emb=128,op_com=2, char_split=' ',op_var='pdimemb',  lambdav = 1e9 ,  alpha=0.5, beta=5, nex=1, opDeltaH=0):
    IT_WANG=5
    p_leading=0.5
   
    name_list = ['facebook/0','facebook/107', 'facebook/1684','facebook/1912','facebook/3437', 'facebook/348', 'facebook/3980','facebook/414','facebook/686','facebook/698','amazon5','youtube5']
    #name_list = ['cora/cora'] #, 'citeseer/citeseer']
    
    char_split_dict={}
    char_split_dict['facebook/0']=' '
    char_split_dict['facebook/107']=' '
    char_split_dict['facebook/1684']=' '
    
    results_folder="../results_Tese_2021_4/"
    time_folder="../results_Tese_2021_4/"
     
    #params: TODO: variar
    lambdav = 1e9
    alpha=0.5
    beta=5
                   
    IT=10 #10    
    
    #SpecG
    num_pareto=11
    n_gen=10 #10
    n_pop=10  #3 #ERROR!!
    p_offs=0.3 #0.6          
    ensemble_thr=0.5
    
    #SpecOV
    bool_ov=True
            
    # gamma1=0.4
    # gamma2=0.3
    # gamma3=0.3
    if nex==1:
        gamma_list1=np.array([0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1])
        gamma_list2=np.array([0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1])
    else:
        #gamma_list1=np.array([0.4])
        #gamma_list2=np.array([0.3])
    
        gamma_list1=np.array([0.1])
        gamma_list2=np.array([0.7])

            
    MO_bool=True
    for gamma1 in gamma_list1:
        for gamma2 in gamma_list2[gamma_list2 <= (1-gamma1)]:
            gamma3 = 1 - gamma1 - gamma2
            print("Gamma1=",gamma1,", gamma2=", gamma2, ", gamma3=",gamma3)
                    
            out_results = results_folder + "out_datasets_OV_emb-"+str(op_emb)+'_dim-'+str(dim_emb)+'_com-'+str(op_com)+'_gen-'+ str(n_gen)+'_p-'+str(p_leading)+ '_lb-' + str(lambdav) + '_a-'+str(alpha) + '_b-'+ str(beta) + '_'+ op_var+'_gamma_'+str(gamma1)+'_'+str(gamma2)+'_'+str(gamma3)+'_deltaH='+str(opDeltaH)+'.csv'
            print(out_results)    
        
            df_results = pd.DataFrame(columns=['name', 'n','m', 'mod_spec', 'mod_NMF', 'nmi_spec', 'nmi_NMF', 'f1_macro_spec', 'f1_macro_NMF', 'f1_micro_spec',  'f1_micro_NMF', 'time_spec', 'time_tot'])
        
            lin=0    
            name = name_list[0]
            for name in name_list:

                mod0_avg=0
                mod_avg=0
                nmi0_avg=0
                nmi_avg=0
                f1_macro_spec_avg=0
                f1_macro_U_avg=0
                f1_micro_spec_avg=0
                f1_micro_U_avg=0
                time0_avg=0         
                time_tot_avg=0
                
                for ex in np.arange(0,nex):
                    print("\n###########################\n", name, ", ex=", ex)

                    file_expected = '../datasets_overlapping/'+name+'.circles'
                    file_graph = '../datasets_overlapping/'+name+'.edges'
                
                    pareto_file=""
                          
                    base_folder_out = "../community_detection/datasets/"        
                    file_com_out = base_folder_out+name+".com"
            
                    
                    os.makedirs(file_com_out, exist_ok=True)
                    
                    g = sf.read_construct_g(file_graph)
                    n=g.vcount()
                    m=g.ecount()
                    print("n=",n,'m=',m)
                    
                    if 'pdimemb' in op_var:
                        p_leading = dim_emb / g.vcount() ##TEST!!
                        print("-- # P p_leading = ", p_leading)    
               
                    start_time = time.perf_counter()
                    print(file_graph)
                    
                    time0=0
                    mod0=0
                    nmi0=0
                    
                    if op_emb == 1:
                        H, U, M, S, C, mod0, mod, nmi0, nmi, f1_macro_spec, f1_micro_spec, f1_macro_U, f1_micro_U, time0 = SpecNMF.wang_main(IT_WANG, MO_bool, IT, p_leading, file_graph, file_expected, results_folder, file_com_out,time_folder, pareto_file, num_pareto, ensemble_thr, n_gen, n_pop, p_offs, bool_ov,op_com=op_com,dim_emb=dim_emb, char_split=char_split)
                    elif op_emb == 2:
                         mod, nmi,  f1_macro_spec, f1_micro_spec = Spectral.spectral_emb_main(MO_bool, IT, p_leading, file_graph, file_expected, results_folder, file_com_out, time_folder, pareto_file, num_pareto, ensemble_thr, n_gen, n_pop, p_offs, bool_ov, op_com=op_com, char_split=char_split,name=name)    
                         f1_macro_U=-1
                         f1_micro_U=-1
                    elif op_emb == "NMF":
                        print("Wang+Rp")
                        H, U, M, S, C, mod0, mod, nmi0, nmi, f1_macro_spec, f1_micro_spec, f1_macro_U, f1_micro_U, time0 = SpecNMF.wang_Rp_main(IT_WANG, MO_bool, IT, p_leading, file_graph, file_expected, results_folder, file_com_out,time_folder, pareto_file, num_pareto, ensemble_thr, n_gen, n_pop, p_offs, bool_ov,op_com=op_com,dim_emb=dim_emb, char_split=char_split, name=name,op_var=op_var,lambdav=lambdav, alpha=alpha, beta=beta)
                    elif "AutoEnc" in op_emb:
                        H, U, M, S, C, mod0, mod, nmi0, nmi, f1_macro_spec, f1_micro_spec, f1_macro_U, f1_micro_U, time0 = SpecRp.emb_Rp_Attr_main(IT_WANG, MO_bool, IT, p_leading, file_graph, file_expected, results_folder, file_com_out,time_folder, pareto_file, num_pareto, ensemble_thr, n_gen, n_pop, p_offs, bool_ov,op_com=op_com,dim_emb=dim_emb, char_split=char_split, name=name,op_var=op_var,lambdav=lambdav, alpha=alpha, beta=beta,gamma1=gamma1,gamma2=gamma2,gamma3=gamma3, opDeltaH=opDeltaH)
                
               
                    print(name)             
                    
            
                    time_tot = (time.perf_counter() - start_time)
                    print("--- %s seconds ---" , time_tot)
                
                    mod0_avg+=mod0
                    mod_avg+=mod
                    nmi0_avg+=nmi0
                    nmi_avg+=nmi
                    f1_macro_spec_avg += f1_macro_spec
                    f1_macro_U_avg += f1_macro_U
                    f1_micro_spec_avg += f1_micro_spec
                    f1_micro_U_avg += f1_micro_U
                    time0_avg += time0            
                    time_tot_avg+=time_tot
                   
                mod0_avg=mod0_avg/nex
                mod_avg=mod_avg/nex
                nmi0_avg=nmi0_avg/nex
                nmi_avg=nmi_avg/nex
                f1_macro_spec_avg = f1_macro_spec_avg/nex
                f1_macro_U_avg = f1_macro_U_avg/nex
                f1_micro_spec_avg = f1_micro_spec_avg/nex
                f1_micro_U_avg = f1_micro_U_avg/nex
                time0_avg = time0_avg/nex
                time_tot_avg = time_tot_avg/nex
                    
                df_results.loc[lin] = [name, n,m, round(mod0_avg,4), round(mod_avg,4), round(nmi0_avg,4), round(nmi_avg,4), round(100*f1_macro_spec_avg,2), round(100*f1_macro_U_avg,2), round(100*f1_micro_spec_avg,2), round(100* f1_micro_U_avg,2), round(time0_avg,2), round(time_tot_avg,2)]
                lin+=1
                      
                
                df_results.to_csv(out_results,sep='\t', index=False, header=True) #ok
                
            print(df_results)
        
    return df_results

#%%% Call - Datasets node class

def call_community_embedding_datasets(op_emb,dim_emb=2, it_emb=5,op_com=2, op_var='pdimemb', lambdav = 1e9, alpha=0.5, beta=5, nex=1):
    p_leading=0.5 #pleading 
    IT_WANG=5
    
    #SpecOV
    if TEST_DISJ_OV == True:
        bool_ov=True
    else:
        bool_ov=False
    
    name_list = ['WebKB/webkb/cornell','WebKB/webkb/texas', 'WebKB/webkb/washington', 'WebKB/webkb/wisconsin','cora/cora', 'citeseer/citeseer']
    #name_list = ['cora/cora'] #, 'citeseer/citeseer']
    
    char_split_dict={}
    char_split_dict['WebKB/webkb/cornell']='\t'
    char_split_dict['WebKB/webkb/texas']='\t'
    char_split_dict['WebKB/webkb/washington']='\t'
    char_split_dict['WebKB/webkb/wisconsin']='\t'
    char_split_dict['cora/cora']='\t'
    char_split_dict['cora/cora']='\t'
    char_split_dict['citeseer/citeseer']='\t'
    
    #name_list = ['WebKB/webkb/cornell'] #, 'WebKB/webkb/texas']
    #name_list = ['cora/cora']
    
    results_folder="../results_Tese_2021_4/"
    time_folder="../results_tese_3021_4/"

    IT=10 #10    
    
    #SpecG
    num_pareto=11
    n_gen=10 #10
    n_pop=5  #3 #ERROR!!
    p_offs=0.6          
    ensemble_thr=0.5
    
    ov_thr=0.5
    bool_verbose=False
    out_verbose=[]    
    
    # gamma1=0.4
    # gamma2=0.3
    # gamma3=0.3
    
    if nex==1:
        gamma_list1=np.array([0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1])
        gamma_list2=np.array([0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1])
        
        gamma_list1=np.array([0.1])
        gamma_list2=np.array([0.9])
        
    else:
        #gamma_list1=np.array([0.4])
        #gamma_list2=np.array([0.3])
    
        gamma_list1=np.array([0.1])
        gamma_list2=np.array([0.7])
   
    MO_bool=True
    for gamma1 in gamma_list1:        
        for gamma2 in gamma_list2[gamma_list2 <= (1-gamma1)]:           
            print(gamma2)
            gamma3 = np.round( 1 - gamma1 - gamma2,1)
            print("Gamma1=",gamma1,", gamma2=", gamma2, ", gamma3=",gamma3)
                    
            #out_results = results_folder + "out_datasets_OV_emb-"+str(op_emb)+'_dim-'+str(dim_emb)+'_com-'+str(op_com)+'_gen'+ str(n_gen)+'_p-'+str(p_leading)+'_'+op_var+'.csv'                
            out_results = results_folder + "out_datasets_DISJ_emb-"+str(op_emb)+'_dim-'+str(dim_emb)+'_com-'+str(op_com)+'_gen-'+ str(n_gen)+'_p-'+str(p_leading)+ '_lb-' + str(lambdav) + '_a-'+str(alpha) + '_b-'+ str(beta) + '_'+ op_var+'_gamma_'+str(gamma1)+'_'+str(gamma2)+'_'+str(gamma3)+'.csv'
            print(out_results)    
                
            df_results = pd.DataFrame(columns=['name', 'n','m', 'mod_spec', 'mod_NMF', 'nmi_spec', 'nmi_NMF', 'f1_macro_spec', 'f1_macro_NMF', 'f1_micro_spec',  'f1_micro_NMF', 'time_spec', 'time_tot'])
            
            lin=0    
            name = name_list[0]
            for name in name_list:
                mod0_avg=0
                mod_avg=0
                nmi0_avg=0
                nmi_avg=0
                f1_macro_spec_avg=0
                f1_macro_U_avg=0
                f1_micro_spec_avg=0
                f1_micro_U_avg=0
                time0_avg=0         
                time_tot_avg=0
                
                for ex in np.arange(0,nex):
                    print("\n###########################\n", name, ", ex=", ex)
                    
                    #name = 'WebKB/webkb/cornell'
                    file_expected = '../datasets_classification/'+name+'_expected.txt'
                    file_graph = '../datasets_classification/'+name+'.net'
                
                    pareto_file=""
                    #file_expected="karate_expected.txt"
                          
                    base_folder_out = "../community_detection/datasets/"        
                    file_com_out = base_folder_out+name+".com"
                    
                    char_split = char_split_dict[name]
                    
                    os.makedirs(file_com_out, exist_ok=True)
             
                    #start_time = time.time()
                    start_time = time.perf_counter()
                    print(file_graph)
                    
                    g = sf.read_construct_g(file_graph)
                    n=g.vcount()
                    m=g.ecount()
                    print("n=",n,'m=',m)
                    
                    if 'pdimemb' in op_var:            
                        p_leading = dim_emb / g.vcount() 
                        print("-- # P p_leading = ", p_leading)    
                    
                    time0=0
                    mod0=-1
                    nmi0=-1
                    
                    ##LINK PREDICTION  290921
                    ##link_pred_main(IT_WANG, MO_bool, IT, p_leading, file_graph, file_expected, results_folder, file_com_out,time_folder, pareto_file, num_pareto, ensemble_thr, n_gen, n_pop, p_offs, bool_ov,op_com=op_com,dim_emb=dim_emb, char_split=char_split, name=name,op_var=op_var,lambdav=lambdav, alpha=alpha, beta=beta)
                
                    if op_emb == 1:
                        H, U, M, S, C, mod0, mod, nmi0, nmi, f1_macro_spec, f1_micro_spec, f1_macro_U, f1_micro_U, time0 = SpecNMF.wang_main(IT_WANG, MO_bool, IT, p_leading, file_graph, file_expected, results_folder, file_com_out,time_folder, pareto_file, num_pareto, ensemble_thr, n_gen, n_pop, p_offs, bool_ov,op_com=op_com,dim_emb=dim_emb, char_split=char_split)
                    elif op_emb == 2:
                         mod, nmi,  f1_macro_spec, f1_micro_spec = Spectral.spectral_emb_main(MO_bool, IT, p_leading, file_graph, file_expected, results_folder, file_com_out, time_folder, pareto_file, num_pareto, ensemble_thr, n_gen, n_pop, p_offs, bool_ov, op_com=op_com, char_split=char_split)    
                         f1_macro_U=-1
                         f1_micro_U=-1
                    if op_emb == "NMF":
                        H, U, M, S, C, mod0, mod, nmi0, nmi, f1_macro_spec, f1_micro_spec, f1_macro_U, f1_micro_U, time0 = SpecNMF.wang_Rp_main(IT_WANG, MO_bool, IT, p_leading, file_graph, file_expected, results_folder, file_com_out,time_folder, pareto_file, num_pareto, ensemble_thr, n_gen, n_pop, p_offs, bool_ov,op_com=op_com,dim_emb=dim_emb, char_split=char_split, name=name,op_var=op_var,lambdav=lambdav, alpha=alpha, beta=beta, TEST_DISJ_OV=TEST_DISJ_OV)
                    if "AutoEnc" in op_emb:
                        H, U, M, S, C, mod0, mod, nmi0, nmi, f1_macro_spec, f1_micro_spec, f1_macro_U, f1_micro_U, time0 = SpecRp.emb_Rp_Attr_main(IT_WANG, MO_bool, IT, p_leading, file_graph, file_expected, results_folder, file_com_out,time_folder, pareto_file, num_pareto, ensemble_thr, n_gen, n_pop, p_offs, bool_ov,op_com=op_com,dim_emb=dim_emb, char_split=char_split, name=name,op_var=op_var,lambdav=lambdav, alpha=alpha, beta=beta,gamma1=gamma1,gamma2=gamma2,gamma3=gamma3)
            
                    time_tot = (time.perf_counter() - start_time)
                    #print("--- %s seconds ---" % (time.time() - start_time))
                    print("*--- %s seconds ---" , time_tot)
                
                    mod0_avg+=mod0
                    mod_avg+=mod
                    nmi0_avg+=nmi0
                    nmi_avg+=nmi
                    f1_macro_spec_avg += f1_macro_spec
                    f1_macro_U_avg += f1_macro_U
                    f1_micro_spec_avg += f1_micro_spec
                    f1_micro_U_avg += f1_micro_U
                    time0_avg += time0            
                    time_tot_avg+=time_tot  
                   
                mod0_avg=mod0_avg/nex
                mod_avg=mod_avg/nex
                nmi0_avg=nmi0_avg/nex
                nmi_avg=nmi_avg/nex
                f1_macro_spec_avg = f1_macro_spec_avg/nex
                f1_macro_U_avg = f1_macro_U_avg/nex
                f1_micro_spec_avg = f1_micro_spec_avg/nex
                f1_micro_U_avg = f1_micro_U_avg/nex
                time0_avg = time0_avg/nex
                time_tot_avg = time_tot_avg/nex
                
                print("f1_macro_spec_avg:", f1_macro_spec_avg)
                print("f1_micro_spec_avg:", f1_micro_spec_avg)
                
                df_results.loc[lin] = [name, n,m, round(mod0_avg,4), round(mod_avg,4), round(nmi0_avg,4), round(nmi_avg,4), round(100*f1_macro_spec_avg,2), round(100*f1_macro_U_avg,2), round(100*f1_micro_spec_avg,2), round(100* f1_micro_U_avg,2), round(time0_avg,2), round(time_tot_avg,2)]
                lin+=1
            
                
                df_results.to_csv(out_results,sep='\t', index=False, header=True) #ok
                
                print(df_results)
        
    return df_results

    print("...")


#%% run main
if __name__ == "__main__":

    bool_verbose=False
    out_verbose=None

    nex=10
    op_run_list = ['disj'] #, 'ov']# 'disj','ov']
    op_alg_list = ['autorpNorm3_SpecDecOV'] #,'wang-SpecDecOV']
    opDeltaH_list = [1]

    for op_run in op_run_list:
        #op_alg_list = ['autorp']
        for op_alg in op_alg_list:
            print("OP_ALG:", op_alg)
            TEST_DISJ_OV, op_emb, op_com, op_var, lambdav_list, alpha_list, beta_list = get_params_from_op_alg(op_alg)
            dim_emb=128
            print("op_var: ")
    
            char_split=' '

    
            lambdav = lambdav_list[0]
            alpha = alpha_list[0]
            beta = beta_list[0]

            for lambdav in lambdav_list:
                for alpha in alpha_list:
                    for beta in beta_list:
                        print("######################################################")
                        print("---- alpha={}, beta={}, lambda={}".format(alpha,beta,lambdav))
            
                        print("---- op_var={}".format(op_var))
                        
                        if op_run == 'disj':
                            call_community_embedding_datasets(op_emb=op_emb,dim_emb=dim_emb,op_com=op_com,op_var=op_var, lambdav=lambdav, alpha=alpha, beta=beta,nex=nex )     
                        elif op_run == 'ov':
                            for opDeltaH in opDeltaH_list:
                                call_community_embedding_overlapping(op_emb=op_emb,dim_emb=dim_emb,op_com=op_com,op_var=op_var, lambdav=lambdav, alpha=alpha, beta=beta, nex=nex,opDeltaH=opDeltaH)
                                    
    

