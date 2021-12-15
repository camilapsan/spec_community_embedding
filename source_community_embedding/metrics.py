from pathlib import Path
import sys
path = str(Path(Path(__file__).parent.absolute()).parent.absolute())

sys.path.insert(0, path)
sys.path.insert(0, path + '/source_community_detection/')
sys.path.insert(0, path + '/source_community_embedding/')
sys.path.insert(0, path + '/source_graph_util/')


import MOSpecG_partitioning_OV as SpecG
import SpecOV_partitioning_v2 as SpecOV
import spectral_functions as sf
import community_embedding as ComEmb
import spectral_based as Spectral
import SpecNMF as SpecNMF
import SpecRp as SpecRp

from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score

import pandas as pd
import numpy as np
import subprocess
import shutil


def jaccard_communities(com_c1,com_c2):
    return len(np.intersect1d( com_c1,com_c2)  ) / len(np.union1d( com_c1,com_c2)    )
    #test sklearn

def f1score_communities(com_c1,com_c2):
    return f1_score(com_c2,com_c1)

#TODO: USE FUNCTION FOR vGraph as well
def score_overlapping( Sov_best,com_expected, lin_expected, type_sim="jaccard"):
    
    overall_val = 0
    
    for c1 in range(com_expected.shape[1]):
        term_1 =0 
        
        for c2 in range(Sov_best.shape[1]):
            #print("c1",c1)
            #print("c2",c2)
            com_c1 = np.where(com_expected[:,c1] >0)[0]  #vertices start at 0 in python
            com_c2 = np.where(Sov_best[:,c2] >0)[0]  #vertices start at 0 in python
            
            if len(com_c1) > 0 and len(com_c2):
                if type_sim == "jaccard": 
                    sim_val = jaccard_communities(com_c1,com_c2)
                elif type_sim == "f1score":
                    sim_val = f1_score(com_expected[:,c1],Sov_best[:,c2].astype(int))
                else:
                    sim_val=-1
                #print(sim_val)
                term_1 = max( term_1, sim_val)
            #print(term_1)
            
        term_1 = term_1/(2*com_expected.shape[1])
        overall_val += term_1
        #print(term_1)
        #print(overall_val)
                
    for c2 in range(Sov_best.shape[1]):
        term_2 =0 
        
        for c1 in range(com_expected.shape[1]):    
            com_c1 = np.where(com_expected[:,c1] >0)[0]  #vertices start at 0 in python
            com_c2 = np.where(Sov_best[:,c2] >0)[0]  #vertices start at 0 in python
            
            if len(com_c1) > 0 and len(com_c2):
                if type_sim == "jaccard": 
                    sim_val = jaccard_communities(com_c1,com_c2)
                elif type_sim == "f1score":
                    sim_val = f1_score(com_expected[:,c1],Sov_best[:,c2].astype(int))
                else:
                    sim_val=-1
                #print(sim_val)
                term_2 = max( term_2, sim_val)            
            
        term_2 = term_2/(2*Sov_best.shape[1])    
        overall_val += term_2
        
    return overall_val


#%%% Node classification- Score
def metrics_overlapping(H, H0, U, U0,ri_vert_pos, g, results_folder, nome_alg, name, file_com_out, file_expected,bool_U0=True):
     
    H_geq_0 = np.where(H >0)
    H[H_geq_0] = 1
    
    print("\n---------\nCommunity detection:")
    base_folder_out = results_folder+""+nome_alg + "" + "/"+ name +"/"
    file_com_out = base_folder_out+"lines_net_" + name + ".com"
            
    SpecOV.print_com_lines_file(file_com_out, H) #231220
    
    shutil.copyfile(file_com_out, "f1.txt")
    shutil.copyfile(file_expected, "f2.txt")
    
    #os.system('./mutual3/mutual "f1.txt" "f2.txt"')  
    
    print("\n---------\nONMI:")

    #onmi = os.system('./mutual3/mutual "f1.txt" "f2.txt"')       
    outcmd= subprocess.check_output('.././Overlapping-NMI-master/onmi "f1.txt" "f2.txt"', shell=True)
    print("  oNMI=",outcmd)
    nmi= float(outcmd.decode().split('\n')[2].split('\t')[1])
    print("  oNMI=",nmi)        
    
    com_expected,lin_expected  = sf.read_partition_expected_OV(file_expected, g, char_split="\t")
    
    #print(lin_expected)
    #print(com_expected)
    jacc_val0 = score_overlapping( H0,com_expected, lin_expected, type_sim="jaccard")        
    f1_val0 = score_overlapping( H0,com_expected, lin_expected, type_sim="f1score")
    print("jacc_score0",jacc_val0)
    print("f1_score0",f1_val0)
    
    jacc_val = score_overlapping( H,com_expected, lin_expected, type_sim="jaccard")        
    f1_val = score_overlapping( H,com_expected, lin_expected, type_sim="f1score")
    print("jacc_score - Wang",jacc_val)
    print("f1_score -Wang",f1_val)
    mod0=0
    nmi0=0

    mod=0
    
    metric_1 = f1_val
    metric_2 = jacc_val
    #print("\n---------\nClassification using ri_vert_pos")        
    #print('\n---------\n')
    
    f1_macro_spec=f1_val0
    f1_micro_spec=jacc_val0
    

    return mod0, nmi0, mod, nmi, metric_1, metric_2, f1_macro_spec, f1_micro_spec

def metrics_disjoint(H, H_nonzero, H0, U, U0,ri_vert_pos, g, results_folder, nome_alg, name, file_com_out, file_expected,char_split,bool_U0=True):
 
    #--- node classification 
    p_expected = sf.read_partition_expected(file_expected,char_split)

    #if p_best is None:
    p_best_0 = sf.create_part_from_Sov(H0)        
    p_best = sf.create_part_from_Sov(H_nonzero)
    
    print("\n---------\nCommunity detection:")
    print("Number of communities - H0", len(pd.unique(p_best_0)))
    
    mod0 = g.modularity(p_best_0, g.es['weight'])
    print("mod - H0=",mod0)

    nmi0 = sf.calc_nmi(p_best_0, p_expected)
    print("nmi- H0=",nmi0)
    
    print("Number of communities - H", len(pd.unique(p_best)))
    mod = g.modularity(p_best, g.es['weight'])
    print("mod=",mod)

    nmi = sf.calc_nmi(p_best, p_expected)
    print("nmi=",nmi)
    
    #if op_com > 0:
    print('\n---------\n')
    print("Classification using ri_vert_pos")
    f1_macro_spec, f1_micro_spec = node_classification(ri_vert_pos, p_expected)
    f1_macro_U=f1_macro_spec
    f1_micro_U=f1_micro_spec


    if bool_U0 == True:
                
        print("\nClassification using U")
        f1_macro_U, f1_micro_U = node_classification(U, p_expected)
        print('\n---------\n')    
        
        print("Initial U: Classification using U0")
        f1_macro_U0, f1_micro_U0 = node_classification(U0, p_expected)
        print('\n---------\n')
        
    metric_1=f1_macro_U
    metric_2=f1_micro_U
    
    return mod0, nmi0, mod, nmi, metric_1, metric_2, f1_macro_spec, f1_micro_spec



def node_classification(X,y, test_size=0.2):
    
    f1_macro_avg=0
    f2_micro_avg=0
    
    for it in np.arange(100):
        x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=test_size) #, random_state=0)
        log_regr = LogisticRegression(max_iter=1000)
        log_regr.fit(x_train, y_train)
        
        
        y_pred = log_regr.predict(x_test)
        
        f1_macro = f1_score(y_test,y_pred, average='macro')
        f1_micro = f1_score(y_test, y_pred, average='micro')
   
        f1_macro_avg+=f1_macro
        f2_micro_avg+=f1_micro
        
        
    f1_macro_avg=f1_macro_avg/100
    f2_micro_avg=f2_micro_avg/100
    
        
    print('AVG Macro-F1=', f1_macro_avg)
    print('AVG Micro-F1=', f2_micro_avg)

    return f1_macro_avg, f2_micro_avg
