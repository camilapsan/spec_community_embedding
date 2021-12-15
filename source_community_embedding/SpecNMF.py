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
import spectral_based as Spectral
import community_embedding as ComEmb
import SpecNMF as SpecNMF
import SpecRp as SpecRp
import metrics as metrics
from sklearn.metrics.pairwise import cosine_similarity,cosine_distances

from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score

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


very_low_num = 10E-10   

#cos_sim=cosine_similarity(A.reshape(1,-1),B.reshape(1,-1))
def wang_Rp_main( it_emb, MO, IT, p_leading, file_graph, file_expected, results_folder, file_com_out, time_folder, pareto_file, num_pareto, ensemble_thr, n_gen, n_pop, p_offs, bool_ov, ov_thr=0,op_com=1,dim_emb=2, char_split=' ',name='', op_var='pdimemb',  lambdav = 1e9,  alpha=0.5, beta=5,bool_link_pred=False,bool_proposed=True,TEST_DISJ_OV=False):
    it_spec=IT
    #it_emb=30
    #pos_neg=True
    pos_neg=False
    print("TEST_DISJ_OV=",TEST_DISJ_OV)
    #M: basis matrix (n x m);
    #U: representations of nodes n x m;
    #H: community indicator matrix (n x k);
    #C: representations of communities (k x m);
    print("-- op_var", op_var)
    start_time_g = time.perf_counter()    
    time_g = (time.perf_counter() - start_time_g)        
    g = sf.read_construct_g(file_graph, bool_link_pred)    #bool_link_pred==True, isolated also True
    print("vert = ", g.vcount(), "number of edges = ", g.ecount())
    
    #file_graph = sf.create_edgelist_i0(file_graph)
    sf.create_edgelist_from_pajek(file_graph)
    p_best=None
    
    #---------------------------- results
    if file_expected is not None:        
        #---- Overlapping communities
        if TEST_DISJ_OV==False and bool_ov==True and file_expected is not None:
            com_expected,lin_expected  = sf.read_partition_expected_OV(file_expected, g, char_split="\t")
            p_expected=None
            k_expected = len(lin_expected)
        #---- Disjoint communities   
        elif (TEST_DISJ_OV==True or bool_ov==False) and file_expected is not None:  
            p_expected = sf.read_partition_expected(file_expected,char_split)
            k_expected = len(pd.unique(p_expected))
            
        print("* Number of expected communities = ", k_expected)
    else:
        p_expected = None
        
    #------------ Solve initial problems 

    start_time_spec = time.perf_counter()
    #solve H problem
    gamma1=1
    gamma2=1
    gamma3=0
    if op_com=="SpecDecOV" or op_com=="SpecDecOV-Rp":
        nome_alg="SpecDecOV"
        #specOV       
        q_best, Sov_best, g, eig_val, ri_vert_pos, ri_vert_neg, Rp, Rn, mod_mat,A,P = SpecOV.spec_ov_main(file_graph, it_spec, p_leading, gamma1, gamma2, gamma3, ov_thr, bool_ov, file_com_out, bool_verbose, out_verbose)       
    else:
        if op_com=="SpecG" or op_com=="SpecG-Rp":        
            nome_alg="SpecG"
            ##Genetic algorithm
            p_best, graph, params, eig, pop, offs, Sov_best, Rp,Rn,list_gen_Q,mod_mat, A, P = SpecG.SpecG_single_main(False, it_spec, p_leading, file_graph, results_folder, time_folder, pareto_file, num_pareto, ensemble_thr, n_gen, n_pop, p_offs, bool_ov,KNumber_pre=-1,pos_neg=pos_neg,bool_link_pred=bool_link_pred)            

        p_best=p_best[0]
        Sov_best=Sov_best[0]
        Rp=Rp[0]
        Rn=Rn[0]
        ri_vert_pos = graph.ri_vert_pos
        ri_vert_neg = graph.ri_vert_neg
        g=graph.g
        A=A.astype(float)
    print("* dim_emb=",dim_emb)
    dim_p = min(dim_emb,Rp.shape[1]) 
    
    ##new: embedding 290921
    ri_vert_pos = ri_vert_pos[:,0:dim_p ]
    Rp = Rp[:,0:dim_p ]
    # print("Rp ", Rp.shape)
    # print("Rn ", Rn.shape)

    time_spec = (time.perf_counter() - start_time_spec) + time_g 

    if "Rp" not in op_com:
        H = Sov_best    
        #print(H)
        #print("here")
    else:
        H, _ = SpecRp.autoencoder_keras(ri_vert_pos,ri_vert_pos, k_expected)

    print("DIM Rp:",Rp.shape[1])
    if bool_proposed == True:
        #U, M, S = solve_wang_U(g,dim_emb=dim_p)                
        U, M, S = solve_wang_U(g, p_expected,dim=dim_p)       
        #U, M, S = read_attributes(name, g, p_expected, dim=dim_p)
        U0 = U.copy()         
    
    else:
        U, M, S = solve_wang_U(g, p_expected,dim=dim_emb)
        U0 = U.copy()
        
    if bool_proposed == True:
        C = ( Rp - np.min(Rp) )/ (  np.max(Rp) - np.min(Rp)) #normalizar
    else:
        C = np.full((Rp.shape[0], Rp.shape[1]), 1)
         
    M0 = M
    S0 = S
    H0 = H.copy()
    C0 = C
    
    #print(U)
    #if op_com > 0: 
    Sov_nonzero = Sov_best[:, np.where( np.all(Sov_best[..., :] == 0, axis=0) == False )[0]]
    Sov_nonzero.shape
        #SpecOV.plot_vertex_vectors(g, min_max_norm(U0), min_max_norm(U0), Rp, Rn, Sov_nonzero, op="emb")
            
    #WAIT input("init ok. Next?")
    for it in np.arange(0, it_emb):
        print("---------it:", it, "H_UCt")
        #update M
        M = normalize2( update_wang_M(M, S, U) )
		# update U 
        U = normalize2(update_wang_U(S, U, M, C, H,alpha)    )     
        #PLOT SpecOV.plot_vertex_vectors(g,min_max_norm(U), min_max_norm(U), Rp, Rn, H, op="emb")
		# Update C
        C = normalize2(update_wang_C(C, H, U) )
        #print(M)
        H, p_best = normalize2(update_H(P, H, A, U, C, alpha, beta, lambdav, op_var,g,  ri_vert_pos, ri_vert_neg, Rp, Rn, p_best) )
       # mod0, nmi0, mod, nmi, metric_1, metric_2, f1_macro_spec, f1_micro_spec = metrics_disjoint(H, H,H0, U, U0, ri_vert_pos,g, results_folder, nome_alg, name, file_com_out, file_expected,char_split);

        H_nonzero = H[:, np.where( np.all(H[..., :] == 0, axis=0) == False )[0]]

    H_nonzero = H[:, np.where( np.all(H[..., :] == 0, axis=0) == False )[0]]    
           

    print("H shape")
    print(H.shape)
    
    # p_best = create_part_from_Sov(H_nonzero).astype(int)
    # plot_embedding(H_nonzero, C,p_best, file+"H_wang.png")    
    # plot_embedding(U, C,p_best, file+"U_wang.png")
    # plot_embedding(U0, C,p_best, file+"U0_wang.png")
    
#---------------------------- results
    if file_expected is not None and bool_link_pred==False:
        #---- Overlapping communities
        if TEST_DISJ_OV ==False and bool_ov==True and file_expected is not None:
            #must save on lines format 
            mod0, nmi0, mod, nmi, metric_1, metric_2, f1_macro_spec, f1_micro_spec = metrics.metrics_overlapping(H, H0, U, U0, ri_vert_pos,g, results_folder, nome_alg, name, file_com_out, file_expected );

        #---- Disjoint communities   
        elif (TEST_DISJ_OV==True or bool_ov==False) and file_expected is not None:  
            mod0, nmi0, mod, nmi, metric_1, metric_2, f1_macro_spec, f1_micro_spec = metrics.metrics_disjoint(H, H,H0, U, U0, ri_vert_pos,g, results_folder, nome_alg, name, file_com_out, file_expected,char_split);

        return H, U, M, S, C, mod0, mod, nmi0, nmi, f1_macro_spec, f1_micro_spec, metric_1, metric_2, time_spec
    
    else:
        return H, U, M, S, C, -1, -1, -1, -1, -1, -1, -1, -1, -1



#%% NMF - Embedding Wang - main 

#cos_sim=cosine_similarity(A.reshape(1,-1),B.reshape(1,-1))
def wang_main( it_emb, MO, it_spec, p_leading, file_graph, file_expected, results_folder, file_com_out, time_folder, pareto_file, num_pareto, ensemble_thr, n_gen, n_pop, p_offs, bool_ov, ov_thr=0,op_com=1,dim_emb=2, char_split=' '):
#if 1==1:    
    #M: basis matrix (n x m);
    #U: representations of nodes n x m;
    #H: community indicator matrix (n x k);
    #C: representations of communities (k x m);

    #TODO: calc p_leading to dim_emb

    start_time_g = time.perf_counter()

        
    g = sf.read_construct_g(file_graph)
    time_g = (time.perf_counter() - start_time_g)
    
    gamma1=1  
    gamma2 =1
    mod_mat = sf.construct_modularity_generalized(g, gamma1, gamma2).transpose()
    A = np.array(g.get_adjacency(attribute="weight").data) #mat_adj
    P = np.array(sf.construct_null_model(g, gamma2)) #mat_null: P
    
    lambdav = 1e9
    alpha=0.5 #100
    beta=5 #10
    
    p_best=None
    #------------ Solve initial problems 
    
    start_time_spec = time.perf_counter()
    #solve H problem
    if op_com==1:
        #specOV
        q_best, Sov_best, g, eig_val, ri_vert_pos, ri_vert_neg, Rp, Rn = SpecOV.spec_ov_main(file_graph, it_spec, p_leading, gamma1, gamma2, gamma3, ov_thr, bool_ov, file_com_out, bool_verbose, out_verbose)       
    elif op_com==2:
        gamma1=1
        gamma2=1
        print("gammas", gamma1, ",", gamma2)
        ##Genetic algorithm
        p_best, graph, params, eig, pop, offs, Sov_best, Rp,Rn,list_gen_Q = SpecG.SpecG_single_main(False, it_spec, p_leading, file_graph, results_folder, time_folder, pareto_file, num_pareto, ensemble_thr, n_gen, n_pop, p_offs, bool_ov, mo_weight_Qin=gamma1, mo_weight_Qnull=gamma2,gamma3=gamma3, mod_orig=True)
        
        p_best=p_best[0]
        Sov_best=Sov_best[0]
        Rp=Rp[0]
        Rn=Rp
        ri_vert_pos = graph.ri_vert_pos
        g=graph.g
    elif op_com==3:
        ##Genetic algorithm
        #p_best, graph, params, eig, pop, offs, Sov_best, Rp,list_gen_Q = ensemble_MOSpecG_main(MO_bool, num_it, p_leading, file_graph, results_folder, time_folder, pareto_file, num_pareto, ensemble_thr, n_gen, n_pop, p_offs,  bool_ov, ov_thr)                
        p_best, graph, params, eig, pop, offs, Sov_best, Rp, Rn,list_gen_Q = SpecG.SpecG_single_main(False, it_spec, p_leading, file_graph, results_folder, time_folder, pareto_file, num_pareto, ensemble_thr, n_gen, n_pop, p_offs, bool_ov, mo_weight_Qin=gamma1, mo_weight_Qnull=gamma2, mod_orig=False)
        
        p_best=p_best[0]
        Sov_best=Sov_best[0]
        Rp=Rp[0]
        Rn=Rp
        ri_vert_pos = graph.ri_vert_pos
        g=graph.g
        
    time_spec = (time.perf_counter() - start_time_spec) + time_g 

       # plot_vertex_vectors(g, ri_vert_pos, ri_vert_neg, Rp, Rn, Sov, op="vectors")
    if op_com>0:
        H = Sov_best 
        k_com = np.shape(Sov_best)[1]    
        
        #280921 TESTE
        row_sums = H.sum(axis=1)
        H = H / row_sums[:, np.newaxis]
    
        #280921 TESTE
        row_sums = Rp.sum(axis=1)
        Rp = Rp / row_sums[:, np.newaxis]
    
    else: #random init
        print("NÃO USAR AQUi")
        #TODO: overlapping        
        p_expected = sf.read_partition_expected(file_expected,char_split)
        #k_expected = len(np.unique(p_expected))
        k_com = np.unique(p_expected).shape[0]
        #H = np.random.rand(g.vcount(),k_com)
        #H = np.random.randint(2, size=(g.vcount(),k_com))
        H = np.zeros(( g.vcount(),k_com ))
        for i in np.arange(0,g.vcount()):
            ki = np.random.randint(k_com, size=(1))[0]
            H[i,ki]=1
            
        Rp=None
        Rn=None
        
    #TODO: non_zero
    print("NaN----------")
    print(np.isnan( Rp[np.isnan(Rp)==True ] ))
          
    dim_emb = Rp.shape[1] #160921
    U, M, S = solve_wang_U(g,dim_emb=dim_emb)
    
    C = np.full((k_com,dim_emb), 1)
    
    print("NaN----------")
    print(np.isnan( C[np.isnan(C)==True ] ))
    
    C = update_wang_C(C, H, U) #190921
    #print(C)
    
    #TODO: non_zero
    print("NaN----------")
    print(np.isnan( C[np.isnan(C)==True ] ))
    #010421:
    #C = Rp[:,0:dim_emb]
    #C must be positive
    #C = (C + abs(np.min(C))) 
    #C= C / abs(np.max(C))
    
    U0 = U
    M0 = M
    S0 = S
    #H0 = H    
    H0 = H.copy()
    C0 = C

    print("U NaN----------")
    #print(np.isnan( U[np.isnan(U)==True ] ))
    if op_com > 0: 
        #idx = np.argwhere(np.all(Sov_best[..., :] == 0, axis=0))[0]
        #idx= np.where( np.all(Sov_best[..., :] == 0, axis=0) == False )[0]
        Sov_nonzero = Sov_best[:, np.where( np.all(Sov_best[..., :] == 0, axis=0) == False )[0]]
        Sov_nonzero.shape
        #SpecOV.plot_vertex_vectors(g, ri_vert_pos, ri_vert_neg, Rp, Rn, Sov_best, op="vectors") #070321
        #SpecOV.plot_vertex_vectors(g, ri_vert_pos, ri_vert_neg, Rp, Rn, Sov_nonzero, op="vectors") #070321
        #PLOT SpecOV.plot_vertex_vectors(g, ri_vert_pos, ri_vert_neg, Rp, Rn, Sov_nonzero, op="emb") #070321
        #SpecOV.plot_vertex_vectors(g, ri_vert_pos, ri_vert_neg, Rp, Rn, H0, op="emb")    
        
        #SpecOV.plot_vertex_vectors(g, min_max_norm(U0), min_max_norm(U0), Rp, Rn, H0, op="emb")    
        SpecOV.plot_vertex_vectors(g, min_max_norm(U0), min_max_norm(U0), Rp, Rn, Sov_nonzero, op="emb")
        #SpecOV.plot_vertex_vectors(g, C, C, Rp, Rn, Sov_best, op="emb")                
    
    #WAIT input("init ok. Next?")
    for it in np.arange(0, it_emb):
        print("it:", it, "H_UCt")
        #update M
        M = update_wang_M(M, S, U)
		# update U 
        U = update_wang_U(S, U, M, C, H,alpha)
        #print(U)
        #PLOT SpecOV.plot_vertex_vectors(g,min_max_norm(U), min_max_norm(U), Rp, Rn, H, op="emb")
        
		# Update C
        C = update_wang_C(C, H, U)
        
        #TODO FUNCTION 
        PH = np.dot(P, H)
        HHH = np.dot(H , np.dot( H.transpose() , H ) )
        
        #delta =  (2*beta*mult_null_H)^2 + 16*lambda*HHH * (2*beta*mat_adj%*%H + 2*alpha*U%*%t(C) + (4*lambda - 2*alpha)*H )       
        delta= np.power(2*beta*PH,2) + np.multiply( 16*lambdav*HHH , ( 2*beta*np.dot(A,H) + 2*alpha* np.dot(U,C.transpose()) + (4*lambdav - 2*alpha)*H  ) )
        H = np.multiply(H, ( np.sqrt( (-2*beta*PH + np.sqrt(delta)) /  np.maximum( very_low_num, (8*lambdav*HHH)) ) ) )	
        
        #280921 TESTE
        row_sums = H.sum(axis=1)
        H = H / row_sums[:, np.newaxis]
        
        H_nonzero = H[:, np.where( np.all(H[..., :] == 0, axis=0) == False )[0]]
        #PLOT SpecOV.plot_vertex_vectors(g,min_max_norm(U), min_max_norm(U), Rp, Rn, H_nonzero, op="emb")
        #print(U)
        print("U NaN----------")
        print(np.isnan( U[np.isnan(U)==True ] ))
   
    #input("it " + str(it) + " ok. Next?") 
    # SpecOV.plot_vertex_vectors(g, ri_vert_pos, ri_vert_neg, Rp, Rn, H, op="emb")
    H_nonzero = H[:, np.where( np.all(H[..., :] == 0, axis=0) == False )[0]]
    #PLOT 
    
    if op_com > 0: 
        SpecOV.plot_vertex_vectors(g,min_max_norm(U), min_max_norm(U), Rp, Rn, H, op="emb")

#---------------------------- results
    if file_expected is not None:
        
        #---- Overlapping communities
        if bool_ov==True and file_expected is not None:
            #must save on lines format 
            #must calc oNMI (external code)]    
            
            #print(graph.g.vs['Index'])
            H_geq_0 = np.where(H >0)
            H[H_geq_0] = 1
            
            print("\n---------\nCommunity detection:")
            base_folder_out = results_folder+""+nome_alg + "" + "/"+ name +"/"
            file_com_out = base_folder_out+"lines_net_" + name + ".com"
                    
            SpecOV.print_com_lines_file(file_com_out, H) #231220
            
            shutil.copyfile(file_com_out, "f1.txt")
            shutil.copyfile(file_expected, "f2.txt")
    
            os.system('./mutual3/mutual "f1.txt" "f2.txt"')  
        
            #print(graph.g.vcount())
            #com_expected,lin_expected  = sf.read_partition_expected_OV(file_expected,  graph.g.vcount(), char_split="\t")
    
            com_expected,lin_expected  = sf.read_partition_expected_OV(file_expected, g, char_split="\t")
            
            #print(lin_expected)
            #print(com_expected)
            jacc_val = metrics.score_overlapping( H0,com_expected, lin_expected, type_sim="jaccard")        
            f1_val = metrics.score_overlapping( H0,com_expected, lin_expected, type_sim="f1score")
            print("jacc_score0",jacc_val)
            print("f1_score0",f1_val)
            
            jacc_val = metrics.score_overlapping( H,com_expected, lin_expected, type_sim="jaccard")        
            f1_val = metrics.score_overlapping( H,com_expected, lin_expected, type_sim="f1score")
            print("jacc_score - Wang",jacc_val)
            print("f1_score -Wang",f1_val)
            mod0=0
            nmi0=0
            nmi=0
            mod=0
            
            metric_1 = f1_val
            metric_2 = jacc_val
            #print("\n---------\nClassification using ri_vert_pos")        
            #print('\n---------\n')
        
            f1_macro_spec=0
            f1_micro_spec=0
            
        #---- Disjoint communities   
        elif bool_ov==False and file_expected is not None:  
        
            #--- node classification 
            #TODO: FUNCTION?
            p_expected = sf.read_partition_expected(file_expected,char_split)
            #k_expected = len(np.unique(p_expected))
    
            #if p_best is None:
            p_best_0 = sf.create_part_from_Sov(H0)        
            p_best = sf.create_part_from_Sov(H_nonzero)
            
            #TODO reportar NMI e mod de ambos os H
            
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
            
            if op_com > 0:
                print('\n---------\n')
                print("Classification using ri_vert_pos")
                f1_macro_spec, f1_micro_spec = node_classification.node_classification(ri_vert_pos, p_expected)
            else: 
                f1_macro_spec=-1
                f1_micro_spec=-1
            print("\nClassification using U")
            f1_macro_U, f1_micro_U = node_classification(U, p_expected)
            print('\n---------\n')
    
            print("Initial U: Classification using U0")
            f1_macro_U0, f1_micro_U = node_classification(U0, p_expected)
            print('\n---------\n')
            
            
            metric_1=f1_macro_U
            metric_2=f1_micro_U

    return H, U, M, S, C, mod0, mod, nmi0, nmi, f1_macro_spec, f1_micro_spec, f1_macro_U, f1_micro_U, time_spec


#%%% Updates
def update_wang_U(S, U, M, C, H, alpha=1):
    #aux_U = ( np.dot(S.transpose(), M) + (alpha * np.dot(H,C) ) )/ np.dot( U , ( np.dot( M.transpose(), M) + (alpha* np.dot( C.transpose(), C) ) )    )       
    aux_U = ( np.dot(S.transpose(), M) + (alpha * np.dot(H,C) ) )/ (
            np.maximum( very_low_num,   np.dot( U , ( np.dot( M.transpose(), M) + (alpha* np.dot( C.transpose(), C) ) )    )       )
            )
    U = np.multiply(U, aux_U)
    
    
    #280921 TESTE
    #row_sums = U.sum(axis=1)
    #U = U / row_sums[:, np.newaxis]
    
    return U
    

def update_wang_M(M, S, U):
    #aux_M = np.dot(S, U) / np.dot( M , np.dot( U.transpose(), U))  
    aux_M = np.dot(S, U) / np.maximum( very_low_num, np.dot( M , np.dot( U.transpose(), U))  )
    M = np.multiply(M, aux_M)
    # M = min_max_norm(M)
	
    #280921 TESTE
    #row_sums = M.sum(axis=1)
    #M = M / row_sums[:, np.newaxis]
        
    return M


def update_wang_C(C, H, U):
    #TODO: min>0
    #aux_C =  np.dot( H.transpose() , U) / (  np.dot( np.dot(C, U.transpose()) , U  )  )
    aux_C =  np.dot( H.transpose() , U) /  np.maximum( very_low_num, (  np.dot( np.dot(C, U.transpose()) , U  )  ) )
    C = np.multiply( C , aux_C)
    # C = min_max_norm(C)
    
    #280921 TESTE
   #row_sums = C.sum(axis=1)
    #C = C / row_sums[:, np.newaxis]
    #sys.float_info.min
    return C

def update_H (P, H, A, U, C, alpha, beta, lambdav, op_var, g,  ri_vert_pos, ri_vert_neg, Rp, Rn, p_best):
    print(op_var)
    if 'H' in op_var: 
        print('H heuristic')
        Sov=H
        R_group_pos = Rp
        R_group_neg = Rn
        pcurr=p_best
        #H = H_update_heuristic(H, C, rp, Rp )
        Rp, Rn, p_best, H, valQ=H_update_heuristic(g,H, C, ri_vert_pos, ri_vert_neg, Rp, Rn, p_best, U)
        
    else:
        #TODO!! change H
        print('classical H')
       
        #*: np.dot
        #.*: np.multiply
        
        # print("H")1
        # print(H)
        # print("sum")
        # print(np.sum(H, axis=1)[0:3])
        PH = np.dot(P, H)
        HHH = np.dot(H , np.dot( H.transpose() , H ) )            
        UC = np.dot(U,C.transpose())
        delta_comp1= np.square(2*beta*PH) 
        delta_comp2 = np.multiply( 
            16*lambdav*HHH , 
            (
                2*beta*np.dot(A,H)
                + 2*alpha* UC
                + (4*lambdav - 2*alpha)*H  ) 
            )
        delta = delta_comp1 + delta_comp2
        ##delta[ np.where(delta<0)] = 0 #300921
        
        aux = ( (-2*beta*PH + np.sqrt(delta)) /  np.maximum( very_low_num, (8*lambdav*HHH)) ) 
        aux[np.where(aux<0)]=0
        # print(H.shape)
        H = np.multiply(H, np.sqrt(aux) )	
        # print(H.shape)
        # print("Hup")

        #print(H)
        # print("sum")
        # print(np.sum(H, axis=1)[0:3])
        
        #nao está somando 1
        
        
        #print(H==H0)                
        #280921 TESTE
        #row_sums = H.sum(axis=1)
        #H = H / row_sums[:, np.newaxis]
        #print(H)
        # PH = np.dot(P, H)
        # HHH = np.dot(H , np.dot( H.transpose() , H ) )            
        # delta= np.power(2*beta*PH,2) + np.multiply( 16*lambdav*HHH , ( 2*beta*np.dot(A,H) + 2*alpha* np.dot(U,C.transpose()) + (4*lambdav - 2*alpha)*H  ) )
        # H = np.multiply(H, ( np.sqrt( (-2*beta*PH + np.sqrt(delta)) /  np.maximum( very_low_num*100, (8*lambdav*HHH)) ) ) )	
 
    return H,p_best

#%%% Embedding Wang - aux

def min_max_norm(x):
    return ( x - np.min(x) ) / (np.max(x) - np.min(x)) 
    #TEst
    #return x 



#%%% Not used H heuristic            

def calc_H_SpecNMF(R_group_pos, R_group_neg, divtotal,Sov, U,C, lambdav, alpha, beta):
  valQ=0
  # print("num_groups="+str(np.shape(R_group_pos)[0]))
  for s in range(0,np.shape(R_group_pos)[0]):
    valQ= valQ + np.inner( R_group_pos[s,:], R_group_pos[s,:] ) - np.inner( R_group_neg[s,:], R_group_neg[s,:] )  

  valQ = valQ/divtotal  
  
  H_UC = Sov -  np.dot(U, C.transpose() )   
  #val_UC = alpha*np.power(linalg.norm(H_UC),2)
  val_UC = alpha*np.trace( np.dot( H_UC, H_UC.transpose()   ))    
    
  valQ -= val_UC
  
  return valQ, val_UC

def calc_gain_H_SpecNMF(valHUC_old, R_group_pos, R_group_neg, t_new, t_old, ri_vert_pos, ri_vert_neg, v, Sov, U,C, lambdav, alpha, beta):  
    val_H = np.inner( R_group_pos[t_new,:], ri_vert_pos[v,:] ) - np.inner( R_group_neg[t_new,:], ri_vert_neg[v,:] )
    #val -= np.inner( R_group_pos[t_old,:], ri_vert_pos[v,:] ) - np.inner( R_group_neg[t_old,:], ri_vert_neg[v,:] )     #wrong
    #251120  
    val_H -= np.inner( R_group_pos[t_old,:]-ri_vert_pos[v,], ri_vert_pos[v,:] ) - np.inner( R_group_neg[t_old,:]-ri_vert_neg[v,], ri_vert_neg[v,:] )    
    #print("new calc update")
    #???  val += np.inner(ri_vert_pos[v,:],ri_vert_pos[v,:]) -  np.inner(ri_vert_neg[v,:],ri_vert_neg[v,:])   
    
    val_H *= beta 
    
    
    #TODO: precisa simular o movimento aqui
    
    #TODO: testa e volta

    Sov[v,t_old]=0
    Sov[v,t_new]=1
    H_UC = Sov -  np.dot(U, C.transpose() ) 
    Sov[v,t_old]=1
    Sov[v,t_new]=0
    
    val_UC =alpha* np.trace( np.dot( H_UC, H_UC.transpose()   ))    
    #np.trace( np.dot(Sov, Sov.transpose()) - 2* np.dot( np.dot(Sov, C), U.transpose()) + np.dot( np.dot( np.dot(U, C.transpose()) , C), U.transpose()) )     
    #val_UC = alpha*np.power(linalg.norm(H_UC),2)
         
    val = val_H - (val_UC - valHUC_old)
    valHUC_delta = (val_UC - valHUC_old) 
    return val,valHUC_delta

def H_update_heuristic(g,Sov, C, ri_vert_pos, ri_vert_neg, R_group_pos, R_group_neg, pcurr, U):
      
    #This is going to be part of the genetic algorithm..
    #def spec_local_search(g, ri_vert_pos, ri_vert_neg, eig_pos_ind, eig_neg_ind, comms_est, R_group_pos, R_group_neg, Sov, pcurr, it_max=10,bool_verbose=False):
    it_ls=1
    it_max=10
    bool_improve=True
    
    Q_ini, valHUC_old = calc_H_SpecNMF(R_group_pos, R_group_neg,  2*g.ecount(),Sov, U,C, lambdav, alpha, beta)
    print("Q_ini=",Q_ini)    
    while bool_improve==True and it_ls < it_max: 
      bool_improve=False
    
      allv = np.arange(0,g.vcount(),1)
      allv_sample = np.random.choice(allv, len(allv), replace=False )
    
      for v in allv_sample:
        #print("** Try to move vertex " + str(v))
        val_best = -100
        t_best = -1
        t_old = pcurr[v] #current community of v
        
        #find the best candidate community t_best to insert v into
        comms_cand = [c for c in np.unique(pcurr) if c>=0 and c!=t_old ]
        # print("candidates")
        # print(comms_cand)
    
        for t in comms_cand:     
          # print("t_old"+str(t_old))
          # print("t="+str(t))
          # print("v="+str(v))
                    
          #val = calc_gainQ_Rgroup_update(R_group_pos, R_group_neg, t, t_old, ri_vert_pos, ri_vert_neg, v)
          val,valHUC_delta = calc_gain_H_SpecNMF(valHUC_old,R_group_pos, R_group_neg, t, t_old, ri_vert_pos, ri_vert_neg, v, Sov, U,C, lambdav, alpha, beta)  #TODO
          #print(valHUC_delta)
          #print(valHUC_delta>0)
          #print(valHUC_delta > 0 (val>val_best or t_best==-1))
          #if val>0 and (val>val_best or t_best==-1):          
          if valHUC_delta > 0 and (val>val_best or t_best==-1):          
            #print(valHUC_delta)
            #print("*** found new best, Check " + str(v) + " to community " + str(t) + ", val=" + str(val), ", val_best" + str(val_best) + ", t_best=", t_best)     
            val_best = val
            t_best = t 
    
        if t_best>-1 and t_best!= t_old: #just checking again for t_best!=t_old
          #move vertex v to community t_best
          #if bool_verbose:
          ##print("Move " + str(v) + " to community " + str(t_best))
    
          pcurr[v] = t_best
          Sov[v,t_old]=0
          Sov[v,t_best]=1
                                      
          R_group_pos, R_group_neg = sf.move_vertex_R_group(R_group_pos, R_group_neg, t_best, t_old, ri_vert_pos, ri_vert_neg, v)
          bool_improve = True
    
        #all valid communities, disconsidering old_o      
      it_ls+=1
    
    #print("it ls=",it_ls)
    #do not need
    #valQ = sf.calc_Q_Rgroup(R_group_pos, R_group_neg, 2*g.ecount())
    #valQ = calc_Q_H_SpecNMF(R_group_pos, R_group_neg, 2*g.ecount()) #TODO
    valQ = calc_H_SpecNMF(R_group_pos, R_group_neg,  2*g.ecount(),Sov, U,C, lambdav, alpha, beta)
    
    if bool_verbose==True:
      #print(".. LS, IT = "+str(it_ls))
      conf_Q_vec = g.modularity(pcurr, g.es['weight'])      
      print("..Q-LS:"+ str(valQ)  + ", Q_igraph-LS="+str(conf_Q_vec))


    #%TODO: atualiza U?       
    return R_group_pos, R_group_neg, pcurr, Sov, valQ

#pcurr, Sov, valQ = local_search(g, ri_vert_pos, ri_vert_neg, eig_pos_ind, eig_neg_ind, comms_est, R_group_pos, R_group_neg, Sov, pcurr)  
#print(pcurr) 
#print(valQ)

    
       

#%% Wang - aux

#%%% U
    
def solve_wang_U(g,p_expected,dim=2,gamma1=1,par_eta=5):
    
    adj = g.get_adjacency(attribute="weight")
    #S1 = gamma1*np.matrix(adj.data)    
    S1 = gamma1*np.array(adj.data)    
    S2 = np.zeros((g.vcount(), g.vcount()))
    
    print("Start cosine similarity")
    S2 = cosine_similarity(S1)
    print("End cosine similarity")    
    
    #S = S1 + par_eta * S2
    S = S1 +par_eta*S2
    
    print("Run NMF")
    #||S-MU^T||^2
    
    #011021
    model = NMF(n_components=dim, init='nndsvdar') #, init='random') #b, init='random' random_state=0,
    M = model.fit_transform(S) #M
    U = np.transpose(model.components_) #U
    
    #f1_macro, f1_micro= node_classification(U0, p_expected, test_size=0.3)  
    #U=U0
    #x_train = S

    #U1, _ = autoencoder_keras(x_train,x_train, dim)
    #f1_macro, f1_micro= node_classification(U1, p_expected, test_size=0.3)  
     
    # M = ( np.random.rand(g.vcount(),dim) )
    # M = update_wang_M(M, S, U) 
    
    # from karateclub import DANMF
    # G = nx.read_pajek(file_graph)    
    # G = G.to_undirected()
    # zip_mapping = zip(G.nodes(), np.arange(0, len(G.nodes())) )
    # mapping = dict(zip_mapping)
    # G=nx.relabel_nodes(G, mapping)

    # model = DANMF(layers= [144, 99],)
    # model.fit(G)


    # embedding = model.get_embedding()
    #f1_macro, f1_micro= node_classification(decoded_img, p_expected, test_size=0.3)  
    
    #x_train = np.zeros((g.vcount(), g.vcount(),1 ))
    
    #for i in np.arange(g.vcount()):
        #x_train[i][:,:] = S[i]
    #x_train[0,1,1] = S
    # x_train = S
    # A_enc, A_dec = autoencoder_keras(x_train, x_train)  
    # U = A_dec[0]    
    # print("********** AUTOENCODER")
    
    # f1_macro, f1_micro= node_classification(U, p_expected, test_size=0.3)  
    # f1_macro, f1_micro= node_classification(U0, p_expected, test_size=0.3)  
    
    #U = min_max_norm(U)
    #M = min_max_norm(M)
    #S = min_max_norm(S)
    
    #280921 TESTE
    #row_sums = M.sum(axis=1)
    #M = M / row_sums[:, np.newaxis]
    
    #280921 TESTE
    #row_sums = U.sum(axis=1)
    #U = U / row_sums[:, np.newaxis]
    
    #M = ( np.random.rand(g.vcount(),dim_p) )
    
    #------------
    #https://www.analyticsvidhya.com/blog/2021/06/complete-guide-on-how-to-use-autoencoders-in-python/



    print(S.shape)
    print(U.shape)
    
    return U, M, S
        

#%%% Score overlapping

def normalize(mat):
    #return mat/np.sum(mat)
    return ( mat - np.min(mat) )/ (  np.max(mat) - np.min(mat)) 
def normalize2(mat):
    return mat