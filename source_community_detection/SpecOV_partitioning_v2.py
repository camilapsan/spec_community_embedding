#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  4 20:09:14 2020

@author: camila
"""

#%% Import

import igraph
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

import math
import scipy.sparse.linalg
import random

import spectral_functions as sf

import multiprocessing
import queue

import time
import os
import pathlib

#%% Defines

bool_verbose=False
bool_verbose_thread =False


#%% Part from Sov
def create_part_from_Sov(Sov):
    #Sov.shape
    part = np.zeros(Sov.shape[0])
    for v in np.arange( Sov.shape[0]):
        part[v] = np.argmax(Sov[v,:])

    return part

#%% Output
def print_com_lines_file(file_out, Sov):
    dir_out = '/'.join( file_out.split('/')[:-1] )
    
    if os.path.exists(os.path.dirname(dir_out)) == False:
        pathlib.Path(dir_out).mkdir(parents=True, exist_ok=True)
    
    f = open(file_out, "w")    

    k_num = np.shape(Sov)[1]
    print("knum=",k_num)
    
    k_num_not_empty=0
    for c in range(0, k_num):
        #line c 
        
        #returns only cols, since Sov[:,c] is one dim -> use #[0]       
        com_c_vert = np.where(Sov[:,c] >0)[0] + 1 #vertices start at 0 in python
        
        #print(Sov[com_c_vert,c])
        if len(com_c_vert)>0:     
            k_num_not_empty+=1
            
            np.savetxt(f, com_c_vert.reshape(1, com_c_vert.shape[0]), newline='\n', fmt='%i')
            #print(com_c_vert) 
        #else: 
            #print("community ", c , " is empty")

    #print("vertices in partition")            
    #print_com_part(file_out, Sov) #debug
    print("knum not empty=",k_num_not_empty)
    f.close()       


def print_com_part(file_out, Sov):
    f = open("test_part.txt", "w")    

    k_num = np.shape(Sov)[1]
    v_num = np.shape(Sov)[0]
    
    for v in range(0, v_num):        
        #returns only cols, since Sov[:,c] is one dim -> use #[0]       
        vert_com = np.where(Sov[v,:] > 0)[0] + 1 #vertices start at 0 in python
        
        #print(Sov[com_c_vert,c])
        if len(vert_com)>0:            
             np.savetxt(f, vert_com.reshape(1, vert_com.shape[0]), newline='\n', fmt='%i')
            #print(vert_com) #, newline='\n'            
        else: 
            print("Vertex ", v , " is empty")
    f.close()       
    
#%% Eigendecomposition 

def spec_ov_eigen_dec(g, mod_mat, plarg):
 
    start_time = time.perf_counter()
    eig_val, eig_vec, eig_pos_ind, eig_neg_ind = sf.calc_eigen(g,mod_mat, plarg, which="LM")
    print("--- scipy: %s seconds ---" , (time.perf_counter() - start_time))

    #eig_val, eig_vec, eig_pos_ind, eig_neg_ind = sf.calc_eigen_SLEPc(g,mod_mat, plarg, which="LM")
    
    #eig_val, eig_vec, eig_pos_ind, eig_neg_ind = sf.calc_eigen(g,mod_mat, plarg, which="LR")
    #comms_est = len(eig_val[eig_val > math.sqrt( max(eig_val) )])
    comms_est = sf.est_number_communities(eig_val)
    
    print("Estimation k: " + str( comms_est))
    
    #print("** Calculate vertex vectors ri... **")
    ri_vert_pos = sf.calc_ri_vert_pos(g, eig_val, eig_vec, eig_pos_ind)
    ri_vert_neg = sf.calc_ri_vert_neg(g, eig_val, eig_vec, eig_neg_ind)
        
   
    return eig_val, eig_vec, eig_pos_ind, eig_neg_ind, comms_est, ri_vert_pos, ri_vert_neg

#%% Overlapping
 
#TODO: C
def spec_overlapping(g, ri_vert_pos, ri_vert_neg, eig_pos_ind, eig_neg_ind, comms_est, R_group_pos, R_group_neg, Sov, pcurr, it_max=10, bool_verbose=False,opDeltaH=0):
  bool_continue=True

  it_ov=0  
  allv = np.arange(0,g.vcount(),1)

  if opDeltaH==1:
    IDelta = np.linspace(0.5,0.5,num=1)#011021
  elif opDeltaH==2:
    IDelta = np.linspace(0.1,0.9,num=2) #231220
  elif opDeltaH==4:
    IDelta = np.linspace(0.1,0.9,num=4) #231220
  elif opDeltaH==10:
    IDelta = np.linspace(0.1,1,num=10)

  print("IDelta: ", opDeltaH) #231220
  print(IDelta) #231220
  
  #set_comm is the set with all possible communities
  set_comm = np.arange(comms_est)  #print(set_comm)

  while bool_continue==True and it_ov<it_max:
    bool_continue=False
    allv_sample = np.random.choice(allv, len(allv), replace=False )
  
    for v in allv_sample:
      val_best = -100
      t_best = -1
      b_best=-1
      deltaS_best=-1

      for b_orig in set_comm:        
        #restr_IDelta = [d for d in IDelta if d <= Sov[v,b_orig] ]
        restr_IDelta = IDelta[IDelta <= Sov[v,b_orig]]        

        if len(restr_IDelta)>0:
          for deltaS in restr_IDelta:            
            for t in set_comm[set_comm!=b_orig]:                            
              val = sf.calc_gainQ_ov_Rgroup_update(R_group_pos, R_group_neg, t, b_orig, deltaS, ri_vert_pos, ri_vert_neg, v)
              #print("** Move vertex: v="+str(v)+", b_orig"+str(b_orig)+", DeltaS="+str(deltaS)+", t="+str(t),", val="+str(val))
              if val > val_best or t_best==-1:
                val_best = val
                t_best = t
                b_best = b_orig
                deltaS_best = deltaS

      if t_best >= 0:
        bool_continue=True
        R_group_pos, R_group_neg = sf.move_vertex_ov_R_group(R_group_pos, R_group_neg, t_best, b_best, deltaS_best, ri_vert_pos, ri_vert_neg, v)
        #print(Sov[v,:])
        Sov[v,b_best] -= deltaS_best
        Sov[v,t_best] += deltaS_best
        
        # if(deltaS_best>0.1):
        #   print("* Move " + str(v) + " to community " + str(t_best)+", delta="+str(deltaS_best))
          
        if bool_verbose==True:
          print("* Move " + str(v) + " to community " + str(t_best)+", delta="+str(deltaS_best))
          #print(Sov[v,:])
      
    it_ov+=1
  
  #print("it_ov=",it_ov)
  valQ = sf.calc_Q_Rgroup(R_group_pos, R_group_neg, 2*np.sum(g.strength(weights='weight'))/2)

  return R_group_pos, R_group_neg, pcurr, Sov, valQ


def spec_post_processing(Sov, ov_thr):
  Sov[(Sov>0) & (Sov<ov_thr)]=0
  
  return Sov

#%% Simple LS alg
# Overlapping distributed

def spec_ov_alg(th_num, res_queue, g, comms_est, ri_vert_pos, ri_vert_neg, eig_pos_ind, eig_neg_ind, bool_ov, ov_thr, bool_verbose, out_verbose,opDeltaH=0):  
  #Thread
  np.random.seed()
  
  R_group_pos = np.zeros(( comms_est, len(eig_pos_ind) ))
  R_group_neg = np.zeros(( comms_est, len(eig_neg_ind) ))
  
  #initial disjoint partition
  #debug print("---- Initial partition, (" + str(th_num) + ")")
  R_group_pos, R_group_neg, Sov, pcurr, valQ = sf.spec_initial_partition(g, R_group_pos, R_group_neg, ri_vert_pos, ri_vert_neg, eig_pos_ind, eig_neg_ind, comms_est, bool_verbose=False) #bool_verbose)
  
  
  #debug  print("Initial: CalcQ="+str(valQ) + ", (" + str(th_num) + ")")
  if bool_verbose==True:
    print(pcurr)

  #Local search
  #debug  print("---- Local search, ("+ str(th_num) + ")")
  R_group_pos, R_group_neg, pcurr, Sov, valQ = sf.spec_local_search(g, ri_vert_pos, ri_vert_neg, eig_pos_ind, eig_neg_ind, comms_est, R_group_pos, R_group_neg, Sov, pcurr, it_max=10, bool_verbose=False)  
  
  #debug print("LS: CalcQ="+str(valQ) + ", (" + str(th_num) + ")")
  if bool_verbose==True:
    print(pcurr)
  
 
  #Overlapping
  if bool_ov==True:
    #debug  print("---- Start overlapping, (" + str(th_num) + ")")
    R_group_pos, R_group_neg, pcurr, Sov, valQ = spec_overlapping(g, ri_vert_pos, ri_vert_neg, eig_pos_ind, eig_neg_ind, comms_est, R_group_pos, R_group_neg, Sov, pcurr, it_max=10, bool_verbose=False,opDeltaH=opDeltaH)  
    #debug print("OV: CalcQ="+str(valQ) + ", (" + str(th_num) + ")")
    
    if bool_verbose==True:
      print(Sov)
    #Post processing
    Sov = spec_post_processing(Sov, ov_thr)

    #print("ov ok ", str(th_num) )
    
  res_queue.put([th_num, valQ, Sov, R_group_pos, R_group_neg])


#%% new SpecOV - SPARSIFY

#%% SpecOV - main


#%%% SpecOV - multi

def spec_ov_main(file_graph, nex=10, plarg=0.3, gamma1=1, gamma2=0, gamma3=0, ov_thr=0.5, bool_ov=True, file_out=None, bool_verbose=False, out_verbose=[],KNumber_pre=-1, mod_orig=True, T=None, opDeltaH=0):
    g = sf.read_construct_g(file_graph)
    
    print("--- IDelta: ", opDeltaH) #231220

    if mod_orig==False:    
        #S= sf.construct_cosine(g)        
        g,mod_mat,A,P = sf.construct_modularity_cosine(g, gamma1, gamma2, gamma3,T=T)     
        #print(mod_mat)
        mod_mat = mod_mat.transpose()
    else:
        print("MOD original")
        print("gamma1=",gamma1,",2=",gamma2)
        mod_mat,A,P = sf.construct_modularity_generalized(g, gamma1, gamma2)
        mod_mat = mod_mat.transpose()

    eig_val, eig_vec, eig_pos_ind, eig_neg_ind, comms_est, ri_vert_pos, ri_vert_neg = spec_ov_eigen_dec(g, mod_mat, plarg)  

    if KNumber_pre > 0:
        comms_est = KNumber_pre
     
    valQ_max = -100
    nmi_max = -2
    
    threads = []  
    manager = multiprocessing.Manager()
    res_queue = manager.Queue()
    
    for it in range(0,nex):
        #print("---------------" + str(it))
        t = multiprocessing.Process(target=spec_ov_alg,args=[it, res_queue, g, comms_est, ri_vert_pos, ri_vert_neg, eig_pos_ind, eig_neg_ind, bool_ov, ov_thr, bool_verbose, out_verbose, opDeltaH])
        threads.append(t)
        t.start()
        #threads[it].start()
        if bool_verbose_thread == True:
            print("thread started..." , it)
        #t.start()
    
  
    q_best = -2
    for it in range(0,nex):
        #threads[it].terminate()
        threads[it].join()
        threads[it].terminate()
        res_t = res_queue.get()
        print("thread finished...exiting" + ": " + str(res_t[0]) + ", Q=" + str(res_t[1]))
        
        if res_t[1] > q_best:
            q_best = res_t[1]
            Sov_best = res_t[2]
            R_group_pos = res_t[3]
            R_group_neg = res_t[4]
    
    
    print("best Q = ", q_best)

    return q_best, Sov_best, g, eig_val, ri_vert_pos, ri_vert_neg, R_group_pos, R_group_neg, mod_mat,A,P


#%%% SpecOV - seq

def spec_ov_main_single(file_graph, nex, plarg, gamma1, gamma2, ov_thr, bool_ov, file_out="com_out", bool_verbose=False, out_verbose=[]):
    g = sf.read_construct_g(file_graph)
    mod_mat = sf.construct_modularity_generalized(g,gamma1,gamma2).transpose()
    eig_val, eig_vec, eig_pos_ind, eig_neg_ind, comms_est, ri_vert_pos, ri_vert_neg = spec_ov_eigen_dec(g, mod_mat, plarg)  
           
    valQ_max = -100
    nmi_max = -2
        
    manager = multiprocessing.Manager()
    res_queue = manager.Queue()
    
    for it in range(0,nex):
        #print("---------------" + str(it))
        #t = threading.Thread(target=spec_ov_alg,args=[it, res_queue, g, comms_est, ri_vert_pos, ri_vert_neg, eig_pos_ind, eig_neg_ind, bool_ov, ov_thr, bool_verbose, out_verbose])
        spec_ov_alg(it, res_queue, g, comms_est, ri_vert_pos, ri_vert_neg, eig_pos_ind, eig_neg_ind, bool_ov, ov_thr, bool_verbose, out_verbose)
        
    q_best = -2
    for it in range(0,nex):        
        res_t = res_queue.get()        
        if res_t[1] > q_best:
            q_best = res_t[1]
            Sov_best = res_t[2]
            R_group_pos = res_t[3]
            R_group_neg = res_t[4]
    
    
    print("best Q = ", q_best)
    #Sov  (vert) x (communities)
    print_com_lines_file(file_out, Sov_best)
    #ri_vert_pos, ri_vert_neg --> plot: does not change depending on the algorithm.
    
    #TODO: find the best solution.. 
    #write in file
    #each line is a different community
   
    return q_best, Sov_best, g, eig_val, ri_vert_pos, ri_vert_neg
#% SpecOV: Call main

# def call_main(file_graph, plarg, bool_ov):
#     #TODO: input params
#     print("TODO implement")


#%% plot 

  #Plot
  #cont teoric_eigenvalues(g, mod_mat, plarg):
def plot_vertex_vectors(g, rp, rn, Rp, Rn, Sov,op="vectors", list_sol=None,file_out=None,labels=None): #,
    figure, ax = plt.subplots(figsize=(10, 10))
    
    #print("rp---")
    #print(rp)
    #draw_circle = plt.Circle((0, 0), thr)
    #ax.set_xlim((-max(rp[:,0]), max(rp[:,0])))
    #ax.set_ylim((-max(rp[:,1]), max(rp[:,1])))
       
    # ax.set_xlim((-1,1))
    # ax.set_ylim((-1,1))

    
    #draw_circle = plt.Circle((0, 0), thr, fill=False)
    
    #colors=['blue', 'red', 'green','pink', 'orange','violet','yellow','gray','magenta','cyan','black', 'blue', 'red', 'green','black', 'blue', 'red', 'green',
     #       'blue', 'red', 'green','pink', 'orange','violet','yellow','gray','magenta','cyan','black', 'blue', 'red', 'green','black', 'blue', 'red', 'green']
    
    colors = cm.rainbow(np.linspace(0, 1, rp.shape[0] ))
    
    if op=="vectors":
        scale=5
        max_Sov= max( abs(np.max(Rp)) , abs(np.min(Rp)) )/scale + 0.1
    
        ax.set_xlim((-max_Sov,max_Sov))
        ax.set_ylim((-max_Sov,max_Sov))
    
        for i in np.arange(0,g.vcount()):
            #print("vertex",i)
            com_i =np.where(Sov[i,:]>0)[0][0]
            #com_i =np.where(Sov[i,:]>0)[1][0]
            #print("com_i",com_i)
        #ax.arrow(np.ones(g.vcount()), np.ones(g.vcount()), rp[:,0], rp[:,1], label="Vertex vectors")      
           #print(colors[com_i])
            #ax.arrow(0, 0, rp[i,0], rp[i,1], color = colors[k],width=0.0005, head_width=0.01)
            ax.annotate("", xy = (0,0), xytext = (rp[i,0], rp[i,1]), arrowprops = dict(arrowstyle="<-", shrinkA = 0, shrinkB = 0, linestyle = '--', mutation_scale = 30, color = colors[com_i],))
            
            #if list_sol is None:
            ax.text(rp[i,0]+0.015, rp[i,1]+0.015, i+1 ,fontsize=10)
            #elif i in list_sol:
            #ax.text(rp[i,0]+0.015, rp[i,1]+0.015, i+1 ,fontsize=10)
            
            #com_ov=np.where(Sov[i,:]>0 & Sov[i,:]<1 )[0][0]
            
            com_ov = np.where( np.logical_and(Sov[i,:]>0 , Sov[i,:]<1 )==True)[0]
            #Out[49]: array([0, 1])
            print("com_ov")
            print(com_ov)
  
            #Sov[i, np.where( np.logical_and(Sov[i,:]>=0 , Sov[i,:]<1.1 )==True)[0] ]
            #Out[50]: array([1., 0.])
      
            print(com_ov)
      
        for k in np.arange(0,np.shape(Sov)[1]):   
            #print(Rp[k,:])
            print(k)
            ax.arrow(0, 0, Rp[k,0]/scale, Rp[k,1]/scale, color = colors[k], width=0.015,head_width=0.03)      
            ax.text(Rp[k,0]/scale+0.015, Rp[k,1]/scale+0.015, i+1 ,fontsize=12, color= colors[k])
            
    elif op=="emb":
        print("emb_plot")        
        max_Sov= max( abs(np.max(rp)) , abs(np.min(rp)) ) #+ 0.1 #300321
        print(max_Sov)
        #ax.set_xlim((-max_Sov,max_Sov))
        #ax.set_ylim((-max_Sov,max_Sov))
        ax.set_xlim((np.min(rp),np.max(rp)))
        ax.set_ylim((np.min(rp),np.max(rp)))
        
        
        #ax.scatter(x=rp[:,0], y=rp[:,1] ) #, label="Vertex vectors")      
        #rp = np.matrix(rp)
        for i in np.arange(0,g.vcount()):
            #TODO: GET MAX
            if Sov is not None:                            
                com_i =np.where(Sov[i,:]>0)[0][0]
            else: 
                com_i=0
            
            #print("----------------")
            
            #print(com_i)
            ax.scatter(x=rp[i,0], y=rp[i,1], color=colors[com_i]) #, label="Vertex vectors")       #s=10 ,
            #ax.annotate("", xy = (0,0), xytext = (rp[i,0], rp[i,1]), arrowprops = dict(arrowstyle="<-", shrinkA = 0, shrinkB = 0, linestyle = '--', mutation_scale = 30, color = colors[com_i],))        
            if labels is None:
                if list_sol is None:                
                    ax.text(rp[i,0]+0.015, rp[i,1]+0.015, i+1 ,fontsize=10) #10
                elif i in list_sol:
                    ax.text(rp[i,0]+0.015, rp[i,1]+0.015, i+1 ,fontsize=10) #10
            #else:
                #print(labels)
                ##ax.text(rp[i,0]+0.015, rp[i,1]+0.015, labels[i] ,fontsize=10) #10

            
    #ax.scatter([-thr,thr], [0,0], label="Threshold", color="red", marker="s")  
    #ax.set_aspect(1)
    #ax.add_artist(draw_circle)
    #plt.title('')S
    if file_out is None:
        plt.savefig('plot' + op + '.png')
    else: 
        print("plot in",file_out)
        plt.savefig(file_out)
        
    plt.legend()
    plt.show()

#plot_vertex_vectors(g, ri_vert_pos, ri_vert_neg, Rp, Rn, Sov, op="vectors")

# gc = g
# ri_vert_posc = ri_vert_pos
# ri_vert_negc = ri_vert_neg
# Sovc = Sov

#%% Call main - Example
def call_spec_ov_example():
    #Default parameters:    
    output_time="time_aux.txt"
    net_type="real"
    plarg=1
    gamma1=1
    gamma2=1
    bool_ov=True
    ov_thr=0
    bool_verbose=False
    out_verbose=[]
        
    #Karate    
    base_folder="../grafos_nao_direcionados/"
    #file_graph=base_folder+"karate.net"
    file_graph=base_folder+"dolphins.paj"
  
    print(file_graph)
    base_folder_out = "../SpecDecOv/grafos_nao_direcionados/lines_grafos_nao_direcionados_1/"    
    file_com_out = base_folder_out+"lines_karate_ov.com"

    os.makedirs(base_folder_out, exist_ok=True)
    
    start_time = time.perf_counter()
    
    q_best, Sov, g, eig_val, ri_vert_pos, ri_vert_neg, Rp, Rn,mod_mat,A,P = spec_ov_main(file_graph, 10, plarg, gamma1, gamma2, ov_thr, bool_ov, file_com_out, bool_verbose, out_verbose)
    #q_best, Sov = spec_ov_main_single(file_graph, 4, plarg, gamma1, gamma2, ov_thr, bool_ov, bool_verbose, out_verbose)
    print("--- %s seconds ---" , (time.perf_counter() - start_time))
          
    return q_best, Sov, g, eig_val, ri_vert_pos, ri_vert_neg, Rp, Rn

#q_best, Sov, g, eig_val, ri_vert_pos, ri_vert_neg, Rp, Rn = call_spec_ov_example()
#plot_vertex_vectors(g, ri_vert_pos, ri_vert_neg, Rp, Rn, Sov, op="vectors")

#%% Call main - LFR
def call_spec_ov_LFR(lfr_set="L"):
    #Default parameters:    
    output_time="time_aux.txt"
    net_type="real"
    #plarg=0.95
    plarg=0.3
    gamma1=1
    gamma2=1
    bool_ov=True
    #ov_thr=0.5
    ov_thr=0.1
    bool_verbose=False
    out_verbose=[]
    
    folder_res="results_Tese_2021_3"
    nome_alg = "SpecDecOV"             
    #LFR
    lfr_type="Binary_DirNormal2_OV_2017" #Binary2017_DirNormal2_OV_2017

    tam = 1000
    
    max_it=4
    # for mu in range(1,9):
    #     for n2 in range(1,6):  
    ex=1        
    for mu in range(1,9):
        for n2 in range(1,6):
            inst_base =  lfr_type +"_Lancichinetti2011_"+ lfr_set 
            base_folder_in="/home/camila/not_sync/Lancichinetti2017/" + inst_base + "/paj_" + lfr_type + "_" + lfr_set + "/"
            inst_name = "net_" + lfr_type + "_" + lfr_set + str(tam) + "_" + str(mu) + "_" + str(n2)
            file_graph=base_folder_in + inst_name + ".net"

            #base_folder_exp="/home/camila/not_sync/Lancichinetti2017/" + lfr_type +"_Lancichinetti2011_"+ lfr_set + "/com_" + lfr_type + "_" + lfr_set + "/"
            #file_exp=base_folder_in+"net_" + lfr_type + "_" + lfr_set + str(tam) + "_" + str(mu) + "_" + str(n2) + ".net"
        
            base_folder_out = folder_res+"/"+nome_alg + "/" + inst_base  +"/lines_"+ inst_base + "_" + str(ex)+ "/"
            file_com_out = base_folder_out+"lines_" + inst_name + "_ov.com"
            file_time_out = base_folder_out+"time_" + inst_name + "_ov.txt"
            file_k_num_out = base_folder_out+"num_" + inst_name + "_ov.txt"
         
            print(file_graph)
            print(base_folder_out)
            
            os.makedirs(base_folder_out, exist_ok=True)
    
            start_time = time.perf_counter()
            q_best, Sov,g, eig_val, ri_vert_pos, ri_vert_neg, Rp, Rn = spec_ov_main(file_graph, max_it, plarg, gamma1, gamma2, ov_thr, bool_ov, file_com_out, bool_verbose, out_verbose)
            #q_best, Sov = spec_ov_main_single(file_graph, 4, plarg, gamma1, gamma2, ov_thr, bool_ov, bool_verbose, out_verbose)
            time_tot = time.perf_counter() - start_time
            
            print_com_lines_file(file_com_out, Sov) #231220            
            print("--- %s seconds ---" , time_tot)
            f = open(file_time_out, "w")                
            f.write(str(time_tot))
            f.close() 
            
            idx= np.where( np.all(Sov[..., :] == 0, axis=0) == False )[0]
            Sov_nonzero = Sov[:,idx]
            k_num_best = Sov_nonzero.shape[1]

            f = open(file_k_num_out, "w")                
            f.write(str( k_num_best ))
            f.close() 
            
                
            #TODO: save time
    
    return q_best, Sov,g, eig_val, ri_vert_pos, ri_vert_neg, Rp, Rn


# Call 
if __name__ == "__main__":
    #q_best, Sov,g, eig_val, ri_vert_pos, ri_vert_neg, Rp, Rn = call_spec_ov_LFR(lfr_set = "L")
    q_best, Sov,g, eig_val, ri_vert_pos, ri_vert_neg, Rp, Rn = call_spec_ov_LFR(lfr_set = "S")
#    plot_vertex_vectors(g, ri_vert_pos, ri_vert_neg, Sov, op="vectors")
    

#TODO


#base_folder="/home/camila/not_sync/Lancichinetti2017/Binary_2017_Lancichinetti2011_S/paj_Binary_2017_S/"
#base_expected="/home/camila/not_sync/Lancichinetti2017/Binary_2017_Lancichinetti2011_S/com_Binary_2017_S/"



#%% X  Call main - Example - SPARSIFY
def call_spec_ov_example_sparsify():
    #Default parameters:    
    output_time="time_aux.txt"
    net_type="real"
    plarg=0.2
    gamma1=1
    gamma2=1
    bool_ov=True
    ov_thr=0
    bool_verbose=False
    out_verbose=[]
        
    #Karate    
    base_folder="grafos_nao_direcionados/"
    #file_graph=base_folder+"karate.net"
    file_graph=base_folder+"dolphins.paj"
  
    print(file_graph)
    base_folder_out = "SpecDecOv/grafos_nao_direcionados/lines_grafos_nao_direcionados_1/"    
    file_com_out = base_folder_out+"lines_karate_ov.com"

    os.makedirs(base_folder_out, exist_ok=True)
    
    start_time = time.perf_counter()
    q_best, Sov, g, eig_val, ri_vert_pos, ri_vert_neg, Rp, Rn = spec_ov_main_sparsify(file_graph, 10, plarg, gamma1, gamma2, ov_thr, bool_ov, file_com_out, bool_verbose, out_verbose)
    #q_best, Sov = spec_ov_main_single(file_graph, 4, plarg, gamma1, gamma2, ov_thr, bool_ov, bool_verbose, out_verbose)
    print("--- %s seconds ---" , (time.perf_counter() - start_time))
          
    return q_best, Sov, g, eig_val, ri_vert_pos, ri_vert_neg, Rp, Rn

# if __name__ == "__main__":
#     q_best, Sov,g, eig_val, ri_vert_pos, ri_vert_neg, Rp, Rn = call_spec_ov_example_sparsify()
    

#%% X call LFR sparsify 


def call_spec_ov_LFR_sparsify():
    #Default parameters:    
    output_time="time_aux.txt"
    net_type="real"
    #plarg=0.95
    plarg=0.3
    gamma1=1
    gamma2=1
    bool_ov=True
    #ov_thr=0.5
    ov_thr=0.1
    bool_verbose=False
    out_verbose=[]
    
    folder_res="results_Tese_2021"
    nome_alg = "SpecDecOV"             
    #LFR
    lfr_type="Binary_DirNormal2_OV_2017" #Binary2017_DirNormal2_OV_2017
    lfr_set = "S"
    tam = 1000
    # for mu in range(1,9):
    #     for n2 in range(1,6):  
    ex=1        
    for mu in range(1,9):
        for n2 in range(1,6):
            inst_base =  lfr_type +"_Lancichinetti2011_"+ lfr_set 
            base_folder_in="/home/camila/not_sync/Lancichinetti2017/" + inst_base + "/paj_" + lfr_type + "_" + lfr_set + "/"
            inst_name = lfr_type + "_" + lfr_set + str(tam) + "_" + str(mu) + "_" + str(n2)
            file_graph=base_folder_in + "net_" + inst_name + ".net"

            #base_folder_exp="/home/camila/not_sync/Lancichinetti2017/" + lfr_type +"_Lancichinetti2011_"+ lfr_set + "/com_" + lfr_type + "_" + lfr_set + "/"
            #file_exp=base_folder_in+"net_" + lfr_type + "_" + lfr_set + str(tam) + "_" + str(mu) + "_" + str(n2) + ".net"
        
            base_folder_out = folder_res+"/"+nome_alg + "/" + inst_base  +"/lines_"+ inst_base + "_" + str(ex)+ "/"
            file_com_out = base_folder_out+"lines_" + inst_name + "_ov.com"
          
            print(file_graph)
            print(base_folder_out)
            
            os.makedirs(base_folder_out, exist_ok=True)
    
            start_time = time.perf_counter()
            q_best, Sov,g, eig_val, ri_vert_pos, ri_vert_neg, Rp, Rn = spec_ov_main(file_graph, 4, plarg, gamma1, gamma2, ov_thr, bool_ov, file_com_out, bool_verbose, out_verbose)
            #q_best, Sov = spec_ov_main_single(file_graph, 4, plarg, gamma1, gamma2, ov_thr, bool_ov, bool_verbose, out_verbose)
            print("--- %s seconds ---" , (time.perf_counter() - start_time))
    
            #TODO: save time
            #TODO: must calculate with oNMI
            #base_folder_expected="/home/camila/not_sync/Lancichinetti2017/" + inst_base + "/com_" + lfr_type + "_" + lfr_set + "/"
            #file_expected=base_folder_expected+"com_"+inst_name+".dat"  
            #p_expected = sf.read_partition_expected(file_expected)
            #sf.calc_nmi(p_best[0], p_expected)
      
    return q_best, Sov,g, eig_val, ri_vert_pos, ri_vert_neg, Rp, Rn


# # Call 
# if __name__ == "__main__":
#     q_best, Sov,g, eig_val, ri_vert_pos, ri_vert_neg, Rp, Rn = call_spec_ov_LFR_sparsify()
# ##    plot_vertex_vectors(g, ri_vert_pos, ri_vert_neg, Sov, op="vectors")
    
