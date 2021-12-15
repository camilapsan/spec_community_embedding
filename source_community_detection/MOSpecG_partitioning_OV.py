#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  4 19:57:13 2020

@author: camila
"""

#%% Import


import tensorflow as tf
#tf.enable_eager_execution()

import igraph
import numpy as np
import matplotlib.pyplot as plt
import math
import scipy.sparse.linalg
import random

from pathlib import Path
import sys
path = str(Path(Path(__file__).parent.absolute()).parent.absolute())
sys.path.insert(0, path)
sys.path.insert(0, path + '/source_community_detection/')
sys.path.insert(0, path + '/source_community_embedding/')
sys.path.insert(0, path + '/source_graph_util/')

import SpecOV_partitioning_v2 as spec_ov
#import threading, 
import copy

import spectral_functions as sf


import os

#from multiprocessing import Process
import multiprocessing
import queue

import time

#%% Defines

bool_verbose=False

bool_verbose_thread =False
    
#%% Classes with variables

class Pop:    
    part=None
    Q=None
	
    RGroup_pos = None
    RGroup_neg = None
    
    Sov = None
    
# class Offs:
#     RGroup_pos = None
#     RGroup_neg = None
    	    
#     part=None   
#     Q=None
    
    
class Params:
    num_pareto = None    
    n_gen = None
    n_pop = None
    n_pop_off = None
    p_offs = None
    #it_max_ls_gen = 4 #2 #5
    it_max_ls_ini = 2#290921 1 #2 #5 #260721
    it_max_ls_gen = 3#290921 1 #2 #5 #260721
    it_max_ls_final = 5#290921 #4  #10
    ov_thr=0.1

class Ensemble: 
    pareto_parts = None    
    p_thr = None
    DEF_SEL_PARETO=0.8
    consensus = None
    
class Graph:
    g = None
    KNumber = None
    
    ri_vert_pos = None
    ri_vert_neg = None
    
class Eig:
    val = None
    vec = None
    pos_ind = None
    neg_ind = None
    
    

    

#%% Initialize

def initialize_params(g, num_pareto, ensemble_thr, n_gen, n_pop, p_offs,ov_thr=0):
    params = Params()
    
    params.num_pareto = num_pareto
    
    ensemble = Ensemble()
    ensemble.pareto_parts = np.zeros(( num_pareto, g.vcount()))
    ensemble.p_thr = ensemble_thr
    
    params.n_gen = n_gen
    params.n_pop = n_pop
    # params.n_pop_off = n_pop + 1
    params.n_pop_off = n_pop
    params.p_offs = p_offs
    
    params.ov_thr=ov_thr
     
    #return pareto_parts, n_pop_off, pop_part, pop_Q, offs_part, offs_Q
    return params, ensemble
    
def initialize_pop_arrays(g, n_pop_off, KNumber, eig_pos_ind, eig_neg_ind ):
    pop = Pop()    
    pop.part = np.zeros((n_pop_off, g.vcount()),dtype=int)
    pop.Q = np.zeros(n_pop_off)		    

    pop.RGroup_pos = np.zeros(( n_pop_off, KNumber,len(eig_pos_ind)))
    pop.RGroup_neg = np.zeros(( n_pop_off, KNumber,len(eig_neg_ind)))

    pop.Sov = np.zeros(( n_pop_off, g.vcount(), KNumber)) #201220
    #print(pop.Sov)
    
  
    offs = Pop()
    offs.part = np.zeros((n_pop_off, g.vcount()),dtype=int)    
    offs.Q = np.zeros(n_pop_off)
    
    offs.Sov = np.zeros(( n_pop_off, g.vcount(), KNumber)) #201220
    
    offs.RGroup_pos =  np.zeros((n_pop_off, KNumber,len(eig_pos_ind)))
    offs.RGroup_neg =  np.zeros((n_pop_off, KNumber,len(eig_neg_ind)))
    
    return pop, offs

#%% print file

def print_com_part_Sov(file_out, Sov):
    f = open(file_out, "w")    

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

def print_com_part(file_com_out, part):
    f = open(file_com_out, "w")    

    # v_num = len(part)
    # k_num = len(np.unique(part))
    
    #for v in range(0, v_num):                
    np.savetxt(f, part, newline='\n', fmt='%i')

    f.close()    
#%% Aux 

def print_ind(graph, pop_off, ind):
    print("Ind: "+ str(ind))
    
    conf_Q = sf.calc_Q_Rgroup(pop_off.RGroup_pos[ind], pop_off.RGroup_neg[ind], 2* np.sum(graph.g.strength(weights='weight'))/2) #121021
    conf_Q_vec = graph.g.modularity(pop_off.part[ind],  graph.g.es['weight'])
    print("Q:"+ str(conf_Q) + ", Q_pop=" + str(pop_off.Q[ind])+", Q_igraph="+str(conf_Q_vec))
    
    #print("pop_part:")


def move_vert(graph, eig, offs, ind_offs, v, ld, lold):
    # print("v="+ str(v))

    #print("offs_part:")
    #print(offs.part[ind_offs])
    
    offs.part[ind_offs,v]=ld
    
    #201220 TODO: c
    #always move everything?    
    #print("move ", offs.Sov[ind_offs,v,lold],"\n")
    
    #300321
    #if offs.Sov[ind_offs,v,lold] == 0: 
    #    print("move 0 offs")
        
    deltaS = offs.Sov[ind_offs,v,lold]
    offs.Sov[ind_offs,v,ld] += deltaS
    offs.Sov[ind_offs,v,lold]=0
  
    #offs.RGroup_pos[ind_offs], offs.RGroup_neg[ind_offs] = sf.move_vertex_R_group(offs.RGroup_pos[ind_offs], offs.RGroup_neg[ind_offs], ld, lold, graph.ri_vert_pos,  graph.ri_vert_neg, v) #todo: return new
    offs.RGroup_pos[ind_offs], offs.RGroup_neg[ind_offs] = sf.move_vertex_ov_R_group(offs.RGroup_pos[ind_offs], offs.RGroup_neg[ind_offs], ld, lold, deltaS, graph.ri_vert_pos,  graph.ri_vert_neg, v) #231220 
    
    # offs.RGroup_pos[ind_offs][ls] -= eig.ri_vert_pos[v]
    # offs.RGroup_neg[ind_offs][ls] -= eig.ri_vert_neg[v]
    
    # offs.RGroup_pos[ind_offs][ld] += eig.ri_vert_pos[v]
    # offs.RGroup_neg[ind_offs][ld] += eig.ri_vert_neg[v]

    #deltaQ = sf.calc_gainQ_Rgroup_move(offs.RGroup_pos[ind_offs], offs.RGroup_neg[ind_offs], ld, lold, graph.ri_vert_pos, graph.ri_vert_neg, v)
    deltaQ = sf.calc_gainQ_ov_Rgroup_update(offs.RGroup_pos[ind_offs], offs.RGroup_neg[ind_offs], ld, lold, deltaS, graph.ri_vert_pos, graph.ri_vert_neg, v)        
    
    # print("deltaQ="+str(deltaQ)) 
    return deltaQ

#%% Threads aux
    
def th_get_res_queue_pop(res_queue, ind, pop):
    
    res_t = res_queue.get() #False? not wait

    th_ind = res_t[0]
    pop.RGroup_pos[ind] = res_t[1]
    pop.RGroup_neg[ind] = res_t[2]
    pop.Sov[ind] = res_t[3] #201220
    pop.part[ind] = res_t[4]
    pop.Q[ind] = res_t[5]
    
    return pop, th_ind, res_queue 

#%% Crossover

def print_details_crossover(pop, offs, pSourceQ2, pDestQ1, ld, ls, ind_offs):
    print("TODO: print detail in files")
    print("crossover; " + str(ind_offs) + "\tls; " + str(ls) + "\tld; " + str(ld))    
    print("crossover; " + str(ind_offs) + ";\tsource ;" + str(pSourceQ2)+  ";\tQ=" + str(pop.Q[pSourceQ2]) + ";");		
    print(pop.part[pSourceQ2])
    print("crossover; " + str(ind_offs) + ";\tdest ;" + str(pDestQ1)+  ";\tQ=" + str(pop.Q[pDestQ1]) + ";");		
    print(pop.part[pDestQ1])
    print("crossover; " + str(ind_offs) + ";\toffspring ;" + str(ind_offs)+  ";\tQ=" + str(offs.Q[ind_offs]) + ";");		
    print(offs.part[ind_offs])


def copy_rgroup_pop(offs, ind_offs, pop, pDestQ1):
    # print("ind_offs: "+str(ind_offs))
    # print("pDestQ1:" + str(pDestQ1))
    offs.Q[ind_offs] = pop.Q[pDestQ1]
    offs.part[ind_offs] = pop.part[pDestQ1].copy()
    
    offs.RGroup_pos[ind_offs] = pop.RGroup_pos[pDestQ1].copy()
    offs.RGroup_neg[ind_offs] = pop.RGroup_neg[pDestQ1].copy()
    
    
    offs.Sov[ind_offs] = pop.Sov[pDestQ1].copy() #201220
    
    return offs

def select_crossover_dest_cluster(graph, pop, pSourceQ2, pDestQ1, ls):
    val_best = -2
    s_best = -1
    
    for s in np.arange(0, graph.KNumber,1):
        val_pos = np.inner(pop.RGroup_pos[pSourceQ2,ls], pop.RGroup_pos[pDestQ1,s])
        val_neg = np.inner(pop.RGroup_neg[pSourceQ2,ls], pop.RGroup_neg[pDestQ1,s])
        val = val_pos + val_neg
        
        if s_best == -1 or val > val_best:
            val_best = val
            s_best = s
       #print("s="+str(s)+ ", cos_pos")    
    return s_best


def crossover_offspring(graph, eig, params, pop, offs,g): #not used
    sumQ=0
    deltaQ=0
    
    #ind_prob = np.zeros(params.n_pop)
    ind_prob = pop.Q + 1
         
    for ind_offs in np.arange(0, params.n_pop,1):
        #roulete whell
        #sumQ = sum(pop.Q) + params.n_pop
        sumQ = sum(ind_prob)
        ind_prob = ind_prob/sumQ

        # print("ind_prob")
        # print(ind_prob)        
        pSourceQ2,pDestQ1 = np.random.choice(np.arange(0,params.n_pop,1), 2,p=ind_prob, replace=False)
        
        # print("offs_part b")
        # print(offs.part)
        # #offs[indoffs] receives a copy of pDestQ1
        offs = copy_rgroup_pop(offs, ind_offs, pop, pDestQ1)
        
        # print("offs_part a")
        # print(offs.part)
        
        #one-way crossover
		#source:pSourceQ2, dest:pDestQ1
		#update only if new dest is better than source
        vi = np.random.choice( np.arange(0,graph.g.vcount() ), 1 , replace=False )[0]
        #C++ was implemented using the sampling strategy: it did not work.. try to implement coarsening first
        
        ls = pop.part[pSourceQ2,vi]        
        #TODO: select from Sov...
        
        ld = select_crossover_dest_cluster(graph, pop, pSourceQ2, pDestQ1, ls)  #TODO
    
        if bool_verbose == True:            
            print("... vi="+str(vi) + ", ls="+str(ls)+ ", ld="+str(ld))

        #v in community ls and not in community ld
        v_ls = np.where(pop.part[pSourceQ2] == ls)
        v_not_ld = np.where(pop.part[pDestQ1] != ld)
        v_move = np.intersect1d(v_ls, v_not_ld)
        
        for i in v_move:
            lold = pop.part[pDestQ1,i]
            # print("offs_part:")
            # print(offs.part)
            # print("pop_part:")
            # print(pop.part)
            #deltaQ += move_vert(graph, eig, offs, ind_offs, i, ld, lold)
            offs.Q[ind_offs] = move_vert(graph, eig, offs, ind_offs, i, ld, lold) #181120
            #print(deltaQ)
            offs.Q[ind_offs]
        
        if bool_verbose == True:
            print_ind(graph, offs, ind_offs)
            
    return offs            
        
def crossover_offspring_ind_th(res_queue, graph, eig, params, pop, offs, ind_offs, ind_prob, g):
    np.random.seed()
    #roulete whell
    #sumQ = sum(pop.Q) + params.n_pop
    sumQ = sum(ind_prob)
    ind_prob = ind_prob/sumQ

    # print("ind_prob")
    # print(ind_prob)        
    pSourceQ2,pDestQ1 = np.random.choice(np.arange(0,params.n_pop,1), 2,p=ind_prob, replace=False)
    offs = copy_rgroup_pop(offs, ind_offs, pop, pDestQ1)
    
    # print("offs_part a")
    # print(offs.part)
    
    #one-way crossover
		#source:pSourceQ2, dest:pDestQ1
		#update only if new dest is better than source
    vi = np.random.choice( np.arange(0,graph.g.vcount() ), 1 , replace=False )[0]
    #C++ was implemented using the sampling strategy: it did not work.. try to implement coarsening first
    
    ls = pop.part[pSourceQ2,vi]        
    ld = select_crossover_dest_cluster(graph, pop, pSourceQ2, pDestQ1, ls)  #TODO

    if bool_verbose == True:            
        print("... vi="+str(vi) + ", ls="+str(ls)+ ", ld="+str(ld))

    #v in community ls and not in community ld
    v_ls = np.where(pop.part[pSourceQ2] == ls)
    v_not_ld = np.where(pop.part[pDestQ1] != ld)
    v_move = np.intersect1d(v_ls, v_not_ld)
    
    for i in v_move:
        lold = pop.part[pDestQ1,i]
        # print("offs_part:")
        # print(offs.part)
        # print("pop_part:")
        # print(pop.part)
        offs.Q[ind_offs] = move_vert(graph, eig, offs, ind_offs, i, ld, lold) #181120
        #print(deltaQ)
        
    
    if bool_verbose == True:
        print_ind(graph, offs, ind_offs)  
    
    #res_queue.put([ind_offs, offs])    
    res_queue.put([ind_offs,pop.RGroup_pos[ind_offs], pop.RGroup_neg[ind_offs],  pop.Sov[ind_offs], pop.part[ind_offs], pop.Q[ind_offs]])
        
    
def crossover_offspring_offs_call(graph, eig, params, pop, offs,g):
    sumQ=0
    deltaQ=0
    
    #ind_prob = np.zeros(params.n_pop)
    ind_prob = pop.Q + 1
         
    threads = []
    manager = multiprocessing.Manager()
    res_queue = manager.Queue()

    for ind_offs in np.arange(0, params.n_pop,1):
        t = multiprocessing.Process(target=crossover_offspring_ind_th,args=[res_queue, graph, eig, params, pop, offs, ind_offs, ind_prob, g])
        threads.append(t)
        t.start()
        if bool_verbose_thread == True:
            print("Crossover: thread started..."+ str(ind_offs))        
        
    for ind_offs in np.arange(0, params.n_pop,1):
        threads[ind_offs].join()    
    
    for ind_offs in np.arange(0, params.n_pop,1):
        offs, th_ind, res_queue = th_get_res_queue_pop(res_queue, ind_offs, offs)
        if bool_verbose_thread == True:
            print("Crossover: thread finished...exiting" + ": " + str(th_ind) + ", Q=" + str(offs.Q[ind_offs]))

    return offs            
        
          

#%% Mutation 

def print_details_mutation(offs, ind,lold, lnew,g):
    extra="mutation_" + str(g)
    print("mutation; " + str(ind) + ";\t" + str(extra) + ";\tQ="+ str(offs.Q[ind]))
    print(offs.part[ind])
    
    
def mutation(graph, eig, params, pop, offs,g):
   
    num_mutation= int(np.random.choice( np.arange(1,np.floor(graph.g.vcount()/2)+1,1) ,1, replace=False))
    #print("num_mutation",num_mutation)
    ind = np.random.choice( np.arange(0, params.n_pop),1, replace=False )[0]
    v_shuffle = np.random.choice(np.arange(0,graph.g.vcount(),1) , num_mutation, replace=False )
    
    for vi in v_shuffle:
        #np.arange(0,num_mutation,1):
        lnew = np.random.choice( np.arange(0,graph.KNumber), 1, replace=False )[0]
        lold = offs.part[ind,vi]
        
        # print("ind="+str(ind)+ ", vi="+str(vi))
        # print("lnew = " + str(lnew) + ", lold="+str(lold))
        if lnew != lold:
            offs.Q[ind] += move_vert(graph, eig, offs,ind, vi, lnew, lold)

    if bool_verbose == True:
        print("Ind " + str(ind) + " after mutation ")
        print_ind(graph, offs, ind)
        print_details_mutation(offs, ind,lold, lnew,g)
    
    return offs
    #destroy shuffle 


#%% Update population

def update_population_offspring(pop, offs, params, graph):
    nhighest = int(np.floor( params.p_offs*params.n_pop ))
    #find the highest nreplace individuals from offspring
    if bool_verbose == True:
        print("nhighest="+str(nhighest))
    #print("nhighest="+str(nhighest))
    
    ind_offs = np.argsort(offs.Q)[-nhighest:]
    ind_pop = np.argsort(pop.Q)[:nhighest]
    #print("ind_offs")
    #print(ind_offs)
    #print("ind_pop")
    #print(ind_pop)
    
    for a in np.arange(0,nhighest):
            
        #copy_rgroup_pop(offs, ind_offs[a], pop, ind_pop[a]) 
        #260721
        pop = copy_rgroup_pop( pop, ind_pop[a], offs, ind_offs[a]) 
        #print(a)
        #print_ind(graph, offs, ind_pop[a])
    
    #print("print offs")
    ind_best = np.argsort(pop.Q)[-1:][0]
    #print(ind_best)
    mod_best=graph.g.modularity(pop.part[ind_best],  graph.g.es['weight'])
    #print("Best Q=" ,mod_best)        
    
    return pop        

#%% Initial partition

def initial_partition_ind_th(res_queue, graph, pop, ind, eig, it_max, bool_verbose):
    np.random.seed()
    pop.RGroup_pos[ind], pop.RGroup_neg[ind], pop.Sov[ind], pop.part[ind], pop.Q[ind] = sf.spec_initial_partition(graph.g, pop.RGroup_pos[ind], pop.RGroup_neg[ind], graph.ri_vert_pos, graph.ri_vert_neg, eig.pos_ind, eig.neg_ind, graph.KNumber, bool_verbose)            
    

    #res_queue.put([pop.RGroup_pos[ind], pop.RGroup_neg[ind], pop.Sov, pop.part[ind], pop.Q[ind]])    
    res_queue.put([ind, pop.RGroup_pos[ind], pop.RGroup_neg[ind], pop.Sov[ind], pop.part[ind], pop.Q[ind]])    


def initial_partition_ov_ind_th(res_queue, graph, pop, ind, eig, params, it_max, bool_verbose):
    np.random.seed()    
    pop.RGroup_pos[ind], pop.RGroup_neg[ind], pop.Sov[ind],  pop.part[ind], pop.Q[ind] = sf.spec_initial_partition(graph.g, pop.RGroup_pos[ind], pop.RGroup_neg[ind], graph.ri_vert_pos, graph.ri_vert_neg, eig.pos_ind, eig.neg_ind, graph.KNumber, bool_verbose)            
    #print("ini Q=", pop.Q[ind])
    
    #TEST: local search on initial partition?
    
    pop.RGroup_pos[ind], pop.RGroup_neg[ind], pop.part[ind], pop.Sov[ind], pop.Q[ind] = sf.spec_local_search(graph.g, graph.ri_vert_pos, graph.ri_vert_neg, eig.pos_ind, eig.neg_ind, graph.KNumber, pop.RGroup_pos[ind], pop.RGroup_neg[ind], pop.Sov[ind], pop.part[ind], it_max= it_max, bool_verbose=bool_verbose)    
    pop.RGroup_pos[ind], pop.RGroup_neg[ind], pop.part[ind], pop.Sov[ind], pop.Q[ind]  = spec_ov.spec_overlapping(graph.g, graph.ri_vert_pos, graph.ri_vert_neg, eig.pos_ind, eig.neg_ind,  graph.KNumber,pop.RGroup_pos[ind], pop.RGroup_neg[ind], pop.Sov[ind],  pop.part[ind], it_max= it_max, bool_verbose=False)      
    #print("ls Q=", pop.Q[ind])
    res_queue.put([ind, pop.RGroup_pos[ind], pop.RGroup_neg[ind],  pop.Sov[ind], pop.part[ind], pop.Q[ind]])    


def initial_partition_pop_call(params, graph, pop, eig, it_max, bool_ov):
    threads = []
    #res_queue = queue.Queue()
    manager = multiprocessing.Manager()
    res_queue = manager.Queue()
    #qOut = manager.Queue()

    #res_queue = multiprocessing.Queue()
    if bool_verbose_thread == True:
        print("Initial partitions thread")
        
    for ind in np.arange(0, params.n_pop, 1):    	
        # if bool_verbose == True:
        #     print("\n## ind="+ str(ind))        

        #pop.RGroup_pos[ind], pop.RGroup_neg[ind], pop.Sov, pop.part[ind], pop.Q[ind] = sf.spec_initial_partition(graph.g, pop.RGroup_pos[ind], pop.RGroup_neg[ind], graph.ri_vert_pos, graph.ri_vert_neg, eig.pos_ind, eig.neg_ind, graph.KNumber, bool_verbose)            
        #t = threading.Thread(target=spec_th_initial_partition,args=[res_queue, graph, pop, ind, eig, bool_verbose])
        if bool_ov == False:
            t = multiprocessing.Process(target= initial_partition_ind_th ,args=[res_queue, graph, pop, ind, eig, it_max, bool_verbose])
        else:
            if bool_verbose_thread == True:
                print("initial ov")
                
            t = multiprocessing.Process(target= initial_partition_ov_ind_th ,args=[res_queue, graph, pop, ind, eig, params, it_max, bool_verbose])
        
        
        threads.append(t)
        threads[ind].start()
        if bool_verbose_thread == True:
            print("thread started..."+ str(ind))        
    
    for ind in np.arange(0, params.n_pop, 1):    	
        threads[ind].join()   
        if bool_verbose_thread == True:
            print("Initial: thread finished...exiting:", ind)
        
    for ind in np.arange(0, params.n_pop, 1):    	      
        
        pop, th_ind, res_queue = th_get_res_queue_pop(res_queue, ind, pop)
        if bool_verbose_thread == True:
            print("Initial: thread finished...exiting" + ": " + str(th_ind) + ", Q=" + str( pop.Q[ind]))
        #print("thread finished...exiting" + ": " + str(res_t[0]) + ", Q=" + str(res_t[1]))

    
    #print("Initial: CalcQ="+str(valQ) + ", (" + str(th_num) + ")")
    #if bool_verbose==True:
        #print(pcurr)
            
    return pop

#%% Local search


def local_search_ind_th(res_queue, graph, pop, ind, eig, it_max, bool_verbose):
    np.random.seed()
    pop.RGroup_pos[ind], pop.RGroup_neg[ind], pop.part[ind], pop.Sov[ind], pop.Q[ind] = sf.spec_local_search(graph.g, graph.ri_vert_pos, graph.ri_vert_neg, eig.pos_ind, eig.neg_ind, graph.KNumber, pop.RGroup_pos[ind], pop.RGroup_neg[ind], pop.Sov[ind], pop.part[ind], it_max=it_max, bool_verbose=bool_verbose)  
     
    #res_queue.put([pop.RGroup_pos[ind], pop.RGroup_neg[ind], pop.Sov, pop.part[ind], pop.Q[ind]])    
    res_queue.put([ind, pop.RGroup_pos[ind], pop.RGroup_neg[ind],  pop.Sov[ind], pop.part[ind], pop.Q[ind]])    


def local_search_ov_ind_th(res_queue, graph, pop, ind, eig, params, it_max, bool_verbose):
    np.random.seed()
    
    pop.RGroup_pos[ind], pop.RGroup_neg[ind], pop.part[ind], pop.Sov[ind], pop.Q[ind]  = spec_ov.spec_overlapping(graph.g, graph.ri_vert_pos, graph.ri_vert_neg, eig.pos_ind, eig.neg_ind,  graph.KNumber,pop.RGroup_pos[ind], pop.RGroup_neg[ind], pop.Sov[ind],  pop.part[ind], it_max=it_max, bool_verbose=False)      
               
    #Post processing
    # print("TEST LS POST.")
    # print(pop.Sov[ind]) #OK

    pop.Sov[ind] = spec_ov.spec_post_processing(pop.Sov[ind], params.ov_thr)

    res_queue.put([ind, pop.RGroup_pos[ind], pop.RGroup_neg[ind],  pop.Sov[ind], pop.part[ind], pop.Q[ind]])    
        
#TODO: try to send function as parameter
def local_search_pop_call(params, graph, pop, eig, it_max, bool_ov, bool_verbose):
    threads = []
    manager = multiprocessing.Manager()
    res_queue = manager.Queue()
    #print("Local search thread")
        
    for ind in np.arange(0, params.n_pop, 1):    	
        # if bool_verbose == True:
        #     print("\n## ind="+ str(ind))        
       
        #multiprocessing.Process
        if bool_ov == False:
            t = multiprocessing.Process(target=local_search_ind_th,args=[res_queue, graph, pop, ind, eig, it_max, bool_verbose])
        else:
            t = multiprocessing.Process(target=local_search_ov_ind_th,args=[res_queue, graph, pop, ind, eig, params, it_max, bool_verbose])
                                    
        threads.append(t)
        
    for ind in np.arange(0, params.n_pop, 1):    	        
        threads[ind].start()
        if bool_verbose_thread == True:
            print("Local Search: thread started..."+ str(ind))        
    
    
    for ind in np.arange(0, params.n_pop, 1):    	
        # print("wait to join:",ind)
        threads[ind].join()   
        # print("join ok", ind)
        
    for ind in np.arange(0, params.n_pop, 1):    	        
        pop, th_ind, res_queue = th_get_res_queue_pop(res_queue, ind, pop)
        if bool_verbose_thread == True:
            print("Local Search: thread finished...exiting" + ": " + str(th_ind) + ", Q=" + str( pop.Q[ind]))
        #print("thread finished...exiting" + ": " + str(res_t[0]) + ", Q=" + str(res_t[1]))
        
    return pop

#%% Genetic algorithm 

    
def genetic_algorithm(graph, eig, params, bool_ov):
    
    #nconv, eig_val, eig_vec, eig_pos_ind, eig_neg_ind, n_pop, g, ri_vert_pos, ri_vert_neg, KNumber, pop_part, pop_Q, pop_RGroup_pos, pop_RGroup_neg, offs_RGroup_pos, offs_RGroup_neg ):
    nconv= len(eig.val)     
      
    pop, offs = initialize_pop_arrays(graph.g, params.n_pop_off, graph.KNumber, eig.pos_ind, eig.neg_ind )
    


    list_gen_Q = np.zeros(params.n_gen)		
    
    #initial solution
    #TODO THREAD
    if bool_verbose == True:
        print("\n*** Initial")

    pop = initial_partition_pop_call(params, graph, pop, eig, it_max=params.it_max_ls_ini, bool_ov=bool_ov)        
    #print("\n*** Initial")
    #print_ind(graph, pop, 0)
    #print("\n")
    
    #print("TEST-- error")    
    #print_ind(graph, pop, 0)
    
    for g in np.arange(0,params.n_gen):
        
        if bool_verbose == True:
            print("\n*** Crossover")
            
        #offs = crossover_offspring(graph, eig, params, pop, offs,g)    
        offs = crossover_offspring_offs_call(graph, eig, params, pop, offs,g)    
        
        if bool_verbose == True:
            print("\n*** Mutation")
            
        offs = mutation(graph, eig, params, pop, offs,g)
    #local search - already implemented
        if bool_verbose == True:
            print("\n*** Local search")
            
        if bool_verbose == True:
            print("part to local search")
            #print(pop.part[ind])
        
        #TODO threads..
        #pop.RGroup_pos[ind], pop.RGroup_neg[ind],  pop.part[ind], pop.Sov, pop.Q[ind] = sf.spec_local_search(graph.g, graph.ri_vert_pos, graph.ri_vert_neg, eig.pos_ind, eig.neg_ind, graph.KNumber, pop.RGroup_pos[ind], pop.RGroup_neg[ind], pop.Sov, pop.part[ind], bool_verbose)  
        
        #pop = local_search_pop_call(params, graph, pop, eig, it_max=params.it_max_ls_gen, bool_ov=False, bool_verbose=True)
        #pop = local_search_pop_call(params, graph, pop, eig, it_max=params.it_max_ls_gen, bool_ov=bool_ov, bool_verbose=True)
        #260721
        pop = local_search_pop_call(params, graph, offs, eig, it_max=params.it_max_ls_gen, bool_ov=bool_ov, bool_verbose=bool_verbose)
        
        #260721
        
        pop = update_population_offspring(pop, offs, params, graph);
        
        #260721    
       
        #print(ind_best)              
        #list_gen_Q[g] = 
        #print(pop.Q[ind_best])
        #print(pop.part[ind_best])
        ind_best = np.argsort(pop.Q)[-1:][0]
        #mod_best=graph.g.modularity(pop.part[ind_best],  graph.g.es['weight'])
        print("Best Q=" ,pop.Q[ind_best])
        #list_gen_Q[g] = mod_best
        list_gen_Q[g] = pop.Q[ind_best]
    
    if bool_ov == True:
        #print("---------------")
        #print("final local search")
        #Local search with overlapping only after    
        #pop = local_search_pop_call(params, graph, pop, eig, it_max=2, bool_ov=False, bool_verbose=True)        
        print("---------------")
        print("OV final local search 2")
        pop = local_search_pop_call(params, graph, pop, eig, it_max=params.it_max_ls_final, bool_ov=bool_ov, bool_verbose=True)        
        

        
    ind_best = np.argsort(pop.Q)[-1:]        
    print("Best Q=" + str(pop.Q[ind_best]))
    #print(pop.RGroup_pos)
    if bool_verbose == True:
        for ind in np.arange(0,params.n_pop):
            print_ind(graph, pop, ind)
        
        
        print("Best part=")
        print(pop.part[ind_best,:])

    #post processing Sov
    if bool_ov == True:    
        #TODO: ov_thr must be an input parameter
        pop.Sov[ind_best] = spec_ov.spec_post_processing(pop.Sov[ind_best], ov_thr=0.1) #290321
    
    #print( pop.Sov[ind_best])
    
    #return pop.part[ind_best], pop.Sov[ind_best], pop, offs''
    return pop.part[ind_best], pop.Sov[ind_best], pop, offs, pop.RGroup_pos[ind_best], pop.RGroup_neg[ind_best], list_gen_Q #010421


#%% Ensemble 
def ensemble_threshold(ensemble, graph, params):
    
    # print(ensemble.consensus)
    
    avg_thr = np.mean( ensemble.consensus )
    
    print("Avg threshold before="+str(avg_thr))
    # print(ensemble.p_thr)
    
    nnz=0
    for i in np.arange(0,graph.g.vcount()):
        j_larg = np.argmax( ensemble.consensus[i,:])        
        val_larg = ensemble.consensus[i, j_larg]
        
        
        # np.where: returns rows[a], cols[a] indicate the a-th element that satisfies the condition         
        #returns only cols, since ensemble[i,:] is one dim -> use #consensus_where_zero[0]
        
        consensus_col_where_zero = np.where( ensemble.consensus[i,:] < ensemble.p_thr )[0]
        #only count in rows, otherwise, it would count twice
        nnz_i = len( consensus_col_where_zero) 
        #print(ensemble.consensus[i,:])        
        #print("nnz_"+str(i)+" =" + str(nnz_i))
        nnz+= nnz_i
        
        ensemble.consensus[ i, consensus_col_where_zero ] = 0 
		
        #check if i is disconnected and link to the vert whose consensus value if the highest        
        if nnz_i==0:
            ensemble.consensus[i,j_larg] = val_larg 
            nnz+=1
        
    avg_thr = np.mean( ensemble.consensus )
    
    print("Avg threshold after="+str(avg_thr))
    return ensemble, nnz
    
def ensemble_consensus_number(ensemble, graph, params):
    k_sum = 0
    for ind in np.arange(0,params.num_pareto+1):
        #number of different communities 
        k_num = len(np.unique(ensemble.pareto_parts[ind,:]))
        k_sum += k_num
        
    k_sum = k_sum/params.num_pareto
    return k_sum
        #len( ensemble.pareto_parts[ind,:] )    


def ensemble_consensus_parts(graph, eig, params, ensemble, DEF_SEL_PARETO):
    
    consensus = np.zeros((graph.g.vcount(), graph.g.vcount()), dtype=np.double )
    ind_ini = int(np.floor( np.double(1-DEF_SEL_PARETO)*np.double(params.num_pareto/2) ) )
    ind_fim = int(params.num_pareto -1 -ind_ini)
    
    if bool_verbose == True:
        print("ind_ini="+str(ind_ini) + ", ind_fim="+str(ind_fim))
        
    sel_pareto = ind_fim - ind_ini + 1

    for i in np.arange(0,graph.g.vcount()):
        for j in np.arange(0,graph.g.vcount()):#IMPROVE: SYMMETRIC
            #consensus[i,j]=0            
            for ind in np.arange(ind_ini,ind_fim+1):
                if ensemble.pareto_parts[ind,i] == ensemble.pareto_parts[ind,j]:
                    consensus[i,j] += 1

    consensus = consensus/sel_pareto
    
    # print(consensus)
    
    return consensus

def ensemble_construct_graph_ensemble(graph, ensemble):
    adj = np.matrix(graph.g.get_adjacency(attribute="weight").data)
    adj_ens = np.add(adj, ensemble.consensus)          
    #graph_ens = graph.copy() #TEST
    #shallow copy.. make sure not to use original graph again
    graph_ens = copy.copy(graph)    
    #ERROR: SHOULD BE UNDIRECTED
    graph_ens.g = igraph.Graph.Weighted_Adjacency(adj_ens.tolist(), attr="weight", mode=igraph.ADJ_UNDIRECTED) #replace graph with the adjusted for ensemble
    
    return graph_ens
           

#%% Eigendecomposition 

def spec_eigen_dec(graph, mod_mat, p_leading,pos_neg=True ):
    eig = Eig()
    
    if pos_neg==True:
        eig.val, eig.vec, eig.pos_ind, eig.neg_ind = sf.calc_eigen(graph.g, mod_mat, p_leading, which="LM")
    else:
        eig.val, eig.vec, eig.pos_ind, eig.neg_ind = sf.calc_eigen(graph.g, mod_mat, p_leading, which="LR")
    
    #print(eig_val)
    graph.KNumber = sf.est_number_communities(eig.val)
    print("Estimation k: " + str( graph.KNumber))
    #TODO: benson algorithm to estimate the number of communities
    
    print("** Calculate vertex vectors ri... **")
    
    graph.ri_vert_pos = sf.calc_ri_vert_pos(graph.g, eig.val, eig.vec, eig.pos_ind)
    graph.ri_vert_neg = sf.calc_ri_vert_neg(graph.g, eig.val, eig.vec, eig.neg_ind)
    
    return eig, graph

#%% MOSpecG - main

def MO_main(MO, IT, p_leading, file_graph, results_folder, time_folder, pareto_file, num_pareto, ensemble_thr, n_gen, n_pop, p_offs, mod_orig,bool_ov,T):
    graph = Graph()
    graph.g = sf.read_construct_g(file_graph)
    
    #initalize the pop arrays that are dependent on the eigenvalues
    #pareto_parts, n_pop_off, pop_part, pop_Q, offs_part, offs_Q = construct_modularity_cosine_param
    params, ensemble = initialize_params( graph.g, num_pareto, ensemble_thr, n_gen, n_pop, p_offs)
    
    step = 1/(params.num_pareto-1)
    
    print("step="+str(step))
    for it in np.arange(0,num_pareto,1):    
        print("\n-----------------------\n**** it="+str(it))
    #it=1 #TEST
    #it=5
        mo_weight_Qin=it*step    
        mo_weight_Qnull = 1-mo_weight_Qin
        
        print("mo_Qin="+str(mo_weight_Qin)+ ", mo_Qnull="+str(mo_weight_Qnull))
        
        #mod_mat = sf.construct_modularity_generalized(graph.g, mo_weight_Qin, mo_weight_Qnull).transpose()
            
        if mod_orig==False:    
            S= sf.construct_cosine(graph.g)
            mod_mat = sf.construct_modularity_cosine_param(graph.g,S, mo_weight_Qin, mo_weight_Qnull, T).transpose()        
        else:
            mod_mat = sf.construct_modularity_generalized(graph.g, mo_weight_Qin, mo_weight_Qnull).transpose()
        
        eig, graph = spec_eigen_dec(graph, mod_mat, p_leading  )
        
        print("** Start GA... **")
        #aloc pop arrays        
        p_best, Sov_best, pop, offs, Rp, Rn, list_gen_Q = genetic_algorithm(graph, eig, params, bool_ov)
        ensemble.pareto_parts[it,:] = p_best.copy()
    
    
    return ensemble, graph, params, eig, pop, offs, list_gen_Q

        
def SpecG_single_main(MO, IT, p_leading, file_graph, file_com_out, time_folder, pareto_file, num_pareto, ensemble_thr, n_gen, n_pop, p_offs, bool_ov, ov_thr=0, mo_weight_Qin=1 , mo_weight_Qnull =0, gamma3=0, mod_orig=True,
                      KNumber_pre = -1, pos_neg=True, bool_link_pred=False, T=None):
    graph = Graph()
    graph.g = sf.read_construct_g(file_graph,bool_isolated=bool_link_pred)
    
    print(graph.g.vs.attribute_names())
    print("out:",file_com_out)
    #initalize the pop arrays that are dependent on the eigenvalues
    #pareto_parts, n_pop_off, pop_part, pop_Q, offs_part, offs_Q = 
    params, ensemble = initialize_params( graph.g, num_pareto, ensemble_thr, n_gen, n_pop, p_offs, ov_thr)
    
    print("gamma1=",mo_weight_Qin,", gamma2=",mo_weight_Qnull)
    #
    #
    print("* mo_Qin="+str(mo_weight_Qin)+ ", mo_Qnull="+str(mo_weight_Qnull))
    
    if mod_orig==False:    
        #print("encoder")        
        graph.g, mod_mat, adj, null_model = sf.construct_modularity_cosine(graph.g, mo_weight_Qin, mo_weight_Qnull, gamma3, T=T)        
        #why transpose? 
        mod_mat = mod_mat.transpose()
        
        #mod_mat = sf.construct_modularity_encoder(graph.g, mo_weight_Qin, mo_weight_Qnull).transpose()        
    else:
        mod_mat, adj, null_model = sf.construct_modularity_generalized(graph.g, mo_weight_Qin, mo_weight_Qnull)
        mod_mat = mod_mat.transpose()
    #TODO sparsify mod_mat - STOP SPARSIFY!! 
    #mod_mat[mod_mat < 0] =0 #260721
    
    
    eig, graph = spec_eigen_dec(graph, mod_mat, p_leading,pos_neg  )
    
    if KNumber_pre > 0:
        graph.KNumber = KNumber_pre
    
    print("** Start GA... **")
    #aloc pop arrays
     
    p_best, Sov_best, pop, offs, Rp , Rn, list_gen_Q = genetic_algorithm(graph, eig, params, bool_ov)        
    #END PUT INSIDE ANOTHER FUNCTION!
    mod0 = graph.g.modularity(p_best[0], graph.g.es['weight'])
    print("# mod =",mod0)

    
    
    return p_best, graph, params, eig, pop, offs, Sov_best, Rp, Rn, list_gen_Q, mod_mat, adj, null_model

        
def SpecG_rp_only(MO, IT, p_leading, file_graph, file_com_out, time_folder, pareto_file, num_pareto, ensemble_thr, n_gen, n_pop, p_offs, bool_ov, ov_thr=0, mo_weight_Qin=1 , mo_weight_Qnull =1, mod_orig=True,
                      KNumber_pre = -1, pos_neg=True, bool_link_pred=False, T=None):
    graph = Graph()
    graph.g = sf.read_construct_g(file_graph,bool_isolated=bool_link_pred)
 
    mod_mat, adj, null_model = sf.construct_modularity_generalized(graph.g, mo_weight_Qin, mo_weight_Qnull)
    mod_mat = mod_mat.transpose()
    #TODO sparsify mod_mat - STOP SPARSIFY!! 
    #mod_mat[mod_mat < 0] =0 #260721
        
    eig, graph = spec_eigen_dec(graph, mod_mat, p_leading,pos_neg  )
   
    p_best=[1]
    Sov_best=[1]
    return p_best, graph, -1, eig,  -1,  -1, Sov_best,  -1,  -1, -1, mod_mat, adj, null_model

   
#%% Ensemble - main

def ensemble_MOSpecG_main(MO_bool, num_it, p_leading, file_graph, results_folder, time_folder, pareto_file, num_pareto, ensemble_thr, n_gen, n_pop, p_offs,  bool_ov, ov_thr,  mod_orig=True):
    DEF_SEL_PARETO = 0.8
    DEF_MULT_CONSENSUS = 2
    
    ensemble, graph, params, eig, pop, offs,list_gen_Q = MO_main(MO_bool, num_it, p_leading, file_graph, results_folder, time_folder, pareto_file, num_pareto, ensemble_thr, n_gen, n_pop, p_offs,mod_orig, bool_ov=False)


    print("\n-----------------\n ***** ENSEMBLE \n")
    if bool_verbose == True:
        print("\n ensemble_consensus_parts ")

    ensemble.consensus = ensemble_consensus_parts(graph, eig, params, ensemble,DEF_SEL_PARETO) 
    #k_consensus = ensemble_consensus_number( ensemble, graph, params) #not used
    #print("Consensus number="+str(k_consensus)) #not used
    
    ensemble, nnz = ensemble_threshold(ensemble, graph, params)
    
    #already done - params, ensemble = initialize_params( graph.g, num_pareto, ensemble_thr, n_gen, n_pop, p_offs)
    
    mo_weight_Qin=1  
    mo_weight_Qnull =1
    print("\n-----------------\n ***** MOSpecG \n")
    print("mo_Qin=",mo_weight_Qin, ", mo_Qnull=",mo_weight_Qnull)
              
    #graph_ens is a shallow copy.. make sure not to use original graph again
    graph_ens= ensemble_construct_graph_ensemble(graph, ensemble)
    mod_mat_ens = sf.construct_modularity_generalized(graph_ens.g, mo_weight_Qin, mo_weight_Qnull).transpose()
 
    eig_ens, graph_ens = spec_eigen_dec(graph_ens, mod_mat_ens, p_leading  )
    
    print("** Start GA... **")    
    p_best, Sov_best, pop, offs, Rp,Rn,list_gen_Q = genetic_algorithm(graph_ens, eig_ens, params, bool_ov)        
    
    #0.55 for karate.. ???
    
    return p_best, graph_ens, params, eig,pop, offs, Sov_best, Rp,list_gen_Q


#%% Call main - example

def call_MOSpecG_example():
    #base_folder="/home/camila/not_sync/Lancichinetti2017/Binary_2017_Lancichinetti2011_S/paj_Binary_2017_S/"
    #file_graph=base_folder+"net_Binary_2017_S1000_1_1.net"
    
    #base_folder="/home/camila/not_sync/Lancichinetti2017/Binary_DirNormal2_OV_2017_Lancichinetti2011_S/paj_Binary_DirNormal2_OV_2017_S/"
    #file_graph=base_folder+"net_Binary_DirNormal2_OV_2017_S1000_1_1.net"
    
    base_folder="grafos_nao_direcionados/"
    file_graph=base_folder+"karate.net"
    #file_graph=base_folder+"net_Binary_2017_S1000_1_1.net"
    
    results_folder=""
    time_folder=""
    pareto_file=""
    #file_expected="karate_expected.txt"
      
    num_pareto=11
    ensemble_thr=0.5
    
    num_it=10
    n_gen=10 #10
    n_pop=2 #5  #3 #ERROR!!
    p_offs=0.6
    
    # output_time="time_aux.txt"
    # net_type="real"
    p_leading=1
    #p_leading=0.3  
    
    #TODO:
    bool_ov=False
    ov_thr=0.5
    bool_verbose=False
    out_verbose=[]
    
    
    MO_bool=False

    #start_time = time.time()
    start_time = time.perf_counter()
    print(file_graph)
    #p_best, graph, params, eig, pop, offs, Sov_best = SpecG_single_main(MO_bool, num_it, p_leading, file_graph, results_folder, time_folder, pareto_file, num_pareto, ensemble_thr, n_gen, n_pop, p_offs, bool_ov)
    
    
    #p_best, graph, params, eig, pop, offs, Sov_best, Rp = ensemble_MOSpecG_main(MO_bool, num_it, p_leading, file_graph, results_folder, time_folder, pareto_file, num_pareto, ensemble_thr, n_gen, n_pop, p_offs, bool_ov, ov_thr)
    p_best, graph, params, eig, pop, offs, Sov_best, Rp,Rn,list_gen_Q = SpecG_single_main(MO_bool, num_it, p_leading, file_graph, results_folder, time_folder, pareto_file, num_pareto, ensemble_thr, n_gen, n_pop, p_offs, bool_ov, ov_thr) 
    print(list_gen_Q)
    
    #print("--- %s seconds ---" % (time.time() - start_time))
    print("--- %s seconds ---" , (time.perf_counter() - start_time))
    
    

    return graph, params, eig, pop, offs


if __name__ == "__main__":
    graph, params, eig, pop, offs=call_MOSpecG_example()

#%% Call main - LFR

def call_spec_ov_LFR(nome_alg="SpecG_OV",bool_ov=True,bool_ensemble=True,lfr_set = "S"):
    #Default parameters:    
    output_time="time_aux.txt"
    net_type="real"
    
    gamma1=1
    gamma2=1
    #bool_ensemble=True
    #bool_ov=False
    #ov_thr=0.5
    ov_thr=0.1
    
    bool_verbose=False
    out_verbose=[]
    
    folder_res="results_Tese_2021_3"
    #nome_alg = "SpecG_OV"         
    if bool_ensemble == True:
        nome_alg=nome_alg+"_ens"
    #LFR
       
    # OV:
    if bool_ov==False:
        lfr_type="Binary_2017"
    else:
        lfr_type="Binary_DirNormal2_OV_2017" #Binary2017_DirNormal2_OV_2017
    
    #lfr_type="Binary_2017"
    
    #lfr_type="Binary_Sparse6_OV2_2017"
    
    
    tam = 1000
    
           
    results_folder=""
    time_folder=""
    pareto_file=""
    
    num_pareto=11
    ensemble_thr=0.5
    
    num_it=1
    n_gen=10 #10
    n_pop=5  #3 #ERROR!!
    p_offs=0.6
    
    #p_leading=0.95    
    p_leading=0.3
    
    # for mu in range(1,9):
    #     for n2 in range(1,6):  
        
    print("\n-----Alg=", nome_alg)
    
    ex=1        
    for mu in range(1,8+1): #7
        for n2 in range(1,5+1):
            inst_base =  lfr_type +"_Lancichinetti2011_"+ lfr_set 
            base_folder_in="/home/camila/not_sync/Lancichinetti2017/" + inst_base + "/paj_" + lfr_type + "_" + lfr_set + "/"
            inst_name =  lfr_type + "_" + lfr_set + str(tam) + "_" + str(mu) + "_" + str(n2)
            file_graph=base_folder_in + "net_" + inst_name + ".net"

            #base_folder_exp="/home/camila/not_sync/Lancichinetti2017/" + lfr_type +"_Lancichinetti2011_"+ lfr_set + "/com_" + lfr_type + "_" + lfr_set + "/"
            #file_exp=base_folder_in+"net_" + lfr_type + "_" + lfr_set + str(tam) + "_" + str(mu) + "_" + str(n2) + ".net"
        
          
            print("Graph: ", file_graph)
            if bool_ov == True:
                base_folder_out = folder_res+"/"+nome_alg + "/" + inst_base  +"/lines_"+ inst_base + "_" + str(ex)+ "/"
                file_com_out = base_folder_out+"lines_net_" + inst_name + "_ov.com"
            
                file_time_out = base_folder_out+"time_net_" + inst_name + "_ov.txt"
                file_k_num_out = base_folder_out+"num_net_" + inst_name + "_ov.txt"
                file_gen_out = base_folder_out+"gen_net_" + inst_name + "_ov.txt"
                file_gen_plot_out = base_folder_out+"plot_gen_net_" + inst_name + "_ov.png"
            
            else:
                base_folder_out = folder_res+"/"+nome_alg + "/" + inst_base  +"/part_"+ inst_base + "_" + str(ex)+ "/"
                file_com_out = base_folder_out+"part_net_" + inst_name + ".com"
                
                file_time_out = base_folder_out+"time_net_" + inst_name + ".txt"
                file_k_num_out = base_folder_out+"num_net_" + inst_name + ".txt"
                
                file_gen_out = base_folder_out+"gen_net_" + inst_name + ".txt"
                file_gen_plot_out = base_folder_out+"plot_gen_net_" + inst_name + ".png"
            print("Output: ", file_com_out)
            
            os.makedirs(base_folder_out, exist_ok=True)
    
            #start_time = time.perf_counter()
            #q_best, Sov,g, eig_val, ri_vert_pos, ri_vert_neg, Rp, Rn = spec_ov_main(file_graph, 4, plarg, gamma1, gamma2, ov_thr, bool_ov, file_com_out, bool_verbose, out_verbose)            
            
            MO_bool=False
            #print("out 1:", file_com_out)
            
            start_time = time.perf_counter()            
            
            if bool_ensemble == True:
                p_best, graph, params, eig, pop, offs, Sov_best, Rp,list_gen_Q = ensemble_MOSpecG_main(MO_bool, num_it, p_leading, file_graph, results_folder, time_folder, pareto_file, num_pareto, ensemble_thr, n_gen, n_pop, p_offs,  bool_ov, ov_thr)                
            else:
                print("=======================\n\n")
                p_best, graph, params, eig, pop, offs, Sov_best, Rp,Rn,list_gen_Q = SpecG_single_main(MO_bool, num_it, p_leading, file_graph, file_com_out, time_folder, pareto_file, num_pareto, ensemble_thr, n_gen, n_pop, p_offs, bool_ov, ov_thr)
                print(list_gen_Q)                        
                
                f = open(file_gen_out, "w")    
                print(list_gen_Q.reshape(1, list_gen_Q.shape[0]))
                np.savetxt(f, list_gen_Q.reshape(1, list_gen_Q.shape[0]), newline='\n')            
                f.close() 
                
                #print chart

                fig, ax = plt.subplots()
                ax.plot(list_gen_Q);
                fig.savefig(file_gen_plot_out,dpi=100)
                
            if bool_ov == True:     
                print("ok")
                #------ write communities
                spec_ov.print_com_lines_file(file_com_out, Sov_best[0]) #231220
                
                #------ write k number of communities
                Sov_best=Sov_best[0]
                idx= np.where( np.all(Sov_best[..., :] == 0, axis=0) == False )[0]
                Sov_nonzero = Sov_best[:,idx]
                k_num_best = Sov_nonzero.shape[1]
                f = open(file_k_num_out, "w")                
                f.write(str(k_num_best ))
                f.close() 
                
                
                
            else:          
                #write partition 
                
                base_folder_expected="/home/camila/not_sync/Lancichinetti2017/" + inst_base + "/com_" + lfr_type + "_" + lfr_set + "/"
                file_expected=base_folder_expected+"com_"+inst_name+".dat"

                #------ write partition                
                print_com_part(file_com_out, p_best[0])              
                #------ calc NMI
                print("file expected: ", file_expected)
                p_expected = sf.read_partition_expected(file_expected)
                nmi = sf.calc_nmi(p_best[0], p_expected)
                print("nmi=",nmi)
        
                #------ write k number of communities
                f = open(file_k_num_out, "w")                
                f.write(str(len( np.unique(p_best[0]))))
                f.close() 
            
            #------ write time
            
            #q_best, Sov = spec_ov_main_single(file_graph, 4, plarg, gamma1, gamma2, ov_thr, bool_ov, bool_verbose, out_verbose)
            time_tot = time.perf_counter() - start_time
            print("--- %s seconds ---" , time_tot)
            f = open(file_time_out, "w")                
            f.write(str(time_tot))
            f.close() 
            
            #TODO: save time
    
    #return q_best, Sov,g, eig_val, ri_vert_pos, ri_vert_neg, Rp, Rn
    return p_best,Sov_best

#if __name__ == "__main__":
    #Sov_best = call_spec_ov_LFR(nome_alg="SpecG_OV",bool_ov=True,bool_ensemble=True,lfr_set = "S")
    #Sov_best = call_spec_ov_LFR(nome_alg="SpecG_OV",bool_ov=True,bool_ensemble=True,lfr_set = "L")
    #Sov_best = call_spec_ov_LFR(nome_alg="SpecG",bool_ov=False,bool_ensemble=False)


#%% parameters
    
    
#  Parameters
# 		1. MO
# 			MO=10 to run MOSpecG for maximizing the classical modularity (MOSpecG-mod).
# 		    MO=11 to run MOSpecG to find Pareto sets and obtain a final consensus partition with SpecG algorithm.

# 		2. IT: maximum number of iterations for the local search multiplied by 10*****.

# 		3. p: consider only the largest p*n eigenvalues in absolute value, where n is the number of vertices. 

# 		4. <instance>: full path to the input graph in pajek format.

# 		5. <res_base>: base name of the file to output the partitions in between "".
# 			The algorithm will automatically generate output files.

# 		6. <time_base>: path and base name of the file to output the execution time in seconds in between "".
# 			The algorithm will automatically generate files containing the execution times in between "".

# 		7. <pareto_file>: full path to file to save the objective values

# 		8. NF: number of solutions in the Pareto set --  When M0=10 this input can be set to any value.

# 		9. tau: threshold parameter of SpecG - When M0=10 this input can be set to any value.

# 		10. NG: number of generations of the memetic algorithm.

# 		11. NP: size of the population of the memetic algorithm.

# 		12. NO: the NO% fittest individuals from the offspring replace the NO% least fit individuals from a current population in the memetic algorithm.






#%% test

# a


#%%% test 0
    
    # nex = 5 #10 
    # valQ_max = -100
    # nmi_max = -2
      
    # threads = []
    # res_queue = queue.Queue()
      
    # for it in range(0,nex):
    #   #print("---------------" + str(it))
    #   t = threading.Thread(target=spec_ov_th,args=[it, res_queue, g, comms_est, ri_vert_pos, ri_vert_neg, eig_pos_ind, eig_neg_ind, bool_ov, ov_thr, bool_verbose, out_verbose])
    #   threads.append(t)
    #   t.start()
    
    # for it in range(0,nex):
    #   threads[it].join()    
    #   res_t = res_queue.get()
    #   print("thread finished...exiting" + ": " + str(res_t[0]) + ", Q=" + str(res_t[1]))

    
    #pareto_parts = genetic_algorithm()
    
    #ensemble_clustering()
    
#%%% test 1 

#pop_RGroup_pos, pop_RGroup_neg, offs_RGroup_pos, offs_RGroup_neg 

#R_group_pos, R_group_neg, Sov, pcurr, valQ = sf.spec_initial_partition(g, ri_vert_pos, ri_vert_neg, eig_pos_ind, eig_neg_ind, KNumber, bool_verbose)

# print(R_group_pos)
#print(KNumber)
#pop_RGroup_pos[0]= R_group_pos
#print(pop_RGroup_pos)
#pop_RGroup_pos[3] = R_group_pos.transpose() ##CORRIGIR NO MODULO

#print(pop_RGroup_pos)

#print(len(pop_RGroup_pos))


#qual parte vai ser paralelizada? cada etapa do crossover, solução inicial, etc
#def MOSpecG_th(MO, it, plarg, file_graph, results_folder, time_folder, pareto_file, num_pareto, threshold, n_gen, n_pop, p_offs, bool_verbose=True, out_verbose=True):
#%%% test 2

# a = np.array([9, 4, 4, 3, 3, 9, 0, 4, 6, 0])
# #ret = np.argpartition(a, 4)[-4:]
# #ret = np.argpartition(a, 9)
# ret = np.argsort(a)[-7:]
# print(ret)
# print(a[ret])


# ind_offs = np.sort(offs.Q)[-2:]
# ind_pop = np.sort(pop.Q)[2:]

# print(offs.Q[ind_offs])
# print(pop.Q[ind_pop])
    
    
#print(offs.Q)
# ind
# array([1, 5, 8, 0])
# a[ind]
# array([4, 9, 6, 9])

#%%% test 3
#ls = 0
# #ld = 1 

# v_ls = np.where(pop.part[0] == 0)
# v_not_ld = np.where(pop.part[1] != 1)

# print(v_ls)
# print(v_not_ld)

# #v in ls and in v_not_ld
# print( np.intersect1d(v_ls, v_not_ld) )
#print(v_ls.intersection(v_not_ld ))
# print(v_ls[v_ls in v_not_ld]        )

#%%% test 4

# a = [1,2,4,5]
# p=[0.2,0.1,0.5,0.2]
# p1,p2 = np.random.choice(a, 2,p=p, replace=False)
# print(p1)
# print(p2)

# print(np.inner(a,a))

#%%% test 5
   
# print(pop.Q)
#print(pop.part[0].copy)

# params.n_pop_off


#%%% test 6



#ERROR: 
# Process Process-6176:
# Process Process-6178:
# Process Process-6177:
# Traceback (most recent call last):
# Traceback (most recent call last):
# Process Process-6179:
# Process Process-6180:
#   File "/usr/lib/python3.6/multiprocessing/process.py", line 93, in run
#     self._target(*self._args, **self._kwargs)
#   File "/usr/lib/python3.6/multiprocessing/process.py", line 258, in _bootstrap
#     self.run()
#   File "/usr/lib/python3.6/multiprocessing/process.py", line 258, in _bootstrap
#     self.run()
#   File "/home/camila/sync_files/spectral_python/MOSpecG_partitioning_OV.py", line 340, in crossover_offspring_ind_th
#     pSourceQ2,pDestQ1 = np.random.choice(np.arange(0,params.n_pop,1), 2,p=ind_prob, replace=False)
# Traceback (most recent call last):
#   File "/usr/lib/python3.6/multiprocessing/process.py", line 258, in _bootstrap
#     self.run()
# Traceback (most recent call last):
#   File "/usr/lib/python3.6/multiprocessing/process.py", line 93, in run
#     self._target(*self._args, **self._kwargs)
#   File "/home/camila/sync_files/spectral_python/MOSpecG_partitioning_OV.py", line 340, in crossover_offspring_ind_th
#     pSourceQ2,pDestQ1 = np.random.choice(np.arange(0,params.n_pop,1), 2,p=ind_prob, replace=False)
#   File "/usr/lib/python3.6/multiprocessing/process.py", line 258, in _bootstrap
#     self.run()
# Traceback (most recent call last):
#   File "mtrand.pyx", line 930, in numpy.random.mtrand.RandomState.choice
#   File "/usr/lib/python3.6/multiprocessing/process.py", line 258, in _bootstrap
#     self.run()
#   File "/usr/lib/python3.6/multiprocessing/process.py", line 93, in run
#     self._target(*self._args, **self._kwargs)
#   File "/usr/lib/python3.6/multiprocessing/process.py", line 93, in run
#     self._target(*self._args, **self._kwargs)
# ValueError: probabilities are not non-negative
#   File "mtrand.pyx", line 930, in numpy.random.mtrand.RandomState.choice
#   File "/usr/lib/python3.6/multiprocessing/process.py", line 93, in run
#     self._target(*self._args, **self._kwargs)
#   File "/home/camila/sync_files/spectral_python/MOSpecG_partitioning_OV.py", line 340, in crossover_offspring_ind_th
#     pSourceQ2,pDestQ1 = np.random.choice(np.arange(0,params.n_pop,1), 2,p=ind_prob, replace=False)
#   File "/home/camila/sync_files/spectral_python/MOSpecG_partitioning_OV.py", line 340, in crossover_offspring_ind_th
#     pSourceQ2,pDestQ1 = np.random.choice(np.arange(0,params.n_pop,1), 2,p=ind_prob, replace=False)
# ValueError: probabilities are not non-negative
#   File "/home/camila/sync_files/spectral_python/MOSpecG_partitioning_OV.py", line 340, in crossover_offspring_ind_th
#     pSourceQ2,pDestQ1 = np.random.choice(np.arange(0,params.n_pop,1), 2,p=ind_prob, replace=False)
#   File "mtrand.pyx", line 930, in numpy.random.mtrand.RandomState.choice
#   File "mtrand.pyx", line 930, in numpy.random.mtrand.RandomState.choice
#   File "mtrand.pyx", line 930, in numpy.random.mtrand.RandomState.choice
# ValueError: probabilities are not non-negative
# ValueError: probabilities are not non-negative
# ValueError: probabilities are not non-negative
# Traceback (most recent call last):

#   File "/home/camila/sync_files/spectral_python/MOSpecG_partitioning_OV.py", line 1152, in <module>
#     Sov_best = call_spec_ov_LFR(nome_alg="SpecG_OV",bool_ov=True,bool_ensemble=True)

#   File "/home/camila/sync_files/spectral_python/MOSpecG_partitioning_OV.py", line 1085, in call_spec_ov_LFR
#     p_best, graph, params, eig, pop, offs, Sov_best, Rp,list_gen_Q = ensemble_MOSpecG_main(MO_bool, num_it, p_leading, file_graph, results_folder, time_folder, pareto_file, num_pareto, ensemble_thr, n_gen, n_pop, p_offs,  bool_ov, ov_thr)

#   File "/home/camila/sync_files/spectral_python/MOSpecG_partitioning_OV.py", line 891, in ensemble_MOSpecG_main
#     ensemble, graph, params, eig, pop, offs,list_gen_Q = MO_main(MO_bool, num_it, p_leading, file_graph, results_folder, time_folder, pareto_file, num_pareto, ensemble_thr, n_gen, n_pop, p_offs, bool_ov=False)

#   File "/home/camila/sync_files/spectral_python/MOSpecG_partitioning_OV.py", line 847, in MO_main
#     p_best, Sov_best, pop, offs, Rp, list_gen_Q = genetic_algorithm(graph, eig, params, bool_ov)

#   File "/home/camila/sync_files/spectral_python/MOSpecG_partitioning_OV.py", line 645, in genetic_algorithm
#     offs = crossover_offspring_offs_call(graph, eig, params, pop, offs,g)

#   File "/home/camila/sync_files/spectral_python/MOSpecG_partitioning_OV.py", line 402, in crossover_offspring_offs_call
#     offs, th_ind, res_queue = th_get_res_queue_pop(res_queue, ind_offs, offs)

#   File "/home/camila/sync_files/spectral_python/MOSpecG_partitioning_OV.py", line 217, in th_get_res_queue_pop
#     res_t = res_queue.get() #False? not wait

#   File "<string>", line 2, in get

#   File "/usr/lib/python3.6/multiprocessing/managers.py", line 757, in _callmethod
#     kind, result = conn.recv()

#   File "/usr/lib/python3.6/multiprocessing/connection.py", line 250, in recv
#     buf = self._recv_bytes()

#   File "/usr/lib/python3.6/multiprocessing/connection.py", line 407, in _recv_bytes
#     buf = self._recv(4)

#   File "/usr/lib/python3.6/multiprocessing/connection.py", line 379, in _recv
#     chunk = read(handle, remaining)

# KeyboardInterrupt
