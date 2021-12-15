#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  4 20:20:53 2020

@author: camila
"""

#%% Import 
import igraph
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import math
import scipy.sparse.linalg
import random
import threading, queue

import matplotlib.transforms as transform

import spectral_functions as sf


def community_number_error_nbm(file_graph, file_expected, p_eig_array,gamma1=1,gamma2=1,char_split="\t"):
    g = sf.read_construct_g(file_graph)

 
#%%  teste   
    g = sf.read_construct_g(file_graph)
    base_folder="grafos_nao_direcionados/"
    base_expected="grafos_expected"
    file_graph=base_folder+"karate.net"
    file_expected=base_expected+"karate_expected.txt"
 

#%% Community number error

def community_number_error(file_graph, file_expected, p_eig_array,gamma1=1,gamma2=1,char_split="\t"):
  g = sf.read_construct_g(file_graph)
  mod_mat,_,_ = sf.construct_modularity_generalized(g,gamma1,gamma2)
  mod_mat = mod_mat.transpose()

  p_expected = sf.read_partition_expected(file_expected,char_split)
  k_expected = len(np.unique(p_expected))
  
  #print(p_expected)
  
  res_k_est_array = np.zeros(len(p_eig_array))
  res_k_expected_array = np.zeros(len(p_eig_array))

  print(p_eig_array)

  for plarg in p_eig_array:
    #eig_val, eig_vec, eig_pos_ind, eig_neg_ind = sf.calc_eigen(g,mod_mat, plarg)
    eig_val, eig_vec, eig_pos_ind, eig_neg_ind = sf.calc_eigen(g,mod_mat, plarg, which='LM')

    thr = math.sqrt( max(eig_val) )
    k_est = len(eig_val[eig_val > math.sqrt( max(eig_val) )])
    
    #k_est = len(eig_val[abs(eig_val) > abs( math.sqrt( max(eig_val) ))])
    
    
    #print("Largest eigenvalues to calc: "+str(plarg) )
    #print("Largest eigenvalue: " + str(max(eig_val)))
    #print("Lowest eigenvalue: " + str(max(eig_val)))
    #print("Estimation k: " + str(k_est))

    print("plarg="+str(plarg)+": k_expected="+str(k_expected) + ", k_est=" + str(k_est))    
    res_k_est_array[math.floor(plarg*len(p_eig_array))-1] = k_est
    res_k_expected_array[math.floor(plarg*len(p_eig_array))-1] = k_expected
  return res_k_est_array, res_k_expected_array

#%% Modularity error

def modularity_error(file_graph, file_expected, p_eig_array,gamma1=1,gamma2=1, char_split="\t"):
  g = sf.read_construct_g(file_graph)
  mod_mat,_,_ = sf.construct_modularity_generalized(g,gamma1,gamma2)
  mod_mat = mod_mat.transpose()

  p_expected = sf.read_partition_expected(file_expected,char_split)
  #print(p_expected)

  comms_est = len(np.unique(p_expected))
  #Calc all - calc estimation
  valQ_exact = sf.calc_Q_partition(g, mod_mat, comms_est, 1, p_expected)

  res_errorQ_perc_array = np.zeros(len(p_eig_array))
  print("p_eig_array")
  print(p_eig_array)

  for p in p_eig_array:  
    #valQ_est = sf.calc_Q_partition(g, mod_mat, comms_est, p, p_expected)
    valQ_est = sf.calc_Q_partition(g, mod_mat, comms_est, p, p_expected, which='LM')
    errorQ = valQ_exact - valQ_est
    errorQ_perc = 100*errorQ/valQ_exact
    print("plarg="+str(p)+": valQ_exact="+str(valQ_exact) + ", valQ_est=" + str(valQ_est) + ", error="+str(errorQ) +  ", %=" + str(errorQ_perc))
    #print(math.floor(p*10))
    print(math.floor(p* len(p_eig_array)))
    res_errorQ_perc_array[math.floor(p* len(p_eig_array))-1] = errorQ_perc

  return res_errorQ_perc_array

#%% Plot error 

def plot_error(error_array, p_eig_array, n1_array, file_name, ylabel,off=1,start_off=4):
    
    figure, ax = plt.subplots(figsize=(9, 9))
    #matplotlib.rc('text', usetex = True)
    
    #fig = plt.figure()
    #ax = plt.axes()  
    
    colors=['blue', 'red', 'green','pink', 'orange','violet','yellow','gray','brown','cyan','black']            
    
    #tr = mtrans.offset_copy(ax.transData, fig=figure, x=0.0, y=-105, units='points')
    tr = transform.Affine2D().translate(0, 0) + ax.transData

    if np.max(error_array) > 100:
        interval_y = 50
        max_y=250
    else:
        interval_y = 10
        max_y=50
 
    for i, p in enumerate(p_eig_array):
        if i>=start_off:
            if off > 1:
                tr = transform.Affine2D().translate(0, (i-start_off-1)/off) + ax.transData        
            else:
                tr = transform.Affine2D().translate(0, off) + ax.transData        
                
            ls=[':','--','-.',':','-.'][i%5]

        curve=math.floor(p*len(p_eig_array))-1
        ax.plot(n1_array,error_array[:,curve], color=colors[curve], transform=tr )#, linestyle=ls, linewidth=lw) #, label="Vertex vectors")       #s=10 ,
        #ax.set_xlabel('x-axis', fontsize = 20)
        print(math.floor(p*len(p_eig_array))-1)
        #for n1 in n1_array:
            #ax.text(n1, errorQ_perc_array_avg[n1-1,math.floor(p*10)-1], n1 ,fontsize=10)        
        
        plt.xticks( n1_array, np.round(n1_array/10,1), fontsize = 24,rotation='horizontal') #
        plt.yticks(fontsize = 24)
        plt.yticks(np.arange(0, max_y+1, interval_y))
        
        
    ax.grid()
    
    #plt.plot(n1_array,errorQ_perc_array_avg[:,0], color='blue') #, label="Vertex vectors")       #s=10 ,
    leg_p = ['$\mathit{p}$=' + str(round(p_eig,1)) + '$\mathit{n}$' for p_eig in p_eig_array] 
    
    ax.set_xlabel("Mixture coefficient ($\mu$)",fontsize = 25)
    ax.set_ylabel(ylabel,fontsize = 25)
    
    
    ax.legend(leg_p, prop={'size':18})
    #plt.show()
    plt.savefig(file_name)
     #p_eig_array, labelcolor=colors)
    plt.show()
    
    #plt.show()
      
    
#%% main LFR


def main_teoric_error_lfr(base_folder, base_expected, type_SL='S'):
    results_folder='../results_Tese_2021_3'
    size=1000
    #type_array = ['L']
    #n1_array = np.arange(1,8+1) #1,5+1
    
    n1_array = np.arange(1,8+1) #1,5+1
    n2_array = np.arange(1,5+1) #1,8+1
    
    #n1_array = np.arange(1,1+1) #1,5+1
    #n2_array = np.arange(1,1+1) #1,8+1
    
    p_eig_array = np.linspace(0.1,1,num=10)
    #p_eig_array = np.linspace(0.1,1,num=2)
      
    errorQ_array = np.zeros( (len(n1_array)*len(n2_array) , len(p_eig_array)) ,dtype=float)
    errorQ_perc_array = np.zeros( (len(n1_array)*len(n2_array), len(p_eig_array)) ,dtype=float)
    
    k_est_array = np.zeros( (len(n1_array)*len(n2_array), len(p_eig_array)) ,dtype=float)
    k_expected_array = np.zeros( (len(n1_array)*len(n2_array), len(p_eig_array)) ,dtype=float)
    
    errorQ_array_avg = np.zeros( (len(n1_array) , len(p_eig_array)) ,dtype=float)
    errorQ_perc_array_avg = np.zeros( (len(n1_array), len(p_eig_array)) ,dtype=float)
    
    k_est_array_avg = np.zeros( (len(n1_array)*len(n2_array), len(p_eig_array)) ,dtype=float)
    k_expected_array_avg = np.zeros( (len(n1_array), len(p_eig_array)) ,dtype=float)
    
    
    l=0
    l_avg=0
    for n1 in n1_array:
      
      
        #calc average error for n1: 
        for n2 in n2_array:    
            print("\nGraph: "+str(n1) + "," + str(n2))        
            #file_graph=base_folder+"karate.net"
            file_graph=base_folder+"net_Binary_2017_"+type_SL+"1000_"+str(n1)+"_"+str(n2)+".net"
            #file_expected=base_expected+"karate_expected.txt"
            file_expected=base_expected+"com_Binary_2017_"+type_SL+"1000_"+str(n1)+"_"+str(n2)+".dat"
            
            errorQ_perc_array[l,:] += modularity_error(file_graph, file_expected, p_eig_array)
            errorQ_perc_array_avg[l_avg,:] += abs(errorQ_perc_array[l,:])
            
            k_est_array[l,:], k_expected_array[l,:] = community_number_error(file_graph, file_expected, p_eig_array,1,1)
            
            k_expected_array_avg[l_avg,:] += (  k_est_array[l,:] - k_expected_array[l,:] )
            
            #print(k_est_array)
            l+=1
        
        l_avg+= 1
        
    errorQ_perc_array_avg = errorQ_perc_array_avg / n2_array.shape[0]
    k_expected_array_avg = k_expected_array_avg / n2_array.shape[0]

    file_name_errorQ = results_folder+'/exp_mod_error_'+'net_Binary_2017_'+type_SL+'1000'+'.png'
    plot_error(errorQ_perc_array_avg, p_eig_array, n1_array, file_name_errorQ, "Relative error ($\%$)",off=-0.05,start_off=8)    
    
    file_name_k_error = results_folder+'/exp_k_error_'+'net_Binary_2017_'+type_SL+'1000'+'.png'
    plot_error(k_expected_array_avg, p_eig_array, n1_array, file_name_k_error, "Difference on the number of communities",start_off=4)
    
    #return errorQ_perc_array, k_est_array, k_expected_array

#%% Call - LFR
  
base_folder="/home/camila/not_sync/Lancichinetti2017/Binary_2017_Lancichinetti2011_S/paj_Binary_2017_S/"
base_expected="/home/camila/not_sync/Lancichinetti2017/Binary_2017_Lancichinetti2011_S/com_Binary_2017_S/"
#errorQ_perc_array, k_est_array, k_expected_array=
main_teoric_error_lfr(base_folder, base_expected, "S")


base_folder="/home/camila/not_sync/Lancichinetti2017/Binary_2017_Lancichinetti2011_L/paj_Binary_2017_L/"
base_expected="/home/camila/not_sync/Lancichinetti2017/Binary_2017_Lancichinetti2011_L/com_Binary_2017_L/"
#errorQ_perc_array, k_est_array, k_expected_array=
main_teoric_error_lfr(base_folder, base_expected, "L")



#[print("plarg=",round(0.1+(p/10)),":",np.max(errorQ_perc_array[:,p])) for p in np.arange(0,10)]
#[print("plarg=",round(0.1+(p/10)),":",np.max(errorQ_perc_array[:,p])) for p in np.arange(0,10)]

#%% main_example
def main_teoric_error_example(base_folder, base_expected,char_split="\t"):
    n1_array = np.arange(1,2) #1,5+1
    n2_array = np.arange(1,2) #1,8+1
      
    p_eig_array = np.linspace(0.1,1,num=10)
      
    errorQ_array = np.zeros( (len(n1_array)*len(n2_array) , len(p_eig_array)) ,dtype=float)
    errorQ_perc_array = np.zeros( (len(n1_array)*len(n2_array), len(p_eig_array)) ,dtype=float)
      
    k_est_array = np.zeros( (len(n1_array)*len(n2_array), len(p_eig_array)) ,dtype=float)
    k_expected_array = np.zeros( (len(n1_array)*len(n2_array), len(p_eig_array)) ,dtype=float)
      
    l=0
  
    name='dolphins'
    file_graph=base_folder+name+".paj"
    file_expected=base_expected+name+"_expected.txt"
    
    errorQ_perc_array[l,:] = modularity_error(file_graph, file_expected, p_eig_array,char_split=char_split)
    k_est_array[l,:], k_expected_array[l,:] = community_number_error(file_graph, file_expected, p_eig_array,1,1,char_split)
    print(k_est_array)
        
    return errorQ_perc_array, k_est_array, k_expected_array

  
errorQ_perc_array, k_est_array, k_expected_array=main_teoric_error_example("grafos_nao_direcionados/", "grafos_expected/",char_split=' ')

#%% Teoric analysis

  
#%%% Teoric example: plot

## Eigenvalue distribution

def teoric_eigenvalues(g, mod_mat, plarg):
  zero_thr = 1e-15

  eig_val, eig_vec, eig_pos_ind, eig_neg_ind = sf.calc_eigen(g, mod_mat, plarg)

  thr = math.sqrt( max(eig_val) )
  k_est = len(eig_val[eig_val > math.sqrt( max(eig_val) )])
  print("Largest eigenvalues to calc: "+str(plarg) )
  print("Largest eigenvalue: " + str(max(eig_val)))
  print("Lowest eigenvalue: " + str(max(eig_val)))
  print("Estimation k: " + str(k_est))

  #TODO: Estimation k vs real communities

  #xx = np.linspace(0,100,101)   # consider x values 0, 1, .., 100
  #eigenvalues = np.array([np.sort(np.linalg.eigvals([[40,0,4],[0,0,4],[4,4,x]])) for x in xx])
  y=np.empty(len(eig_val))
  y.fill(0)
  print(y)

  print("eigenvalues'")
  print(sorted(eig_val, reverse=True))
  #print(eig_val)
  #print(eig_val == eig_val.real)

  print("* Zero eigenvalue exists?")
  print(eig_val[(eig_val<= zero_thr) & (eig_val >= -zero_thr)])
  print("...")
  #print ( ((eig_val <= zero_thr) & (eig_val >= -zero_thr)).any() )
  print(np.logical_and( eig_val <= zero_thr , eig_val >= -zero_thr).any())
  eig_zero = np.logical_and( eig_val <= zero_thr , eig_val >= -zero_thr)
  print(eig_zero)
  eig_zero_ind = np.where(eig_zero== True)
  print(eig_zero_ind[0])

  print("* 1 is an eigenvector associated with 0 eigenvalue?")
  #associated eigenvectors
  for c in eig_zero_ind[0]:
    print("ind:"+str(c))
    print(eig_vec[:,c])

#1 nao e um autovetor!

  return eig_val, thr

#teoric_eigenvalues(g, mod_mat, plarg)

#%%% Plot

def plot_eigvalues(eig_val, thr):
  #Plot
  #cont teoric_eigenvalues(g, mod_mat, plarg):

  figure, ax = plt.subplots()
  #draw_circle = plt.Circle((0, 0), thr)
  ax.set_xlim((min(eig_val)-1, max(eig_val)+1))
  ax.set_ylim((-thr-1, thr+1))
  draw_circle = plt.Circle((0, 0), thr, fill=False)
  
  ax.scatter(eig_val, y=np.zeros(len(eig_val)), label="Eigenvalues")
  ax.scatter([-thr,thr], [0,0], label="Threshold", color="red", marker="s")  

  ax.set_aspect(1)
  ax.add_artist(draw_circle)
  plt.title('Circle')
  plt.legend()
  plt.show()



#%%% Teoric example - 

def main_teoric(file_graph,file_expected,plarg,gamma1=1,gamma2=1,char_split="\t"):
  
    g = sf.read_construct_g(file_graph)
    mod_mat, _,_ = sf.construct_modularity_generalized(g,gamma1,gamma2)
    mod_mat = mod_mat.transpose() #por que?
 
    eig_val, thr = teoric_eigenvalues(g, mod_mat, plarg)
    plot_eigvalues(eig_val, thr)

    p_expected = sf.read_partition_expected(file_expected,char_split)
    #print(p_expected)

    p_eig_array = np.linspace(0.1,1,num=10)
    #p_eig_array = np.linspace(0.1,1,num=2)
    modularity_error(file_graph, file_expected, p_eig_array)

#%% Teoric: Call main
    
def call_main_teoric_example():
    base_folder="/home/camila/not_sync/Lancichinetti2017/Binary_2017_Lancichinetti2011_S/paj_Binary_2017_S/"
    base_expected="/home/camila/not_sync/Lancichinetti2017/Binary_2017_Lancichinetti2011_S/com_Binary_2017_S/"
    n1=6
    n2=1
    #file_graph=base_folder+"karate.net"
    #file_expected=base_expected+"karate_expected.txt"
    
    file_graph=base_folder+"net_Binary_2017_S1000_"+str(n1)+"_"+str(n2)+".net"
    file_expected=base_expected+"com_Binary_2017_S1000_"+str(n1)+"_"+str(n2)+".dat"
 
     
    main_teoric(file_graph, file_expected,plarg=0.1,gamma1=1,gamma2=1)

#call_main_teoric_example()