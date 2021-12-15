#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  4 20:10:05 2020

@author: camila
"""

#%% Import
import igraph
import numpy as np
import matplotlib.pyplot as plt
import math
import scipy.sparse.linalg
import random
import threading, queue
from sklearn.metrics.pairwise import cosine_similarity,cosine_distances


import tensorflow as tf
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers, losses
from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.models import Model

#%matplotlib inline
import matplotlib.pyplot as plt
#import seaborn as sns
import pandas as pd
import numpy as np
from pylab import rcParams
import tensorflow as tf
from tensorflow.keras.models import Model, load_model
from tensorflow.keras import layers
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard
from tensorflow.keras import regularizers
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, precision_recall_curve
from sklearn.metrics import recall_score, classification_report, auc, roc_curve
from sklearn.metrics import precision_recall_fscore_support, f1_score
from numpy.random import seed

#%% Partition

def create_edgelist_i0(file_graph):
    
  #print(n)
  #n=max(g.vs['Index'])+1
  
  f = open(file_graph, "r") 
  lines= f.readlines()
  
  file_out=file_graph+"_i0"
  f = open(file_out, "w")    


  vini=999999999
  for c in range(len(lines)):
      edge = np.array(  (lines[c].split('\n')[0]).split(" ") ).astype(int)
      #np.savetxt(f, edge.reshape(1, edge.shape[0]), newline='\n', fmt='%i')
      vini = min(vini, min(edge))
      
  for c in range(len(lines)):
      edge = np.array(  (lines[c].split('\n')[0]).split(" ") ).astype(int)-vini
      #np.savetxt(f, edge.reshape(1, edge.shape[0]), newline='\n', fmt='%i') 
      
      np.savetxt(f, edge, newline=' ', fmt='%i')
      f.write("\n")
      
  f.close()   
  return file_out

def create_edgelist_from_pajek(file_graph):
    
  #print(n)
  #n=max(g.vs['Index'])+1
  
    f = open(file_graph, "r") 
    lines= f.readlines()
    
    file_out=file_graph+"_edges_clean"
    f = open(file_out, "w")    
    
    
    vini=999999999
    start_edges = False
    for c in range(len(lines)):
 
        if start_edges == True:      
            edge = np.array(  (lines[c].split('\n')[0]).split(" ") ).astype(int)
            np.savetxt(f, np.append(edge,1), newline=' ', fmt='%i')
            f.write("\n")
      
        if "*Arcs" in lines[c]:
            #print("*Arcs, line:",c)
            start_edges = True      
        
      
    f.close()   
    return file_out

def read_partition_lines_OV(lines_file, g, char_split=' ', alg=''):
    n=g.vcount()
    f = open(lines_file, "r") 
    lines_in= f.readlines()
    lines = []
    #lines = lines_in
    Sov =np.zeros( (n, len(lines_in) ),dtype=int)
    
    
    #com_expected_lin = np.zeros( (len(lines_expected) ),dtype=int)
    c=0
    for lin in range(len(lines_in)):
        #skip rows starting with #
        if lines_in[lin].startswith("#") == False:            
            #vl indices of the expected file, must correct to the indices in the graph                      
            vl0 = np.array(  (lines_in[lin].split('\n')[0]).split(' ') )
            vl =  vl0[vl0!=''].astype(int)
            
            vindex = np.array(g.vs['orig'])
            for v in vl: 
                #print("...")
                #print(v)
                vg = np.where( vindex == v )[0] ##without -1
                if len(vg) > 0:
                    vg=vg[0]        
                    #print(vg,end=' ')
                    Sov[vg,c] = 1          
            lines.append(vl)
            #lines[c] = vl
            c=c+1
                  
    return Sov, lines
                  
        
        
def read_partition_expected_OV(file_expected, g,  char_split="\t", ignore_0=1):
    
  #print(n)
  #n=max(g.vs['orig'])+1
  n=g.vcount()
  f = open(file_expected, "r") 
  lines_expected = f.readlines()
  lines_expected_2 = lines_expected
  #print(lines_expected)
  com_expected=np.zeros( (n, len(lines_expected) ),dtype=int)
  com_expected_b=np.zeros( (n, len(lines_expected) ),dtype=int)
  
  #com_expected_lin = np.zeros( (len(lines_expected) ),dtype=int)
  if 'amazon' in file_expected or 'youtube' in file_expected:
      for c in range(len(lines_expected)):
          #vl indices of the expected file, must correct to the indices in the graph          
          vl0 = np.array(  (lines_expected[c].split('\n')[0]).split(' ') )
          vl =  vl0[vl0!=''].astype(int)
         
          vindex = np.array(g.vs['orig'])
          for v in vl: 
              #print("...")
              #print(v)
              vg = np.where( vindex == v )[0] ##without -1
              if len(vg) > 0:
                  vg=vg[0]        
                  #print(vg,end=' ')
                  com_expected[vg,c] = 1          
          lines_expected_2[c]=vl
      
         
            
  else:      
      #TODO: normalize communities
      for c in range(len(lines_expected)):
          #vl indices of the expected file, must correct to the indices in the graph          
          vl =  np.array(  (lines_expected[c].split('\n')[0]).split(char_split) )[1:].astype(int)
          
          #print("comm", c)
          #print(vl)
          #print("vg")
          
          #todo: check if vertex exists in graph.. vertex indices now differ
          vindex = np.array(g.vs['orig'])
          for v in vl: 
              #print("...")
              #print(v)
              vg = np.where( vindex == v )[0] ##without -1
              if len(vg) > 0:
                  vg=vg[0]        
                  #print(vg,end=' ')
                  com_expected[vg,c] = 1          
          lines_expected_2[c]= np.array(  (lines_expected_2[c].split('\n')[0]).split(char_split) )[1:-1].astype(int)
      
        
      if len(lines_expected_2[len(lines_expected)-1 ])==0:
              lines_expected_2=lines_expected_2[:-1]
      
  
      #p_expected[i] = 
   
    #TODO: " " for real graphs 

  # for v in range(n):
  #     for c in range(len(lines_expected)):           
  #         print(com_expected[v-1,c])
      
  return com_expected, lines_expected_2

def read_partition_expected(file_expected, char_split="\t"):
  f = open(file_expected, "r") 
  lines_expected = f.readlines()
  #print(lines_expected)
  p_expected=np.zeros(len(lines_expected),dtype=int)
  
  #TODO: normalize communities
  for i in range(len(lines_expected)):
    #print("v" + str(i+1) + ", " + str( lines_expected[i].split("\t") ))    
    #print(lines_expected[i].split("\t")[1].split('\n')[0])
    #\t or blank? p_expected[i] = int( lines_expected[i].split(" ")[1].split('\n')[0])
    if char_split=='\n': #300321
        p_expected[i] = int(lines_expected[i].split('\n')[0])
    else:
        p_expected[i] = int(lines_expected[i].split(char_split)[1].split(char_split)[0])
    
    #TODO: " " for real graphs 
    
    #print(p_expected[i])
  #print('min=',min(p_expected))
  if min(p_expected) > 0:     
      p_expected-=1
      
  return p_expected

def calc_Q_partition(g, mod_mat, comms_est, plarg, part, which="LM"): #290321
  eig_val, eig_vec, eig_pos_ind, eig_neg_ind = calc_eigen(g,mod_mat, plarg)

  ri_vert_pos = calc_ri_vert_pos(g, eig_val, eig_vec, eig_pos_ind)
  ri_vert_neg = calc_ri_vert_neg(g, eig_val, eig_vec, eig_neg_ind)

  R_group_pos = np.zeros(( comms_est, len(eig_pos_ind) ))
  R_group_neg = np.zeros(( comms_est, len(eig_neg_ind) ))

  #OPTIMIZE
  for i in range(g.vcount()):  
    #print(p[i])
    R_group_pos, R_group_neg = insert_vertex_R_group(R_group_pos, R_group_neg, part[i], ri_vert_pos, ri_vert_neg, i)

  valQ = calc_Q_Rgroup(R_group_pos, R_group_neg, 2*np.sum(g.strength(weights='weight'))/2)
  
  return valQ


#%% 
#test 
# adj = np.matrix(g.get_adjacency().data)
 
#%%# Igraph - Construct graph

def read_construct_g(file_graph, bool_isolated=False):
  if "edges" in file_graph:
    #print("read edges")
    g = igraph.read(file_graph, format="edgelist",directed=False)
    #g1=igraph.Graph.Read_Edgelist(file_graph)
  else:
    g = igraph.read(file_graph, format="pajek")
  
  g.to_undirected()  #251120
  
  
  #250921 g = g.simplify() #should not simplify? 250921
  g = g.simplify(loops=False) #250911
  
  #160921
  
  #??

  
  v_inds=list(range(0,g.vcount(),1))
  g.vs[v_inds]['orig']=v_inds
  
  #g.vs[v_inds]['name']=v_inds
  
  if "edges" in file_graph and bool_isolated==False:
      g.delete_vertices(np.where(np.array(g.degree())==0)[0])
      # g.delete_vertices(0)
    

  v_inds=list(range(0,g.vcount(),1))

  g.vs[v_inds]['Index']=v_inds
  
  #print(g.vs[33])
  #g.vs['Index']

  if "edges" in file_graph:
      g.write_pajek(file_graph+'_clean')
      
      
  if "weight" not in g.es.attribute_names():
    #print("set weight = 1")
    g.es['weight'] = 1

  #print("graph attributes")
  #print(g.vs.attribute_names())
  #print(g.es.attribute_names())  
  #g.es['weight'] 

  return g

#%% modularity generalized
def construct_modularity_cosine(g,gamma1=1,gamma2=1,gamma3=1,par_eta=5, T=None):
    print("-- NEW MOD COSINE..")
    adj = g.get_adjacency(attribute="weight")
    adj_mat = np.matrix(adj.data) 
 
    #-- Cosine
    S1 = np.array(adj.data)    
    S2 = cosine_similarity(S1)
    S=S2

    # #-- Modularity
    print("gamma1=",gamma1,", gamma2=",gamma2, ", gamma3=",gamma3)
    #--- T 
    if T is not None:
        print("T shape")
        print(T.shape)

        #061121
        TS = np.dot(T, T.transpose()) 
        TS = 100*TS /g.vcount()
        adj_mat = 100*adj_mat / g.vcount()
        S = 100*S /g.vcount()
        
        adj_cos_add = np.add( np.add( gamma1* adj_mat, gamma3*TS ), gamma2*S) 
        adj_cos_add = adj_cos_add 
    else: 
        adj_cos_add = np.add( gamma1* adj_mat, gamma2*S ) 

    
    g_cos = igraph.Graph.Weighted_Adjacency(adj_cos_add.tolist(), attr="weight", mode=igraph.ADJ_UNDIRECTED) #replace graph with the adjusted for ensemble
    adj_cos = np.matrix( g_cos.get_adjacency(attribute="weight").data) 

    degree = np.matrix(g_cos.strength(weights='weight'),dtype=float)
    
    
    g_cos.vs['orig'] = g.vs['orig']
    g_cos.vs['Index'] = g.vs['Index']
 
    
    null_model = np.dot(degree.transpose(), degree)/np.sum(degree)
    
    
    mod_mat = np.subtract(adj_cos_add, null_model) 
    

  
    return g_cos,mod_mat, adj_cos, null_model

def construct_cosine(g,par_eta=5):
    #-- Modularity
    adj = g.get_adjacency(attribute="weight")
    #-- Cosine
    S1 = np.array(adj.data)    
    S2 = np.zeros((g.vcount(), g.vcount()))
    
    print("v...",end='')
    for i in np.arange(0,g.vcount()):
        print(i,", ", end='')
        for j in np.arange(0,g.vcount()):
            S2[i,j] = cosine_similarity(S1[i,].reshape(1,-1),S1[j,].reshape(1,-1))
            S2[j,i] = S2[i,j]
    
    print("\n")
    #S = S1 + par_eta * S2
    S=S2
    
    print(S)
    return S


def construct_modularity_cosine_param(g,S2,gamma1=1,gamma2=1):
    #-- Modularity
    adj = g.get_adjacency(attribute="weight")
    adj_mat = np.matrix(adj.data) #041120
    
    #print(adj_mat)
    degree = np.matrix(g.strength(),dtype=float)
    #print(degree)
    
    #null_model = np.matmul(degree.transpose(), degree)/sum(g.strength()) 
    null_model = np.dot(degree.transpose(), degree)/sum(g.strength())  #041120
    #print(np.matrix.sum(degree))
    
    #mod_mat = adj_mat - null_model
    mod_mat_orig = np.subtract(adj_mat, null_model) 

  
    mod_mat = gamma1*mod_mat_orig + gamma2*S2
  
    return mod_mat

#%% Modularity matrix
def construct_modularity_generalized(g,gamma1=1,gamma2=1):
  adj = g.get_adjacency(attribute="weight")
  adj_mat = gamma1*np.matrix(adj.data) #041120

  #print(adj_mat)
  degree = np.matrix(g.strength(),dtype=float)
  #print(degree)

  #null_model = np.matmul(degree.transpose(), degree)/sum(g.strength()) 
  null_model = gamma2*np.dot(degree.transpose(), degree)/sum(g.strength())  #041120
  #print(np.matrix.sum(degree))
  
  #mod_mat = adj_mat - null_model
  mod_mat = np.subtract(adj_mat, null_model) 

  #print(null_model-adj)
  #print(mod_mat)
  
  return mod_mat,adj_mat, null_model
#%% Modularity matrix
def construct_modularity_generalized(g,gamma1=1,gamma2=1):
  #adj = g.get_adjacency(attribute="weight")
  #adj_mat = gamma1*np.matrix(adj.data) #041120

  adj_mat = gamma1*np.array(g.get_adjacency(attribute="weight").data)

  #print(adj_mat)
  degree = np.matrix(g.strength(),dtype=float)
  #print(degree)

  null_model = np.array( gamma2*np.dot(degree.transpose(), degree)/sum(g.strength()) ) #041120 
  
  #mod_mat = adj_mat - null_model
  mod_mat = np.subtract(adj_mat, null_model) 

  #print(null_model-adj)
  #print(mod_mat)
  
  return mod_mat,adj_mat, null_model
#construct_modularity_generalized(g,gamma1,gamma2)
#print(mod_mat)

def construct_null_model(g, gamma2=1):
    degree = np.matrix(g.strength(),dtype=float)

    null_model = gamma2*np.dot(degree.transpose(), degree)/sum(g.strength())  #041120
    
    return null_model

def construct_null_model_array(g, gamma2=1): #010220
    degree = np.array(g.strength(),dtype=float)

    null_model = gamma2*np.dot(degree.transpose(), degree)/sum(g.strength())  #041120
    
    return null_model
#%% Non-backtracking matrix

def construct_nnf_matrix(g,gamma1=1):
    g_lin = g.linegraph()
       
    if "weight" not in g_lin.es.attribute_names():
        #print("set weight = 1")
        g_lin.es['weight'] = 1
 
    #TODO: line graph does not have any attributes...
    adj = g_lin.get_adjacency(attribute="weight")
    
    adj_mat = gamma1*np.matrix(adj.data) #041120
    adj_mat = adj_mat.astype("double")
    
    return g_lin, adj_mat
    

#%% NMI 

def calc_nmi(com1, com2):
    #g.compare_communities
    #print("com1")
    #print(com1)
    #print("com2")
    #print(com2)
    nmi = igraph.compare_communities(com1, com2, method='nmi')
    #print("nmi=",nmi)
    return nmi

#%% Autoencoder
def autoencoder_mod(x_train,x_test):

    #(x_train, _), (x_test, _) = fashion_mnist.load_data()
    
    
    #x_train = x_train.astype('float32') / 255.
    #x_test = x_test.astype('float32') / 255.

    
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32') 

    
    print (x_train.shape)
    print (x_test.shape)
    
    latent_dim = 120 #x_train.shape[0]
    
    class Autoencoder(Model):
      def __init__(self, latent_dim):
        super(Autoencoder, self).__init__()
        self.latent_dim = latent_dim   
        self.encoder = tf.keras.Sequential([
          layers.Flatten(),
          #layers.Dense(latent_dim, activation='relu'),
        ])
        self.decoder = tf.keras.Sequential([
          layers.Dense(x_train.shape[1]*x_train.shape[1], activation='sigmoid'),
          layers.Reshape((x_train.shape[1], x_train.shape[1]))
        ])
    
      def call(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded
    
    autoencoder = Autoencoder(latent_dim)
    
    #% train
    
    autoencoder.compile(optimizer='adam', loss=losses.MeanSquaredError())
    
    
    autoencoder.fit(x_train, x_train,
                    epochs=20,
                    shuffle=True,
                    validation_data=(x_test, x_test))
    
    
    #%
    encoded_vec = autoencoder.encoder(x_test).numpy()
    decoded_vec = autoencoder.decoder(encoded_vec).numpy()
    
    #%
    # n = 10
    # plt.figure(figsize=(20, 4))
    # for i in range(n):
    #     # display original
    #     ax = plt.subplot(2, n, i + 1)
    #     plt.imshow(x_test[i])
    #     plt.title("original")
    #     plt.gray()
    #     ax.get_xaxis().set_visible(False)
    #     ax.get_yaxis().set_visible(False)
      
    #     # display reconstruction
    #     ax = plt.subplot(2, n, i + 1 + n)
    #     plt.imshow(decoded_imgs[i])
    #     plt.title("reconstructed")
    #     plt.gray()
    #     ax.get_xaxis().set_visible(False)
    #     ax.get_yaxis().set_visible(False)
    
    # plt.show()
    
    return encoded_vec, decoded_vec

# def autoencoder_mod_2(input_img)
# input_size = 784
# hidden_size = 128
# code_size = 32

# input_img = Input(shape=(input_size,))
# hidden_1 = Dense(hidden_size, activation='relu')(input_img)
# code = Dense(code_size, activation='relu')(hidden_1)
# hidden_2 = Dense(hidden_size, activation='relu')(code)
# output_img = Dense(input_size, activation='sigmoid')(hidden_2)

# autoencoder = Model(input_img, output_img)
# autoencoder.compile(optimizer='adam', loss='binary_crossentropy')
# autoencoder.fit(x_train, x_train, epochs=5)


#%% Autoencoder 2 
def autoencoder_2(df_train, df_valid):
    # scaler = StandardScaler().fit(df_train)
    # df_train_0_x_rescaled = scaler.transform(df_train)
    # df_valid_0_x_rescaled = scaler.transform(df_train)
    # df_valid_x_rescaled = scaler.transform(df_valid.drop(['y'], axis = 1))
    # df_test_0_x_rescaled = scaler.transform(df_test_0_x)
    # df_test_x_rescaled = scaler.transform(df_test.drop(['y'], axis = 1))

    nb_epoch = 200
    batch_size = 128
    input_dim = df_train.shape[1] #num of predictor variables, 
    encoding_dim = input_dim/10 #size of the embedding
    hidden_dim = int(encoding_dim / 2)
    learning_rate = 1e-3
    
    input_layer = Input(shape=(input_dim, ))
    encoder = Dense(encoding_dim, activation="relu", activity_regularizer=regularizers.l1(learning_rate))(input_layer)
    encoder = Dense(hidden_dim, activation="relu")(encoder)
    decoder = Dense(hidden_dim, activation="relu")(encoder)
    decoder = Dense(encoding_dim, activation="relu")(decoder)
    decoder = Dense(input_dim, activation="linear")(decoder)
    autoencoder = Model(inputs=input_layer, outputs=decoder)
    autoencoder.summary()
    
    #---
    autoencoder.compile(metrics=['accuracy'], loss='mean_squared_error', optimizer='adam')
    cp = ModelCheckpoint(filepath="autoencoder_classifier.h5",  save_best_only=True,      verbose=0)
    tb = TensorBoard(log_dir='./logs',
                    histogram_freq=0,
                    write_graph=True,
                    write_images=True)
    
    history = autoencoder.fit(df_train, df_train,
                        epochs=nb_epoch,
                        batch_size=batch_size,
                        shuffle=True,
                        validation_data=(df_valid, df_valid),
                        verbose=1,
                        callbacks=[cp, tb]).history
    

    mod_dec = autoencoder.predict(df_valid)
# mse = np.mean(np.power(df_valid - mod_dec, 2), axis=1)
# error_df = pd.DataFrame({'Reconstruction_error': mse,
#                         'True_class': df_valid['y']})

 #   encoded_vec = autoencoder.encoder(df_valid).numpy()
#    decoded_vec = autoencoder.decoder(encoded_vec).numpy()
    
 

    f1_macro, f1_micro= node_classification(mod_dec, p_expected, test_size=0.3)

    from sklearn.cluster import KMeans
    kmeans = KMeans(n_clusters=7, random_state=0).fit(mod_enc)
    kmeans.labels_
        
    nmi = sf.calc_nmi(kmeans.labels_, p_expected)

#%% Autoencoder mod
    
#2
def construct_modularity_encoder(g,gamma1=1,gamma2=1):
    print("Modularity encoder 2")
    
    #-- Modularity
    adj = g.get_adjacency(attribute="weight")
    adj_mat = np.array(adj.data) #041120

    adj_3d = np.zeros(( 1, g.vcount(), g.vcount() ))
    adj_3d[0] = adj_mat
    A_enc, A_dec = autoencoder_mod(adj_3d, adj_3d)  
    A_dec = A_dec[0]
        


    degree = adj_mat.sum(axis=1)

    #print(adj_mat)
    #degree = np.matrix(g.strength(),dtype=float)
    #print(degree)
    
    #null_model = np.matmul(degree.transpose(), degree)/sum(g.strength()) 
    null_model = np.dot(degree.transpose(), degree)/sum(degree)  #041120
    #print(np.matrix.sum(degree))
    
    #mod_mat = adj_mat - null_model
    mod_mat = np.subtract(A_dec, null_model) 


    #mod_dec = mod_dec[0]

    return mod_mat
   
def construct_modularity_encoder_1(g,gamma1=1,gamma2=1):
    print("Modularity encoder ")
    
    #-- Modularity
    adj = g.get_adjacency(attribute="weight")
    adj_mat = np.matrix(adj.data) #041120
    
    #print(adj_mat)
    degree = np.matrix(g.strength(),dtype=float)
    #print(degree)
    
    #null_model = np.matmul(degree.transpose(), degree)/sum(g.strength()) 
    null_model = np.dot(degree.transpose(), degree)/sum(g.strength())  #041120
    #print(np.matrix.sum(degree))
    
    #mod_mat = adj_mat - null_model
    mod_mat = np.subtract(adj_mat, null_model) 

    mod_mat_3d = np.zeros(( 1, g.vcount(), g.vcount() ))
    mod_mat_3d[0] = mod_mat

    mod_enc, mod_dec = autoencoder_mod(mod_mat_3d, mod_mat_3d)  
    mod_dec = mod_dec[0]

    return mod_dec
#%%  test


def test_autoencoder():
    adj_3d = np.zeros(( 1, g.vcount(), g.vcount() ))
    adj_3d[0] = adj_mat
    A_enc, A_dec = autoencoder_mod(adj_3d, adj_3d)  
    A_dec = A_dec[0]
        
    adj = g.get_adjacency(attribute="weight")
    adj_mat = np.array(adj.data) #041120
 
    A_enc, A_dec = autoencoder_mod(adj_mat, adj_mat)  
    
    
    f1_macro, f1_micro= node_classification(A_dec, p_expected, test_size=0.3)
    
    
    ###
    mo_weight_Qin=0.8
    mo_weight_Qnull=0.2
    S= sf.construct_cosine(graph.g)
    mod_mat = sf.construct_modularity_cosine_param(graph.g,S, mo_weight_Qin, mo_weight_Qnull).transpose()        
 
    
    mod_mat_3d = np.zeros(( 1, g.vcount(), g.vcount() ))
    mod_mat_3d[0] = mod_mat

    mod_enc, mod_dec = autoencoder_mod(mod_mat_3d, mod_mat_3d)  
    mod_dec = mod_dec[0]
    
    f1_macro, f1_micro= node_classification(mod_dec, p_expected, test_size=0.3)

    from sklearn.cluster import KMeans
    kmeans = KMeans(n_clusters=5, random_state=0).fit(mod_enc)
    kmeans.labels_
    
    
    nmi = sf.calc_nmi(kmeans.labels_, p_expected)
    print(nmi)


#%%# Spectral - Eigenvalues
#Outra opção
#https://igraph.org/r/doc/embed_adjacency_matrix.html

def calc_eigen(g,mod_mat, plarg, which="LM"):
    # ‘LM’ : largest magnitude
    # ‘SM’ : smallest magnitude
    # ‘LR’ : largest real part
    # ‘SR’ : smallest real part
    # ‘LI’ : largest imaginary part
    # ‘SI’ : smallest imaginary part

    if plarg<1:      
      plarg_n = math.floor(plarg * g.vcount() )
      #print("calc eigen", plarg_n)     
      #eig_val, eig_vec = scipy.sparse.linalg.eigs(mod_mat, k=plarg_n, which=which)      
      eig_val, eig_vec = scipy.sparse.linalg.eigs(mod_mat, k=plarg_n, which=which, tol=1E-3) #290921 #260321     - estava nesse

      
      print("calc eigenvalues with tol")
      
    else:
      eig_val, eig_vec = scipy.linalg.eigh(mod_mat)
      
    eig_val = eig_val.real
    eig_vec = eig_vec.real
    
    eig_pos_ind = np.where(eig_val >= 0)
    eig_neg_ind = np.where(eig_val < 0)
      
    return eig_val, eig_vec, eig_pos_ind[0], eig_neg_ind[0]


def calc_eigen_mat(mod_mat, plarg, which="LM"):
    # ‘LM’ : largest magnitude
    # ‘SM’ : smallest magnitude
    # ‘LR’ : largest real part
    # ‘SR’ : smallest real part
    # ‘LI’ : largest imaginary part
    # ‘SI’ : smallest imaginary part

    if plarg<1:      
      plarg_n = math.floor(plarg * mod_mat.shape[0] )
      #print(which)
      #print("calc eigen", plarg_n)     

      eig_val, eig_vec = scipy.sparse.linalg.eigs(mod_mat, k=plarg_n, which=which)
    else:
      eig_val, eig_vec = scipy.linalg.eigh(mod_mat)
      
    eig_val = eig_val.real
    eig_vec = eig_vec.real
    
    eig_pos_ind = np.where(eig_val >= 0)
    eig_neg_ind = np.where(eig_val < 0)
      
    return eig_val, eig_vec, eig_pos_ind[0], eig_neg_ind[0]



#%% Number of communities

def est_number_communities(eig_val):
    #print(eig_val)
    k = len(eig_val[eig_val > math.sqrt( max(eig_val) )])    
    if k <=1 :
        k=2
    else:      
		    #Maximum number, because the methods might not use all groups
        k=1.25*k
    k=int(np.floor(k))
    return k

#%%# Spectral - Vector partitioning

def calc_Q_Rgroup(R_group_pos, R_group_neg, divtotal):
  valQ=0
  # print("num_groups="+str(np.shape(R_group_pos)[0]))
  for s in range(0,np.shape(R_group_pos)[0]):
    valQ= valQ + np.inner( R_group_pos[s,:], R_group_pos[s,:] ) - np.inner( R_group_neg[s,:], R_group_neg[s,:] )  

  valQ = valQ/divtotal  
  return valQ

def calc_ri_vert_pos(g, eig_val, eig_vec, eig_pos_ind):
  eig_val_pos = eig_val[eig_pos_ind]
  eig_vec_pos = eig_vec[:,eig_pos_ind]

  ri_vert_pos = np.multiply(np.sqrt(eig_val_pos) , eig_vec_pos)

  return ri_vert_pos

def calc_ri_vert_neg(g, eig_val, eig_vec, eig_neg_ind):
  eig_val_neg = eig_val[eig_neg_ind]
  eig_vec_neg = eig_vec[:,eig_neg_ind]

  sqrt_val_term = np.zeros(( g.vcount(), len(eig_neg_ind) ))

  ri_vert_neg = np.multiply(np.sqrt(-eig_val_neg), eig_vec_neg)

  return ri_vert_neg

def insert_vertex_R_group(R_group_pos, R_group_neg, t_best, ri_vert_pos, ri_vert_neg,v):
  R_group_pos[t_best,:] += ri_vert_pos[v,:]
  R_group_neg[t_best,:] += ri_vert_neg[v,:]

  return R_group_pos, R_group_neg
  
def move_vertex_R_group(R_group_pos, R_group_neg, t_best, t_old, ri_vert_pos, ri_vert_neg, v):
  #Remove contribution from the old community
  R_group_pos[t_old,:] -= ri_vert_pos[v,:]
  R_group_neg[t_old,:] -= ri_vert_neg[v,:]

  #Add contribution to the new community
  R_group_pos[t_best,:] += ri_vert_pos[v,:]
  R_group_neg[t_best,:] += ri_vert_neg[v,:]
  
  return R_group_pos, R_group_neg  

def move_vertex_ov_R_group(R_group_pos, R_group_neg, t_best, b_best, deltaS_best, ri_vert_pos, ri_vert_neg, v):
  R_group_pos[b_best,:] -= ri_vert_pos[v,:]*deltaS_best
  R_group_neg[b_best,:] -= ri_vert_neg[v,:]*deltaS_best
  
  R_group_pos[t_best,:] += ri_vert_pos[v,:]*deltaS_best
  R_group_neg[t_best,:] += ri_vert_neg[v,:]*deltaS_best
  
  return R_group_pos, R_group_neg

def calc_gainQ_Rgroup_insert(R_group_pos, R_group_neg, t, ri_vert_pos, ri_vert_neg, v):
  val = np.inner( R_group_pos[t,:], ri_vert_pos[v,:] ) - np.inner( R_group_neg[t,:], ri_vert_neg[v,:] )
  return val

def calc_gainQ_Rgroup_update(R_group_pos, R_group_neg, t_new, t_old, ri_vert_pos, ri_vert_neg, v):  
  val = np.inner( R_group_pos[t_new,:], ri_vert_pos[v,:] ) - np.inner( R_group_neg[t_new,:], ri_vert_neg[v,:] )
  val -= np.inner( R_group_pos[t_old,:]-ri_vert_pos[v,], ri_vert_pos[v,:] ) - np.inner( R_group_neg[t_old,:]-ri_vert_neg[v,], ri_vert_neg[v,:] )    
 
  return val  

def calc_gainQ_ov_Rgroup_update(R_group_pos, R_group_neg, t_new, t_old, deltaS, ri_vert_pos, ri_vert_neg, v):
  val = deltaS* np.inner( R_group_pos[t_new,:], ri_vert_pos[v,:] ) - deltaS*np.inner( R_group_neg[t_new,:], ri_vert_neg[v,:] ) 
  val += np.inner( deltaS*ri_vert_pos[v,:] ,deltaS* ri_vert_pos[v,:] ) #161121
  val -= deltaS* np.inner( R_group_pos[t_old,:], ri_vert_pos[v,:] ) - deltaS*np.inner( R_group_neg[t_old,:], ri_vert_neg[v,:] )
  val -= np.inner( deltaS*ri_vert_neg[v,:] ,deltaS* ri_vert_neg[v,:] ) #161121

  return val

#%%% Spectral - genetic aux

def calc_gainQ_Rgroup_move(R_group_pos, R_group_neg, t_new, t_old, ri_vert_pos, ri_vert_neg, v):  
  val = np.inner( R_group_pos[t_new,:], ri_vert_pos[v,:] ) - np.inner( R_group_neg[t_new,:], ri_vert_neg[v,:] )  
 
  val -= np.inner( R_group_pos[t_old,:], ri_vert_pos[v,:] ) - np.inner( R_group_neg[t_old,:], ri_vert_neg[v,:] )    

  #TODO: conferir aqui, porque eu nao faço isso no update?
  val += np.inner(ri_vert_pos[v,:],ri_vert_pos[v,:]) -  np.inner(ri_vert_neg[v,:],ri_vert_neg[v,:])   

  return val  

#%% Initial partition


def spec_initial_partition(g, R_group_pos, R_group_neg, ri_vert_pos, ri_vert_neg, eig_pos_ind, eig_neg_ind, comms_est, bool_verbose):

  
  allv = np.arange(0,g.vcount(),1)
  centroids = np.random.choice(allv, comms_est, replace=False)  
  allv_no_centroids = np.setdiff1d(allv,centroids)

  Sov = np.zeros((g.vcount(), comms_est))
  
  pcurr = np.zeros(g.vcount(),dtype=int)
  pcurr[:] = -1
  pcurr[centroids] = np.arange(0,comms_est)

  set_comm = np.arange(0,comms_est)
  
  # Verbose:
  if bool_verbose==True:
    #print("allv")
    #print(allv)

    print("centroids: ",centroids)
    # print("allv without centroids")
    # print(allv_no_centroids)
    #print("initial partition with centroids")
    #print(pcurr)
    # print("set_comm")
    # print(set_comm)

  for c in set_comm:  
    R_group_pos, R_group_neg = insert_vertex_R_group(R_group_pos, R_group_neg, c, ri_vert_pos, ri_vert_neg, centroids[c])
    Sov[centroids[c],c] = 1  

  valQ = calc_Q_Rgroup(R_group_pos, R_group_neg, 2*np.sum(g.strength(weights='weight'))/2)
  allv_no_centroids_sample = np.random.choice(allv_no_centroids, len(allv_no_centroids), replace=False)

  #if bool_verbose==True:
    # print("CalcQ, centroids ="+str(valQ))  
    # print("allv_no_centroids sample:")  
    # print(allv_no_centroids_sample)

  for v in allv_no_centroids_sample:    
    val_best = -100
    t_best = -1
    
    for t in set_comm:
      val = calc_gainQ_Rgroup_insert(R_group_pos, R_group_neg, t, ri_vert_pos, ri_vert_neg, v)

      if val>val_best or t_best==-1:
          val_best = val
          t_best = t
      #print("v="+str(v)+", t_best"+str(t_best))          
    pcurr[v] = t_best
    Sov[v,t_best]=1

    R_group_pos, R_group_neg = insert_vertex_R_group(R_group_pos, R_group_neg, t_best, ri_vert_pos, ri_vert_neg, v)
      
  valQ = calc_Q_Rgroup(R_group_pos, R_group_neg, 2*np.sum(g.strength(weights='weight'))/2)
  if bool_verbose == True:
      conf_Q_vec = g.modularity(pcurr, g.es['weight'])
      print("CalcQ, initial ="+str(valQ))  
      print("Q:"+ str(valQ) + ", Q_igraph="+str(conf_Q_vec))

  #conf_Q_vec = g.modularity(pcurr, g.es['weight'])
  #print("CalcQ, initial ="+str(valQ) + " Q:"+ str(valQ) + ", Q_igraph="+str(conf_Q_vec))
        
  return R_group_pos, R_group_neg, Sov, pcurr, valQ

#%% Local search

#This is going to be part of the genetic algorithm..
def spec_local_search(g, ri_vert_pos, ri_vert_neg, eig_pos_ind, eig_neg_ind, comms_est, R_group_pos, R_group_neg, Sov, pcurr, it_max=10,bool_verbose=False):
  it_ls=1
  bool_improve=True

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
        val = calc_gainQ_Rgroup_update(R_group_pos, R_group_neg, t, t_old, ri_vert_pos, ri_vert_neg, v)
        if val>0 and (val>val_best or t_best==-1):          
          #print("*** found new best, Check " + str(v) + " to community " + str(t) + ", val=" + str(val), ", val_best" + str(val_best) + ", t_best=", t_best)     
          val_best = val
          t_best = t 

      if t_best>-1 and t_best!= t_old: #just checking again for t_best!=t_old
        #move vertex v to community t_best
        #if bool_verbose:
          #print("Move " + str(v) + " to community " + str(t_best))

        pcurr[v] = t_best
        Sov[v,t_old]=0
        Sov[v,t_best]=1
        R_group_pos, R_group_neg = move_vertex_R_group(R_group_pos, R_group_neg, t_best, t_old, ri_vert_pos, ri_vert_neg, v)
        bool_improve = True

      #all valid communities, disconsidering old_o      
    it_ls+=1

  #print("it ls=",it_ls)
  valQ = calc_Q_Rgroup(R_group_pos, R_group_neg, 2*np.sum(g.strength(weights='weight'))/2)
  
  if bool_verbose==True:
    #print(".. LS, IT = "+str(it_ls))
    conf_Q_vec = g.modularity(pcurr, g.es['weight'])      
    print("..Q-LS:"+ str(valQ)  + ", Q_igraph-LS="+str(conf_Q_vec))
    
  return R_group_pos, R_group_neg, pcurr, Sov, valQ

#pcurr, Sov, valQ = local_search(g, ri_vert_pos, ri_vert_neg, eig_pos_ind, eig_neg_ind, comms_est, R_group_pos, R_group_neg, Sov, pcurr)  
#print(pcurr) 
#print(valQ)



#%% Part from Sov
def create_part_from_Sov(Sov):
    #Sov.shape
    part = np.zeros(Sov.shape[0])
    for v in np.arange( Sov.shape[0]):
        part[v] = np.argmax(Sov[v,:])

    return part