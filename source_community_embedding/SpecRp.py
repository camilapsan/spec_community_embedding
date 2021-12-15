import time
import numpy as np 

from pathlib import Path
import sys
path = str(Path(Path(__file__).parent.absolute()).parent.absolute())
sys.path.insert(0, path)

import metrics as metrics 
import spectral_functions as SF
import MOSpecG_partitioning_OV as SpecG
import SpecOV_partitioning_v2 as SpecOV


#import all the dependencies
from keras.layers import Dense,Conv2D,MaxPooling2D,UpSampling2D
from keras import Input, Model
from keras.datasets import mnist
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers, losses

import pandas as pd 

bool_verbose=False
out_verbose=None
    

#%% SpecRp - Embedding + Autoencoders

def emb_Rp_Attr_main( it_emb, MO, IT, p_leading, file_graph, file_expected, results_folder, file_com_out, time_folder, pareto_file, num_pareto, ensemble_thr, n_gen, n_pop, p_offs, bool_ov, ov_thr=0,op_com=1,dim_emb=2, char_split=' ',name='', op_var='pdimemb',  lambdav = 1e9,  alpha=0.5, beta=5,bool_link_pred=False,bool_proposed=True,gamma1=0.4,gamma2=0.3,gamma3=0.3, opDeltaH=0):
    it_spec=IT
    pos_neg=False
    
    print("-- op_var", op_var)
    start_time_g = time.perf_counter()    
    time_g = (time.perf_counter() - start_time_g)        
    g = SF.read_construct_g(file_graph, bool_link_pred)    #bool_link_pred==True, isolated also True
    #print("vert = ", g.vcount(), "number of edges = ", g.ecount())
    
    SF.create_edgelist_from_pajek(file_graph)
    p_best=None
    print("-- bool_ov=",bool_ov)
    #---------------------------- results
    if file_expected is not None:        
        #---- Overlapping communities
        if bool_ov==True and file_expected is not None: #TEST_DISJ_OV==False and 
            print("Read OV")
            com_expected,lin_expected  = SF.read_partition_expected_OV(file_expected, g, char_split="\t")
            p_expected=None
            k_expected = len(lin_expected)
        #---- Disjoint communities   
        elif bool_ov==False and file_expected is not None:   #TEST_DISJ_OV==True or 
            print("Read Disj")
            p_expected = SF.read_partition_expected(file_expected,char_split)
            k_expected = len(pd.unique(p_expected))
        print("-- # Number of expected communities = ", k_expected)
    else:
        p_expected=None
        
    #------------ Solve initial problems 

    start_time_spec = time.perf_counter()
    
    T = read_attributes_T(name, g, p_expected, dim_emb)
    print("op_com=", op_com)
    
    if op_com=="SpecDecOV":
        nome_alg="SpecDecOV-Auto"      
        q_best, Sov_best, g, eig_val, ri_vert_pos, ri_vert_neg, Rp, Rn, mod_mat,A,P = SpecOV.spec_ov_main(file_graph, it_spec, p_leading, gamma1, gamma2, gamma3, ov_thr, bool_ov, file_com_out, bool_verbose, out_verbose,KNumber_pre=-1,mod_orig=False,T=T, opDeltaH=opDeltaH)      
     
    elif op_com=="SpecG":        
        nome_alg="AutoSpec"            
        ##Genetic algorithm
        p_best, graph, params, eig, pop, offs, Sov_best, Rp,Rn,list_gen_Q,mod_mat, A, P = SpecG.SpecG_single_main(False, it_spec, p_leading, file_graph, results_folder, time_folder, pareto_file, num_pareto, ensemble_thr, n_gen, n_pop, p_offs, bool_ov,
                                                               mo_weight_Qin=gamma1, mo_weight_Qnull =gamma2,gamma3=gamma3,mod_orig=False,KNumber_pre=-1,pos_neg=pos_neg,bool_link_pred=bool_link_pred,T=T)            
        print("----------------------------")
        print(nome_alg)
        p_best=p_best[0]
        Sov_best=Sov_best[0]
        Rp=Rp[0]
        Rn=Rn[0]
        ri_vert_pos = graph.ri_vert_pos
        ri_vert_neg = graph.ri_vert_neg
        g=graph.g
        
        print(g.vs.attribute_names())
        A=A.astype(float)
        
    print("* dim_emb=",dim_emb)
    dim_p = min(dim_emb,Rp.shape[1]) 
    
    ##new: embedding 290921
    ri_vert_pos = ri_vert_pos[:,0:dim_p ]
    Rp = Rp[:,0:dim_p ]
    # print("Rp ", Rp.shape)
    # print("Rn ", Rn.shape)

    time_spec = (time.perf_counter() - start_time_spec) + time_g 
    H = Sov_best         

    print("DIM Rp:",Rp.shape[1])
        
    H0 = H.copy()
    if T is not None:
        T0 = T.copy()
    else:
        T0=None
    
    # if "cornell" in name:
    #     plot_embedding_all(g, H, ri_vert_pos, Rp, T, dim_p, p_expected, k_expected, it_spec, it_emb, MO, IT, p_leading, file_graph, file_expected, results_folder, file_com_out, time_folder, pareto_file, num_pareto, ensemble_thr, n_gen, n_pop, p_offs, bool_ov, ov_thr=0,op_com=1,dim_emb=2, char_split=' ',name='', op_var='pdimemb',  lambdav = 1e9,  alpha=0.5, beta=5,bool_link_pred=False,bool_proposed=True,gamma1=0.4,gamma2=0.3,gamma3=0.3)
        
#---------------------------- results
    if file_expected is not None and bool_link_pred==False:
        #---- Overlapping communities
        if bool_ov==True and file_expected is not None:
            #must save on lines format 
            mod0, nmi0, mod, nmi, metric_1, metric_2, f1_macro_spec, f1_micro_spec = metrics.metrics_overlapping(H, H0, ri_vert_pos, T0, ri_vert_pos,g, results_folder, nome_alg, name, file_com_out, file_expected,bool_U0=False );

        #---- Disjoint communities   
        elif (bool_ov==False) and file_expected is not None:  
            mod0, nmi0, mod, nmi, metric_1, metric_2, f1_macro_spec, f1_micro_spec = metrics.metrics_disjoint(H, H,H0, ri_vert_pos, T0, ri_vert_pos,g, results_folder, nome_alg, name, file_com_out, file_expected,char_split,bool_U0=False);

        return H, T, -1, T, -1, mod0, mod, nmi0, nmi, f1_macro_spec, f1_micro_spec, metric_1, metric_2, time_spec
    
    else:
        return H, T, -1, T, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1



#%% Attributes

def read_attributes(name, g, p_expected, dim):
    print("-- Read attributes")
    file_content = 'datasets_classification/'+name+'.content'
    file_cites = 'datasets_classification/'+name+'.cites'
    out_expected = 'datasets_classification/'+name+'_expected.txt'
    out_graph = 'datasets_classification/'+name+'.net'
    
    vert_dict= {} #dict of vertex name and index #not used
    #------------ File expected
    f = open(file_content)                    
    lines = f.readlines()
    
    lin_arr = (lines[0].split('\n')[0]).split('\t')     
    num_features = len(lin_arr[1:-1])
           
    T= np.zeros((g.vcount(),num_features))     
        
    i=0
    for lin in np.arange(len(lines)):
        lin_arr = (lines[lin].split('\n')[0]).split('\t')     
        T[lin,:] = lin_arr[1:-1]
    
    from sklearn.cluster import KMeans
    
    #------------------------------------------
    #autoencoder vai bem sozinho 
    Q, _ = autoencoder_keras(T,T, dim)
    #print(Q)
    f1_macro, f1_micro= metrics.node_classification(Q, p_expected, test_size=0.3)  

    
    return Q,T 

    

def read_attributes_T(name, g, p_expected, dim):
    print("-- Read attributes")
    
    is_attr= False
    if "amazon" not in name and "youtube" not in name:
        if "facebook" not in name:
            is_attr=True
            file_content = '../datasets_classification/'+name+'.content'
            #file_cites = '../datasets_classification/'+name+'.cites'
            #out_expected = '../datasets_classification/'+name+'_expected.txt'
            #out_graph = '../datasets_classification/'+name+'.net'
                        
            #------------ File expected
            f = open(file_content)                    
            lines = f.readlines()
            
            lin_arr = (lines[0].split('\n')[0]).split('\t')     
            num_features = len(lin_arr[1:-1])
        else:
            is_attr=True
            file_content = '../datasets_overlapping/'+name+'.feat'
            #file_cites = 'datasets_overlapping/'+name+'.cites'
            out_expected = '../datasets_overlapping/'+name+'_expected.txt'
            out_graph = '../datasets_overlapping/'+name+'.net'
            
            #------------ File expected
            f = open(file_content)                    
            lines = f.readlines()
            
            lin_arr = (lines[0].split('\n')[0]).split('\t')     
            num_features = len(lin_arr[1:])
    

        print(name)
        print(g.vcount())        
        T= np.zeros((g.vcount(),num_features)) 

        vindex = np.array(g.vs['orig'])
        
        #print("qtd linhas:", len(lines))
        i=0
        for lin in np.arange(len(lines)):
        #for lin in np.arange(0,30):
            lin_arr = (lines[lin].split('\n')[0]).split('\t') 
            #print(lin)
            #print(lin_arr)
            if "facebook" not in name:
                T[lin,:] = lin_arr[1:-1]    
            else:                        
                vg = np.where( vindex == lin )[0] ##without -1
                T[vg,:] = lin_arr[1:]             
                #print(" ")
                
                
        T2, _ = autoencoder_keras(T,T, dim)
        #print(Q)
        #f1_macro, f1_micro= node_classification(T2, p_expected, test_size=0.3)  
     
    else:
        T2=None

    return T2



#%% Autoencoder

def autoencoder_keras(x_train, x_test, dim_p):
    
    encoding_dim = dim_p 
    input_img = Input(shape=(x_train.shape[1],))
    # encoded representation of input
    encoded = Dense(encoding_dim, activation='relu')(input_img)
    # decoded representation of code 
    decoded = Dense(x_train.shape[1], activation='sigmoid')(encoded)
    # Model which take input image and shows decoded images
    autoencoder = Model(input_img, decoded)
    
    # This model shows encoded images
    encoder = Model(input_img, encoded)
    # Creating a decoder model
    encoded_input = Input(shape=(encoding_dim,))
    # last layer of the autoencoder model
    decoder_layer = autoencoder.layers[-1]
    # decoder model
    decoder = Model(encoded_input, decoder_layer(encoded_input))
    
    # from keras.models import Sequential
        
    # autoencoder = Sequential()
    # autoencoder.add(Dense(x_train.shape[0],activation='relu',input_dim=x_train.shape[0]))
    # autoencoder.add(Dense((x_train.shape[0]-encoding_dim)/2,activation='relu'))
    # autoencoder.add(Dense(encoding_dim,activation='relu'))
    # autoencoder.add(Dense((x_train.shape[0]-encoding_dim)/2,activation='relu'))
    # autoencoder.add(Dense(x_train.shape[0],activation='relu'))



    autoencoder.compile(optimizer='adam', loss=losses.MeanSquaredError()) #'binary_crossentropy')
    
   
    autoencoder.fit(x_train, x_train,
                    epochs=100,
                    batch_size=100,
                    validation_data=(x_test, x_test),
                    verbose=0)
    
    
    
    encoded_img = encoder.predict(x_test)
    decoded_img = decoder.predict(encoded_img)
            
    
    return encoded_img, decoded_img

#%%% Conv. autoencoder 

# import keras
# from keras import layers

# input_img = keras.Input(shape=(x_train.shape[0], x_train.shape[0], 1))

# x = layers.Conv2D(16, (3, 3), activation='relu', padding='same')(input_img)
# x = layers.MaxPooling2D((2, 2), padding='same')(x)
# x = layers.Conv2D(8, (3, 3), activation='relu', padding='same')(x)
# x = layers.MaxPooling2D((2, 2), padding='same')(x)
# x = layers.Conv2D(8, (3, 3), activation='relu', padding='same')(x)
# encoded = layers.MaxPooling2D((2, 2), padding='same')(x)

# # at this point the representation is (4, 4, 8) i.e. 128-dimensional

# x = layers.Conv2D(8, (3, 3), activation='relu', padding='same')(encoded)
# x = layers.UpSampling2D((2, 2))(x)
# x = layers.Conv2D(8, (3, 3), activation='relu', padding='same')(x)
# x = layers.UpSampling2D((2, 2))(x)
# x = layers.Conv2D(16, (3, 3), activation='relu')(x)
# x = layers.UpSampling2D((2, 2))(x)
# decoded = layers.Conv2D(1, (3, 3), activation='sigmoid', padding='same')(x)

# autoencoder = keras.Model(input_img, decoded)
# autoencoder.compile(optimizer='adam', loss=losses.MeanSquaredError())

# from keras.datasets import mnist
# import numpy as np

# #(x_train, _), (x_test, _) = mnist.load_data()

# #x_train = x_train.astype('float32') / 255.
# #x_test = x_test.astype('float32') / 255.
# x_train = np.reshape(x_train, ( x_train.shape[0], x_train.shape[0], 1))
# x_test = np.reshape(x_test, ( x_train.shape[0], x_train.shape[0], 1))

# #tensorboard --logdir=/tmp/autoencoder

# autoencoder.fit(x_train, x_train,
#                 epochs=50,
#                 batch_size=128,
#                 shuffle=True,
#                 validation_data=(x_test, x_test))
