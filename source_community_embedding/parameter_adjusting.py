import fnmatch
import os

import pandas as pd
import numpy as np

import matplotlib
import matplotlib.pyplot as plt

import matplotlib.transforms as transform

alg_name = 'AutoEncNorm2'
results_folder="../results_Tese_2021_3/"

#results_pattern = "out_datasets_DISJ_emb-AutoEncNorm_dim-128_com-SpecDecOV_gen-10_p-*_lb-1000000000.0_a-0.5_b-5_pdimemb_" 
results_pattern = "out_datasets_DISJ_emb-"+ alg_name + "_dim-128_com-SpecDecOV_gen-10_p-*_lb-1000000000.0_a-0.5_b-5_pdimemb_" 
#"*.csv"
# _gamma_0.4_0.3_0.3.csv'

#attributes, SpecG, auto
                

gamma_list1=np.array([0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1])
gamma_list2=np.array([0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1])


df_list = {}
lin=0



name_list = ['cornell','texas', 'washington', 'wisconsin','cora', 'citeseer']
dataset='cornell'
 
for dataset in name_list:   
    df_results = pd.DataFrame(columns=['name', 'gamma','gamma1','gamma2', 'mod', 'nmi', 'f1_macro', 'f1_micro','time'])  
    lin=0
    for gamma1 in gamma_list1:
        for gamma2 in gamma_list2[gamma_list2 <= (1-gamma1)]:
            gamma3 = np.round( 1 - gamma1 - gamma2,1)
            #print("Gamma1=",gamma1,", gamma2=", gamma2, ", gamma3=",gamma3)     
            
            results_pattern_gamma = results_pattern + "gamma_" + str(gamma1) + "_" + str(gamma2)+ "_*.csv"
            #print(results_pattern_gamma)
            
            for file in os.listdir(results_folder):
                if fnmatch.fnmatch(file, results_pattern_gamma):
                    print("file: ", file)
                    df_file = pd.read_csv(results_folder+file,sep='\t')
                    #print(df_file)
                    for i in np.arange(df_file.shape[0]):
                        df = df_file.loc[i]
                        if dataset in df['name']:
                           # print(df)                
                            df_results.loc[lin] = [df['name'], "("+str(gamma1)+","+str(gamma2)+","+str(gamma3)+")",gamma1, gamma2, df['mod_NMF'], df['nmi_NMF'], df['f1_macro_NMF']/100, df['f1_micro_NMF']/100, df['time_tot']]
                            lin=lin+1
                            
                                
                            
    #%% 
    figure, ax = plt.subplots(figsize=(30, 15))
    #matplotlib.rc('text', usetex = True)
    
    #fig = plt.figure()
    #ax = plt.axes()
    colors=['blue', 'orange', 'green','violet','orange','yellow','gray','magenta','cyan','black']          
    
    curve_list = ['mod','nmi','f1_macro','f1_micro'] #,'time']  
    tr = transform.Affine2D().translate(0, 0) + ax.transData
    
    for i,curve in enumerate(curve_list):
        if i==3:
             tr = transform.Affine2D().translate(0, -0.003) + ax.transData        
             
        ax.plot(np.arange( df_results.shape[0] ), df_results[curve],color=colors[i], transform=tr) #, label="Vertex vectors")       #s=10 ,
    #ax.set_xlabel('x-axis', fontsize = 20)
    
    #for n1 in n1_array:
    #ax.text(n1, errorQ_perc_array_avg[n1-1,math.floor(p*10)-1], n1 ,fontsize=10)
    #plt.xticks(fontsize = 15)
    x_index= df_results[ (df_results['gamma1'] == 0.1) & (df_results['gamma2']==0.7) ].index[0]
    
    
    plt.xticks(np.arange( df_results.shape[0] ), df_results['gamma'], rotation='vertical',fontsize = 25)
    plt.yticks(fontsize = 15)
    ax.axvline(x=x_index, color='red', linestyle='--')
    
    ax.set_xlabel("($\gamma_1$,$\gamma_2$,$\gamma_3$)",fontsize = 30)
    ax.set_ylabel("Metric value",fontsize = 30)
    
    ax.grid()
    
    #plt.plot(n1_array,errorQ_perc_array_avg[:,0], color='blue') #, label="Vertex vectors")       #s=10 ,
    #leg_p = ['$\mathregular{\mathit{p}}$=' + str(round(p_eig,1)) + 'n' for p_eig in p_eig_array] 
    
    # ax.set_xlabel("Gamma",fontsize = 15)
    # ax.set_ylabel("Value",fontsize = 15)
    
    # labels = [item.get_text() for item in ax.get_xticklabels()]
    # labels = df_results['gamma']
    
    # ax.set_xticklabels(labels)
    
    #ax.legend(df_results['gamma'], prop={'size':12})
    ax.legend(['Modularity','NMI','F1-macro','F1-micro'], prop={'size':30})
    plt.savefig(results_folder+"param_adjusting_"+alg_name+"_"+dataset+".png",bbox_inches='tight')
    plt.show()
