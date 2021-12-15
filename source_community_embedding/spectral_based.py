import MOSpecG_partitioning_OV as SpecG
import SpecOV_partitioning_v2 as SpecOV
import spectral_functions as sf


#%%% Embedding Spectral 

def spectral_emb_main(MO, IT, p_leading, file_graph, file_expected, results_folder,file_com_out, time_folder, pareto_file, num_pareto, ensemble_thr, n_gen, n_pop, p_offs, bool_ov, ov_thr=0, op_com=1, dim_emb=2, file_out_plot=None , list_sol=None, char_split=' ',name=''):    
        
    #if "edges" in file_graph:
        #file_graph = sf.create_edgelist_i0(file_graph)
        #print("edgelist:",file_graph)
        
    graph = SpecG.Graph()
    graph.g = sf.read_construct_g(file_graph)
         
    gamma1=1  
    gamma2 =1
    p_best=None
    
    if op_com==1:
        #specOV
        nome_alg="SpecDecOV"
        #TODO: largest real value as param
        q_best, Sov, g, eig_val, ri_vert_pos, ri_vert_neg, Rp, Rn = SpecOV.spec_ov_main(file_graph, IT, p_leading, gamma1, gamma2, ov_thr, bool_ov, file_com_out, bool_verbose, out_verbose)           
        # print("ri_vert_pos")
        # print(ri_vert_pos)
        
        #dim embedding = 2
        idx= np.where( np.all(Sov[..., :] == 0, axis=0) == False )[0]
        Sov_nonzero = Sov[:,idx]
        Sov_nonzero.shape
  
        if list_sol is not None:   
            for i in list_sol:            #     #TODO: GET MAX                 #print(Sov)
                 if Sov is not None:                            
                     com_i =np.where(Sov[i,:]>0)[0][0]
                     print("vert ", i, ", com=", com_i)
        Sov_best=Sov
        #import SpecOV_partitioning_v2 as SpecOV         #list_sol=None         #g = sf.read_construct_g(file_graph)         #g.plot()            
        #PLOT 
        SpecOV.plot_vertex_vectors(g, ri_vert_pos, ri_vert_neg, Rp, Rn, Sov_nonzero, op="emb",list_sol=list_sol,file_out=file_out_plot ) #, labels=g.vs['id'])
        
    elif op_com==2:
        nome_alg="SpecG_OV_mod"
        print("\n---------SpecG - 2")
        gamma1=1
        gamma2=1
        #gamma1=1
        #gamma2=1
        print("gamma",gamma1,",",gamma2)
        p_best, graph, params, eig, pop, offs, Sov_best, Rp, Rn,list_gen_Q=  SpecG.SpecG_single_main(False, IT, p_leading, file_graph, results_folder, time_folder, 
                                                                             pareto_file, num_pareto, ensemble_thr, n_gen, n_pop, p_offs, bool_ov,
                                                                             mo_weight_Qin=gamma1 , mo_weight_Qnull=gamma2,mod_orig=True,T=T)
  
        p_best=p_best[0]
        Sov_best=Sov_best[0]
        ri_vert_pos = graph.ri_vert_pos
        

    elif op_com ==3:
        #spectral_emb_main(file_graph, p_leading)
        #ensemble_MOSpecG_main
        nome_alg="SpecG_OV_gen"
        print("\n---------SpecG")
        gamma1=0.8
        gamma2=0.2

        p_best, graph, params, eig, pop, offs, Sov_best, Rp,Rn, list_gen_Q=  SpecG.SpecG_single_main(False, IT, p_leading, file_graph, results_folder, time_folder, pareto_file, 
                                                                                 num_pareto, ensemble_thr, n_gen, n_pop, p_offs, bool_ov,
                                                                                 mo_weight_Qin=gamma1 , mo_weight_Qnull=gamma2,mod_orig=True)


        p_best=p_best[0]
        Sov_best=Sov_best[0]
        ri_vert_pos = graph.ri_vert_pos   
    elif op_com ==4:
        #spectral_emb_main(file_graph, p_leading)
        #ensemble_MOSpecG_main
        nome_alg="SpecG_OV_4_enc_mod"
        print("\n---------SpecG - 3")
        gamma1=1
        gamma2=1

        p_best, graph, params, eig, pop, offs, Sov_best, Rp, Rn,list_gen_Q=  SpecG.SpecG_single_main(False, IT, p_leading, file_graph, results_folder, time_folder, pareto_file, num_pareto, ensemble_thr, n_gen, n_pop, p_offs, bool_ov,mo_weight_Qin=gamma1 , mo_weight_Qnull=gamma2,mod_orig=False)


        p_best=p_best[0]
        Sov_best=Sov_best[0]
        ri_vert_pos = graph.ri_vert_pos
    elif op_com ==5:
        #spectral_emb_main(file_graph, p_leading)
        #ensemble_MOSpecG_main
        nome_alg="SpecG_OV_5_enc_gen"
        print("\n---------SpecG")
        gamma1=0.8
        gamma2=0.2
 
        
        p_best, graph, params, eig, pop, offs, Sov_best, Rp,Rn, list_gen_Q=  SpecG.SpecG_single_main(False, IT, p_leading, file_graph, results_folder, time_folder, pareto_file, num_pareto, ensemble_thr, n_gen, n_pop, p_offs, bool_ov,mo_weight_Qin=gamma1 , mo_weight_Qnull=gamma2,mod_orig=False)


        p_best=p_best[0]
        Sov_best=Sov_best[0]
        ri_vert_pos = graph.ri_vert_pos   
    #--- node classification + community detection     
    
    #---- Overlapping communities
    if bool_ov==True and file_expected is not None:
        #must save on lines format 
        #must calc oNMI (external code)]    
        
        #print(graph.g.vs['Index'])
        
        print("\n---------\nCommunity detection:")
        base_folder_out = results_folder+""+nome_alg + "" + "/"+ name +"/"
        file_com_out = base_folder_out+"lines_net_" + name + ".com"
                
        SpecOV.print_com_lines_file(file_com_out, Sov_best) #231220
        
        shutil.copyfile(file_com_out, "f1.txt")
        shutil.copyfile(file_expected, "f2.txt")

        os.system('./mutual3/mutual "f1.txt" "f2.txt"')  
    
        print(graph.g.vcount())
        #com_expected,lin_expected  = sf.read_partition_expected_OV(file_expected,  graph.g.vcount(), char_split="\t")
        com_expected,lin_expected  = sf.read_partition_expected_OV(file_expected, graph.g, char_split="\t")
        #print(com_expected)
        jacc_val = score_overlapping( Sov_best,com_expected, lin_expected, type_sim="jaccard")        
        f1_val = score_overlapping( Sov_best,com_expected, lin_expected, type_sim="f1score")
        print("jacc_score",jacc_val)
        print("f1_score",f1_val)
          
        nmi=0
        mod=0
        
        metric_1 = f1_val
        metric_2 = jacc_val
        #print("\n---------\nClassification using ri_vert_pos")        
        #print('\n---------\n')
    
    #---- Disjoint communities   
    elif bool_ov==False and file_expected is not None:        
        p_expected = sf.read_partition_expected(file_expected,char_split)
        
        if p_best is None:
            p_best = create_part_from_Sov(Sov).astype(int)
                
        print("\n---------\nCommunity detection:")
        mod = graph.g.modularity(p_best, graph.g.es['weight'])
        print("mod=",mod)
    
        nmi = sf.calc_nmi(p_best, p_expected)
        print("nmi=",nmi)
        
        ri_3d = np.zeros(( 1, graph.g.vcount(), graph.g.vcount() ))
        ri_3d[0] = ri_3d
        ri_enc, ri_dec = sf.autoencoder_mod(ri_3d, ri_3d)  
        ri_dec = ri_dec[0]
      
        #k_expected = len(np.unique(p_expected))
        print("\n---------\nClassification using ri_vert_pos")
        f1_macro, f1_micro= node_classification(ri_vert_pos, p_expected, test_size=0.3)
        print('\n---------\n')
        
        #k_expected = len(np.unique(p_expected))
        print("\n---------\nClassification using ri_dec")
        #na washington piorou...
        f1_macro, f1_micro= node_classification(ri_dec, p_expected, test_size=0.3)
        print('\n---------\n')

        metric_1=f1_macro
        metric_2=f1_micro


    return mod, nmi, metric_1, metric_2
        #print_com_part(file_com_out, p_best[0])    
    #todo: node classification



def spectral_emb_main(file_graph, p_leading):
    
    g = sf.read_construct_g(file_graph)
    gamma1=0.8
    gamma2=0.2

    mod_mat = sf.construct_modularity_cosine(g, gamma1, gamma2).transpose()

    
    eig = Eig()
    eig.val, eig.vec, eig.pos_ind, eig.neg_ind = sf.calc_eigen(graph.g, mod_mat, p_leading, which="LM")
    
    #print(eig_val)
    graph.KNumber = sf.est_number_communities(eig.val)
    print("Estimation k: " + str( graph.KNumber))
    
    print("** Calculate vertex vectors ri... **")
    
    ri_vert_pos = sf.calc_ri_vert_pos(g, eig.val, eig.vec, eig.pos_ind)
    ri_vert_neg = sf.calc_ri_vert_neg(g, eig.val, eig.vec, eig.neg_ind)
    
    p_expected = sf.read_partition_expected(file_expected,char_split)
    f1_macro, f1_micro= node_classification(ri_vert_pos, p_expected, test_size=0.3)
        
    return graph.ri_vert_pos  
    
    