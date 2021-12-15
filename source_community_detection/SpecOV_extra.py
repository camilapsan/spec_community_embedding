
def spec_ov_main_sparsify(file_graph, nex=10, plarg=0.3, gamma1=1, gamma2=1, ov_thr=0.5, bool_ov=True, file_out=None, bool_verbose=False, out_verbose=[], KNumber_pre=-1):
    g = sf.read_construct_g(file_graph)
    mod_mat = sf.construct_modularity_generalized(g,gamma1,gamma2).transpose()
     
    #TODO: construct_MConn
    
    
    #TODO sparsify mod_mat
    mod_mat[mod_mat < 0] =0

    
    start_time = time.perf_counter()         
    eig_val, eig_vec, eig_pos_ind, eig_neg_ind, comms_est, ri_vert_pos, ri_vert_neg = spec_ov_eigen_dec(g, mod_mat, plarg)  
    print("--- CALC EIG: %s seconds ---" , (time.perf_counter() - start_time))
    
    comms_est
    
    if KNumber_pre > 0:
        comms_est = KNumber_pre
    #print("eig_val")
    #print(eig_val)
    
    #print("ri_vert_pos")
    #print(ri_vert_pos)
     
    valQ_max = -100
    nmi_max = -2
    
    #TODO: test multi-processing
    threads = []  
    manager = multiprocessing.Manager()
    #res_queue = queue.Queue()
    res_queue = manager.Queue()
    
    for it in range(0,nex):
        #print("---------------" + str(it))
        #t = threading.Thread(target=spec_ov_alg,args=[it, res_queue, g, comms_est, ri_vert_pos, ri_vert_neg, eig_pos_ind, eig_neg_ind, bool_ov, ov_thr, bool_verbose, out_verbose])
        t = multiprocessing.Process(target=spec_ov_alg,args=[it, res_queue, g, comms_est, ri_vert_pos, ri_vert_neg, eig_pos_ind, eig_neg_ind, bool_ov, ov_thr, bool_verbose, out_verbose])
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
    #Sov  (vert) x (communities)
    
    #ri_vert_pos, ri_vert_neg --> plot: does not change depending on the algorithm.
    
    #TODO: find the best solution.. 
    #write in file
    #each line is a different community
    if file_out is not None:
        print_com_lines_file(file_out, Sov_best)
    
    return q_best, Sov_best, g, eig_val, ri_vert_pos, ri_vert_neg, R_group_pos, R_group_neg
   
#% SpecOV: Call main

# def call_main(file_graph, plarg, bool_ov):
#     #TODO: input params
#     print("TODO implement")

