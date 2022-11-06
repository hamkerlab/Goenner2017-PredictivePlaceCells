# Task:
# - Generate a sequence in each network
# - decode the sequence end point
# - initiate naviagtion to that point, or random nav.

from ANNarchy import *

from brian.tools.datamanager import *
#from brian.tools.taskfarm import *

from random import *
from time import time
import pickle
import matplotlib.cm as cm
import gc

from random import seed as pyseed
from numpy.random import seed as npseed
import numpy as np
from scipy.sparse import lil_matrix
from pylab import *

setup(dt = 0.2)

from ann_InitLearn import xypos, xypos_maze, xy_index_maze, plot_sequence, Exp_xyValue
from ann_neuron_model import eqs_if_scaled_new, eqs_if_scaled_new_DG, eqs_if_trace

from ann_network import  Exc_neurons, simple_syn, poisson_inp, context_pop_home, DG_neurons, Inh_neurons, S_Exc_Exc, S_Exc_Inh, S_Inh_Inh, S_Inh_Exc, S_Poi_cont_home, S_DG_Exc
#from ann_network_sigmascale import  Exc_neurons, simple_syn, poisson_inp, context_pop_home, DG_neurons, Inh_neurons, S_Exc_Exc, S_Exc_Inh, S_Inh_Inh, S_Inh_Exc, S_Poi_cont_home, S_DG_Exc

from ann_seqgen import initiate_sequence as init_seq
from ann_define_params import netw_param, maze, nav




lil_w = lil_matrix( np.diag(np.ones(6400)) )
S_context_homeOrAway_DG = Projection(
    pre = context_pop_home, post = DG_neurons, target = 'exc', synapse = simple_syn, name='FFinp'
).connect_from_sparse( weights = lil_w )

def global_run(netw_index, netw_obj, netw_ind, trial_ind):

	# Make sure the network setup is correct

    print("netw_ind, trial_ind = ", netw_ind, trial_ind)
    start_ind = seqstart[netw_ind][trial_ind][0]
    print("start_index (global run) = ", start_ind)

    if np.mod(trial_ind, 2)==0:
        for dend in netw_obj.get(S_context_homeOrAway_DG):
            dend.w = 1e12 * weight_array_home[netw_ind][trial_ind][dend.post_rank] # orig
    else:
        for dend in netw_obj.get(S_context_homeOrAway_DG):
            dend.w = 1e12 * weight_array_away[netw_ind][trial_ind][dend.post_rank]


    t_start = netw_obj.get_time()

	# Generate sequence
    exc_spikes = init_seq(netw_index, netw_obj, M_Exc, M_Inh, M_DG, M_contpop, M_poi, 400.0, start_ind) # second last param. is total run time of the sequence, last param = start index!

	# Decode sequence endpoint
    te, ne = M_Exc.raster_plot(exc_spikes)

	# Start behavioral simulations:
	#	Loop over goal navigation, place cell firing, reward delivery, ...
	#	Loop in steps of 100 msec

	# Save results - after each trial or when all trials have finished?


    result_dict = {}
    result_dict['i_netw'] = netw_ind
    result_dict['i_trial'] = trial_ind



    #print "Finished writing to dict"

    return result_dict



if __name__ == '__main__':


    t_0 = time()

    N = 144
    nTrials = 40
    n_grid = 80 
    identifier = 'ann_Square_maze_ngr80_N'+str(N)+'_DAdecr1st_'+str(nTrials) # on tractus and laptop

    dataman = DataManager(identifier) # I should decide beween using Brian's DataManager with run_tasks() and ANNarchy's parallel_run() method!!!



    weight_array_home = reshape(weight_array_home, (dataman.itemcount(), nTrials, n_grid**2))


    # Define DG input weights
    placevalMat = np.zeros(netw_param['n_exc'])
    placevalMat[list(range(netw_param['n_exc']))] = np.random.rand(netw_param['n_exc']) * 0.16 # * 0.15 # defined in nA # no DG spikes fo 0.15; 0.18 is sufficient to generate DG spikes, 0.3 increases the size of the attractor bump!

    # Start recording
    M_Exc = Monitor(Exc_neurons, ['spike', 'u_mV', 'g_exc', 'g_inh', 'ext_input']) 
    M_Inh = Monitor(Inh_neurons, ['spike', 'u_mV', 'g_exc', 'g_inh'])
    M_DG = Monitor(DG_neurons, ['spike']) 
    M_contpop = Monitor(context_pop_home[0:50], ['spike'])
    M_poi = Monitor(poisson_inp, ['spike'])


    # creates report.tex
    #report()

    netw_obj = Network(everything=True)

    #print "before compile()"
    
    # code generation, build up network
    compile()

    sigma_CA3_scaling = 1.0 # 0.25 # 0.5 #    

    if sigma_CA3_scaling != 1.0:
        print("Computing CA3 rec. weight sum...")
        wsum_CA3rec = 0.0
        for dend in S_Exc_Exc.dendrites:
            wsum_CA3rec += np.sum(dend.w)
        print("S_Exc_Exc.w.sum = ", wsum_CA3rec)
        wsum_CA3default = 20886436.0 # (default case)
        wsum_CA3rec_scaled = 0.0
        for dend in S_Exc_Exc.dendrites:
            dend.w = wsum_CA3default / wsum_CA3rec * np.ones(dend.size) * dend.w # This works!
            wsum_CA3rec_scaled += np.sum(dend.w)
        print("sum(S_Exc_Exc.w), scaled = ", wsum_CA3rec_scaled)


    # Loop over trials and networks:

    n_nets = 10
    first_netw = 0
  
    for i_netw in range(first_netw, n_nets):
         
        # Run several trials in parallel:
        n_parallel_trials = 10 # 
        n_total_trials = 3 # 
        first_trial = 0 # 
        
        for i_trial in range(first_trial, n_total_trials, n_parallel_trials): 
            print("i_netw = ", i_netw)
            print("i_trial (start) = ", i_trial)
            results = parallel_run(method = global_run, number = n_parallel_trials, netw_ind = np.repeat(i_netw, n_parallel_trials), trial_ind = list(range(i_trial, i_trial + n_parallel_trials)))

            file_save = open('pftask_dataloop.txt', 'a')

            for i_par_netw in range(n_parallel_trials):
                pickle.dump(results[i_par_netw], file_save, 0)
            file_save.close()



















