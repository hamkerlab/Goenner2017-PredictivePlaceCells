from ANNarchy import *

import numpy as np
#from ann_network import *
#from ann_functions import run_netw

setup(dt = 0.2)

from ann_define_params import netw_param, maze
from ann_InitLearn import xypos, xypos_maze, xy_index_maze, plot_sequence, Exp_xyValue, Exp_xyValue_normalized
#from ann_neuron_model import eqs_if_scaled_new, eqs_if_scaled_new_DG, eqs_if_trace
#from ann_network import poisson_inp, context_pop_home, DG_neurons, Exc_neurons, Inh_neurons, simple_syn, S_Poi_cont_home, S_DG_Exc, S_Exc_Exc, S_Exc_Inh, S_Inh_Inh, S_Inh_Exc
from ann_network_sigmascale import poisson_inp, context_pop_home, DG_neurons, Exc_neurons, Inh_neurons, simple_syn, S_Poi_cont_home, S_DG_Exc, S_Exc_Exc, S_Exc_Inh, S_Inh_Inh, S_Inh_Exc

from ann_plotting import raster_plot_spikes, bump_trajectory, plot_spectrogram, plot_mempot, plot_syncurrents
from time import time
from pylab import *

def initiate_sequence(netw_index, netw_obj, M_Exc, M_Inh, M_DG, M_contpop, M_poi, t_end=400.0, start_index=0, t_start=0.0):

    #print "netw_index = ", netw_index

    stim_time_ms = 35 
    init_time_ms = 50 
    run_time_ms = t_end - init_time_ms # 350 

    t_init = time()

    #start_index = xy_index_maze(maze['L_cm'] - maze['edge_cm'] - 50, maze['L_cm'] - maze['edge_cm'] - 50, netw_param['n_grid'], maze['L_cm'])

    print("Initiating the network...") 
    start_bias = np.zeros(netw_param['n_exc'])
    start_bias[list(range(netw_param['n_exc']))] = Exp_xyValue_normalized(list(range(netw_param['n_exc'])), netw_param['n_grid'], 0, start_index)

    netw_obj.get(Exc_neurons).I_ext_pA = start_bias* 0.9e3 

    remaining_stim_time_ms = max(0.0, min(t_end - t_start, stim_time_ms - t_start))
    remaining_init_time_ms = max(0.0, min(t_end - (t_start + remaining_stim_time_ms), init_time_ms - (t_start + remaining_stim_time_ms)))
    remaining_seqgen_time_ms = max(0.0, t_end - (t_start  + remaining_stim_time_ms + remaining_init_time_ms))

    simulate(remaining_stim_time_ms, measure_time=True)

    # end of initiation period
    netw_obj.get(Exc_neurons).I_ext_pA = 0

    simulate(remaining_init_time_ms, measure_time=True) #

    print("Switching to place-value noise...")

    netw_obj.get(poisson_inp).rates = 200*np.ones(netw_param['n_exc'])

    t_start = time()
    t_sim = run_time_ms
    t_curr = 0
    '''#
    for step in xrange(10):
        #netw_obj.simulate(t_sim * 0.1) # , measure_time=True)
        simulate(t_sim * 0.1) # , measure_time=True)
        t_estim = int( (9-step) * (time()-t_start) / (step+1) )
        print "%i %% done, approx. %i seconds remaining" %( int((step+1)*10),  t_estim)
    '''
    #simulate(t_sim, measure_time=True)
    simulate(remaining_seqgen_time_ms, measure_time=True)

    print("Duration: %i seconds" %( int(time() - t_start) ))

    dpisize= 300
    textsize = 9
    labelsize = 6
    #t_end = init_time_ms + run_time_ms
    dt = core.Global.config['dt']

    exc_spikes = netw_obj.get(M_Exc).get(['spike'])
    inh_spikes = netw_obj.get(M_Inh).get(['spike'])

    exc_rates_smoothed = netw_obj.get(M_Exc).smoothed_rate(exc_spikes, smooth=50.0)

    print("netw_index, i_maxrate = ", netw_index, np.argmax(exc_rates_smoothed[:, -1])) # "Error: Axis 1 has size 0" ?

    ion()
  
    #raster_plot_spikes(netw_obj.get(M_poi), netw_obj.get(M_contpop), netw_obj.get(M_DG), netw_obj.get(M_Exc), exc_spikes, netw_obj.get(M_Inh), inh_spikes, t_end, dpisize, textsize, labelsize, dt)  

    #print "Total time = ", time()-t_init

    ioff()
    show()

    return exc_spikes






