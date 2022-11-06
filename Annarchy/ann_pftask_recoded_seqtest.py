#from brian import *
from ANNarchy import *

#from brian.tools.datamanager import *
#from brian.tools.taskfarm import *
from random import *
from time import time as ttime
import pickle
import matplotlib.cm as cm
import gc
from random import seed as pyseed
from numpy.random import seed as npseed

from pylab import *
import numpy as np
#import plotting_func as pf

# This file is for testing sequence generation and decoding based on a fixed weight matrix provided as input
# TODO: Fix these ISSUES:
# - Change of Poisson rates from 10 to 200 Hz has no effect (see raster plots)
# - Even with 200 Hz Poisson input (set in the population definition before compiling),
#   there is NO effect of context pop. firing - no DG spikes?!


setup(dt = 0.2)

import ann_InitLearn as net_init # xypos_maze, xy_index_maze, initConstants, quad_dist_grid, Exp_xyValue2
from ann_recoded_functions import rbc_wrapper, cpf_wrapper, nav_to_goal_wrapper, goal_check_wrapper, reward_delivery_wrapper, set_start_location, calc_focal_search_ann, do_focal_search, prepare_sequential_search, set_subgoal, subgoal_coords, subgoal_reached, update_startloc, set_search_goal, continue_search, prepare_sequence, prepare_navigation, reward_placement

from ann_init_vars import agent, task, mvc, maze, sim, data, netw

from ann_network_fullsim import poisson_inp, context_pop_home, context_pop_away, DG_neurons, S_Poi_cont_home, S_Poi_cont_away, S_context_home_DG, S_context_away_DG, Exc_neurons, Inh_neurons, S_Exc_Exc, S_Exc_Inh, S_Inh_Exc, S_Inh_Inh, simple_syn, S_DG_Exc


mon_te, mon_ne = array([]), array([])
mon_xest, mon_yest= array([]), array([])


#--------------------------------------------------------------------------------------------------------
#'''#
def running_bump_center(n):
    # Decodes current bump position from CA3 exc. spiking activity (=spike monitors)
    # Writes decoded position into data['center_mat']
    # Approach:
    # Call every step (0.2ms)
    # Decode from the last 4ms -> combine estimates from 20 consecutive calls

    # Check: Function plot_bump_shift in the modelDB / an_Helge directory

    global data

    global mon_te
    global mon_ne
    global mon_xest
    global mon_yest

    exc_spikes = net_full.get(m_exc).get(['spike'])
    te, ne = net_full.get(m_exc).raster_plot(exc_spikes)
    nz_spiking = nonzero(te)
    ni = ne[nz_spiking]

    if sim['ongoingSeq']==True:
        data = rbc_wrapper(ni, data, netw['n_grid'], maze['L_cm'], sim['trial'], net_full.get_time())

    # TEST
    mon_te = append(mon_te, te)
    mon_ne = append(mon_ne, ne)
    if sum(data['n_spikes_ringbuffer']) > 0:
        xest = sum(data['x_center_sum_ringbuffer']) / sum(data['n_spikes_ringbuffer'])
        yest = sum(data['y_center_sum_ringbuffer']) / sum(data['n_spikes_ringbuffer'])
        mon_xest = append(mon_xest, xest )
        mon_yest = append(mon_yest, yest )

#'''
#--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------    
def check_placefields(n):
    global netw
    if sim['ongoingSeq']==False:
        Exc_neurons_addcurrent, DG_neurons_addcurrent = cpf_wrapper(netw['n_place_cells'], netw['pf_center_x_cm'], netw['pf_center_y_cm'], netw['PlaceFieldRadius_cm'], agent)    
        net_full.get(Exc_neurons).g_excCA3 += np.reshape(Exc_neurons_addcurrent, (netw['n_grid'], netw['n_grid']))
        net_full.get(DG_neurons).g_exc += np.reshape(DG_neurons_addcurrent, (netw['n_grid'], netw['n_grid']))  

#--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
def navigateToGoal(n):
    global agent
    #global sim
    global data
    if sim['ongoingSeq']==False:
        #agent['x_cm'], agent['y_cm'], data['occupancyMap'], agent['direction'] = nav_to_goal_wrapper(mvc, agent, sim, maze, data, get_time(), mvc['DeltaT_step_sec']) 
        agent['x_cm'], agent['y_cm'], data['occupancyMap'], agent['direction'] = nav_to_goal_wrapper(mvc, agent, sim, maze, data, net_full.get_time(), mvc['DeltaT_step_sec']) 

#--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------    
def goalCheck(n):
    global agent
    #global sim
    if sim['ongoingSeq']==False:
        agent['hasGoal'], agent['focal_search'], sim['stopping'] = goal_check_wrapper(agent, sim, net_full.get_time())

#--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------    
def reward_delivery(n):
    global agent
    global task
    global data
    if sim['ongoingSeq']==False:
        data['reward_counter'], sim['reward_found'], agent['hasGoal'], sim['stopping'], DG_cells_dopamine = reward_delivery_wrapper(agent, task, data, sim, net_full.get_time()) 
        #net_full.get(DG_neurons).DA = DG_cells_dopamine
        net_full.get(S_context_home_DG).DA = DG_cells_dopamine
        net_full.get(S_context_away_DG).DA = DG_cells_dopamine

#-------------------------------------------------------------------------------------------------------------    
#'''#

def test_function(n):
    global agent
    #print("Time [sec], agent['hasGoal'], agent['focal_search'], agent['random_search'], sim['stopping'] = ", net_full.get_time()*1e-3, agent['hasGoal'], agent['focal_search'], agent['random_search'], sim['stopping'])
    print(("Time [sec] = ", net_full.get_time()*1e-3))


#'''
#-------------------------------------------------------------------------------------------------------------    

def resume_navigation(sim, context_pop_home):
    sim['stopping'] = False
    return sim

#-------------------------------------------------------------------------------------------------------------    
def simulate_wrapper(t_max_sim_ms, sim_step_ms):
    t_init_ms = net_full.get_time()

    while(net_full.get_time() - t_init_ms < t_max_sim_ms - 0.5*sim_step_ms and sim['stopping']==False):
        # functions to be called between simulation blocks:
        running_bump_center(0) # TEST - clear memory
        check_placefields(0) # disabled for NO SIMULATION
        navigateToGoal(0)
        goalCheck(0)
        reward_delivery(0)
        test_function(0)

        net_full.simulate(min(sim_step_ms, t_max_sim_ms - (net_full.get_time() - t_init_ms)))

#-------------------------------------------------------------------------------------------------------------

def simulate_sequence(t_max_sim_ms, sim_step_ms):
    t_init_ms = net_full.get_time()
    while(net_full.get_time() - t_init_ms < t_max_sim_ms - 0.5*sim_step_ms):
        running_bump_center(0) # Deactivate to obtain a raster plot
        net_full.simulate(min(sim_step_ms, t_max_sim_ms - (net_full.get_time() - t_init_ms)))



#-------------------------------------------------------------------------------------------------------------    
    

#i_sigma_dist_cm, i_wmax_Place_cells_recur, i_wmax_Place_cells_inh,\
#i_wmax_inh_inh, i_wmax_inh_Place_cells, i_pconn_recur, i_tau_membr, i_tau_exc, i_tau_inh, i_dt = range(10)

#-------------------------------------------------------------------------------------------------------------

#netw_params = 	[sigma_dist_cm, wmax_exc_exc, wmax_exc_inh, wmax_inh_inh, wmax_inh_exc, pconn_recur, tau_membr, tau_exc, tau_inh,   dt    ]
#netw_params = 	[50.0, 	        2*0.4114*nA, 12*0.8125*pA, 6*6*13.0*pA, 0.9*5*4.1*pA,  1, 10, 6,   2,  0.2] # 
#print("netw_params= ", netw_params)

# Constants - neuron model and network

useMonitors =  True # False #
if useMonitors:
    print("Caution, using place cell monitors!")


global agent
global task
global sim
global data

data['occupancyMap'] = Inf * ones([40, int(maze['L_cm'] / maze['spatialBinSize_cm']), int(maze['L_cm'] / maze['spatialBinSize_cm'])]) 

global placevalMat
placevalMat = zeros(netw['n_place_cells'])


data['start_index'] = netw['n_grid'] * (netw['n_grid'] - 1) # 45*44 # 45**2 - 1 #0 #44


m_poi = Monitor(poisson_inp[0:50], ['spike'])
m_contpop = Monitor(context_pop_home[0:50], ['spike'])
m_dg = Monitor(DG_neurons, ['spike'])
#m_dg = Monitor(DG_neurons[0], ['DA'])
m_exc = Monitor(Exc_neurons, ['spike']) 
#m_inh = Monitor(Inh_neurons, ['spike'])


#'''#
if useMonitors: 
    net_full = Network(everything=True)

else:
    net_full = Network(everything=True) # remove monitor m_Exc?!

#'''


#compile()
net_full.compile()

#report(filename = "./ann_recoded_model_report.tex")

#dend0 = net_full.get(S_context_home_DG).dendrite(3240) # 0
#m_dend = Monitor(S_context_home_DG, ['spiketrace_pre', 'spiketrace_post','DA', 'w'], period=50.0)
#net_full.add(m_dend)
#net_full.compile()

def stim(saving, nTrials, i_seed): # main function for recall / sequence generation  
    global agent
    global task
    global sim
    global data

    global placevalMat
    global home_index_36

    sim['netw_index'] = i_seed
    
    _latency_array_sec = zeros(nTrials)
    _seq_endpoint_final_array = zeros(nTrials)
    _seq_count_array = zeros(nTrials)
    _seq_start_array = zeros([nTrials, 30])
    _seq_endpoint_array = zeros([nTrials, 30])
    _random_nav_time_array_sec = zeros(nTrials)
    _goal_nav_time_array_sec = zeros(nTrials)
    _focal_search_time_array_sec = zeros(nTrials)
    _weight_array_home = zeros([nTrials, netw['n_place_cells']])
    _weight_array_away = zeros([nTrials, netw['n_place_cells']])
    _center_mat_array = -Inf*ones([nTrials, 100, 100])
    _goal_index_array = zeros(nTrials)

    start_time = ttime()

    pyseed(i_seed + int(0.001 * ttime() + 324823)*i_seed)
    npseed(i_seed + int(0.001 * ttime() + 324823)*i_seed)  

    i_home, j_home = net_init.xypos(mod(i_seed, 36), 6)
    task['home_index'] = net_init.xy_index_maze(maze['edge_cm'] + 0.5*maze['grid_factor'] + i_home * maze['grid_factor'], maze['edge_cm'] + 0.5*maze['grid_factor'] + j_home * maze['grid_factor'], netw['n_grid'], maze['L_cm'])
    home_index_36 = i_seed # = 6*i_home + j_home
    sim['home_trial'] = True
    task['x_reward_cm'], task['y_reward_cm'] = net_init.xypos_maze(task['home_index'], netw['n_grid'], maze['L_cm'])
    print(("Network %i: Home reward location (x,y) = %f, %f" %(i_seed, task['x_reward_cm'], task['y_reward_cm'])))
    task['goal_index'] = task['home_index']
    agent['random_search'] = False


    for _iTrial in range(nTrials):
        print(("Network %i: Trial %i of %i" %(i_seed, _iTrial+1, nTrials)))
        iStim = _iTrial
        sim['trial'] = _iTrial
        _start_loc = rand()
        _goal_index_array[_iTrial] = task['goal_index']    
        agent['visited_index_36'] = zeros(36) # reset for directed random search
        agent['visited_index_6x6'] = zeros([6,6])

        if _iTrial == 0: 
                agent['x_cm'], agent['y_cm'] = set_start_location(_start_loc, maze['edge_cm'], maze['L_cm'], agent['x_cm'], agent['y_cm'], task['x_reward_cm'], task['y_reward_cm'])
                print(("Network %i: Start position (x,y) = %i, %i" %(i_seed, agent['x_cm'], agent['y_cm'])))
                data['start_index'] = net_init.xy_index_maze(agent['x_cm'], agent['y_cm'], netw['n_grid'], maze['L_cm']) 
        if agent['x_cm'] < maze['edge_cm'] or agent['y_cm'] < maze['edge_cm'] or agent['x_cm'] > maze['L_cm'] - maze['edge_cm'] - 1 or agent['y_cm'] > maze['L_cm'] - maze['edge_cm'] - 1:
            print("Starting outside of maze area!")

        sim['reward_found'] = False
        agent['hasGoal'] = False 
        agent['focal_search'] = False
        sim['ongoingSeq'] = False
        _start_time_msec = net_full.get_time() # get_time()

        #'''#
        # TEST: Provide "learned" weight matrix
        filename = 'learned_weights_DGinp'
        file_wdata_VC_PC = open('data/'+filename, 'r') 
        Wdata_VC_PC=pickle.load(file_wdata_VC_PC) 
        file_wdata_VC_PC.close()
        print(("file = ", filename))
        placevalMat = Wdata_VC_PC * 1e12 #*1e9 #* 0.5 # convert to nA - wdata from file are in ampere, max. ca. 1e-9 - caution, tau_exc is now 4x larger than previously!!!
        placevalMat = (Wdata_VC_PC * 1e12)**0.635 # * 1e9)**0.635
        #'''
        print(("wmax (placevalMat) = ", np.max(placevalMat)))



        if sim['home_trial']:
            for dend in net_full.get(S_context_home_DG).dendrites:
                #placevalMat[dend.rank] = dend.w
                dend.w = placevalMat[dend.rank]
        else:
            for dend in net_full.get(S_context_away_DG).dendrites:
                #placevalMat[dend.rank] = dend.w
                dend.w = placevalMat[dend.rank]

        #print('np.reshape( net_full.get(S_context_home_DG).w, len(net_full.get(S_context_home_DG).w) = ', np.reshape( net_full.get(S_context_home_DG).w, len(net_full.get(S_context_home_DG).w)) ) 


        if sim['home_trial']:
            net_full.get(S_Poi_cont_home).w = 21.0 * 1e12
            net_full.get(S_Poi_cont_away).w = 0.0
        else:
            net_full.get(S_Poi_cont_home).w = 0.0
            net_full.get(S_Poi_cont_away).w = 21.0 * 1e12

        counter_0 = 0

        while sim['reward_found'] == False and net_full.get_time()  < (mvc['time_out_sec']*1e3 - 1) : # Global timeout based on all trials 
            counter_0 += 1
            if counter_0 > 1:
                print(("Network %i: counter_0 = %i" %(i_seed, counter_0)))
                
            sim['value_gain'] = 0

            # Sequence generation for goal-setting:
            sim, data, curr_pos_index, _seq_start_array = prepare_sequence(_iTrial, agent, netw, maze, sim, data, _seq_count_array, _seq_start_array)
            #initiate_sequence(curr_pos_index, False, i_seed) #  # True # Disabled: HACK for NO SIMULATION
            initiate_sequence(curr_pos_index, True, i_seed) #

            sim['reward_found'] = True # HACK to terminate loop
            _latency_array_sec[_iTrial] = 1e-3 * (net_full.get_time() - _start_time_msec) # HACK to trick the data manager

            _seq_count_array[_iTrial] += 1
            sim['ongoingSeq'] = False	    
            # Sequence has finished - determine the navigation goal as the end point of the sequence:
            agent['xGoal_cm'], agent['yGoal_cm'] = net_init.xypos_maze(data['center_mat'].argmax(), len(data['center_mat'][0,:]), maze['L_cm'])

        for dend in net_full.get(S_context_home_DG).dendrites:
            _weight_array_home[_iTrial, dend.rank] = dend.w
        for dend in net_full.get(S_context_away_DG).dendrites:
            _weight_array_away[_iTrial, dend.rank] = dend.w

    print(("Network %i: Simulation time: %f" %(i_seed, ttime() - start_time)))

    return _latency_array_sec, _seq_endpoint_final_array, S_context_home_DG.w, _seq_count_array, _seq_start_array, _seq_endpoint_array, _random_nav_time_array_sec,\
                _goal_nav_time_array_sec, _focal_search_time_array_sec, data['occupancyMap'], _weight_array_home, _weight_array_away, data['center_mat_plot'], _center_mat_array,\
                _goal_index_array

#------------------------------------------------------------------------------------------------------------------------
def initiate_sequence(start_index, plotting, netw_index):
    global placevalMat
    global sim
    global data

    sim['ongoingSeq'] = True
    net_full.get(S_Exc_Exc).transmission = True
    start_bias = zeros(netw['n_place_cells'])
    start_bias[list(range(netw['n_place_cells']))] = net_init.Exp_xyValue2(list(range(netw['n_place_cells'])), netw['n_grid'], 0, start_index, 20*80/63.0) 

    #x_centers, y_centers = net_init.xypos_maze( range(netw['n_place_cells']), netw['n_grid'], maze['L_cm'] )
    #print("start_bias decoded: x,y = %f.1, %f.1" %( np.average(x_centers, weights=start_bias), np.average(y_centers, weights=start_bias)))

    # initiation period
    #Exc_neurons.I_ext_pA = 0 # initial condition
    net_full.get(Exc_neurons).I_ext_pA = 0 # initial condition

    print(("Network %i: Sequence generation... with Poisson rate= 200 Hz" %(netw_index)))

    #Exc_neurons.I_ext_pA = start_bias* 0.9e3 
    net_full.get(Exc_neurons).I_ext_pA = start_bias* 0.9e3 

    stim_time_ms = 35
    init_time_ms = 50
    
    #net_seqs_move.simulate(stim_time_ms, measure_time=True) 
    simulate_sequence(stim_time_ms, 2.0) # 2.0

    # end of initiation period
    #Exc_neurons.I_ext_pA = 0
    net_full.get(Exc_neurons).I_ext_pA = 0

    #net_seqs_move.simulate((init_time_ms - stim_time_ms), measure_time=True)
    simulate_sequence((init_time_ms - stim_time_ms), 2.0)

    print(("Network %i: Switching to place-value noise..." %(netw_index)))

    #poisson_inp.rate = 200*ones(netw['n_place_cells'])
    net_full.get(poisson_inp).rate = 200*ones(netw['n_place_cells'])
    print(('net_full.get(poisson_inp).rate = ', net_full.get(poisson_inp).rate))
    #print('poisson_inp.rate = ', poisson_inp.rate)

    '''#
    if netw_index == 0:
        print("Poisson rate = ", poisson_inp.rate[0])
        print("S_Poi_cont_home.w = ", S_Poi_cont_home.w)
        print("S_Poi_cont_away.w = ", S_Poi_cont_away.w)
        print("S_context_home_DG.w = ", max(S_context_home_DG.w))
        print("S_context_away_DG.w = ", max(S_context_away_DG.w))
        print("S_DG_Exc.w = ", max(max(S_DG_Exc.w)))
        print("max. S_Exc_Exc.w= ", max(max(S_Exc_Exc.w)))
        print("sum( S_Exc_Exc.w )= ", sum(S_Exc_Exc.w))
        print("const. S_Exc_Inh.w= ", S_Exc_Inh.w)
        print("const. S_Inh_Inh.w= ", S_Inh_Inh.w)
        print("const. S_Inh_Exc.w= ", S_Inh_Exc.w)
    '''

    #print("Performing sequence generation...")
    if netw_index == 0:
        simulate_sequence(350, 2.0) # 2.0
    else:
        simulate_sequence(350, 2.0) # 2.0

    #poisson_inp.rate = zeros(netw['n_place_cells'])
    net_full.get(poisson_inp).rate = zeros(netw['n_place_cells'])
    
    # Network reset
    #Exc_neurons.I_exc = 0
    net_full.get(Exc_neurons).I_exc = 0
    data['call_counter'] = 0
    data['n_spikes'] = 0
    data['x_center_sum'], data['y_center_sum'] =  0, 0        

    #simulate(30)
    net_full.simulate(30)
    net_full.get(S_Exc_Exc).transmission = False
  
    #'''#
    ion()
    subplot(611)
    poi_spikes = net_full.get(m_poi).get(['spike'])
    te, ne = net_full.get(m_poi).raster_plot(poi_spikes)
    plot(te, ne, '.')
    ylabel('Poisson')

    subplot(612)
    context_spikes = net_full.get(m_contpop).get(['spike'])
    te, ne = net_full.get(m_contpop).raster_plot(context_spikes)
    plot(te, ne, '.')
    ylabel('Home Cont.')

    subplot(613)
    dg_spikes = net_full.get(m_dg).get(['spike'])
    te, ne = net_full.get(m_dg).raster_plot(dg_spikes)
    plot(te, ne, '.')
    ylabel('DG')
    #'''

    subplot(614)
    exc_spikes = net_full.get(m_exc).get(['spike'])
    te, ne = net_full.get(m_exc).raster_plot(exc_spikes)
    #plot(te, ne, '.')
    plot(mon_te, mon_ne, 'b.')
    ylabel('CA3')

    subplot(615)
    xn, yn = net_init.xypos_maze(mon_ne, netw['n_grid'], maze['L_cm'])
    for i in range(len(mon_te)):
        plot(xn[i], yn[i], '.', color=str(1 - mon_te[i]/max(mon_te)))

    subplot(616)
    plot(mon_xest, mon_yest, 'r')
    for i in range(len(mon_xest)):
        plot(mon_xest[i], mon_yest[i], '.', color=str(1 - float(i)/len(mon_xest)))
    axis([0, 450, 0, 450])


    #subplot(414)
    #inh_spikes = net_full.get(m_inh).get(['spike'])
    #te, ne = net_full.get(m_inh).raster_plot(inh_spikes)
    #plot(te, ne, '.')

    savefig('exc_raster_id'+str(np.random.randint(100))+'_t'+str(int(net_full.get_time()))+'.png')
    ioff()
    show()
    #'''

    if plotting:
        #raster_plot(M_sp_DG, M_sp_Exc_neurons, M_sp_Inh)    

        # test:
        #if useMonitors:
        #    figure()
        #    raster_plot(M_sp_Exc_neurons, M_sp_DG)
        #    title('Place cells and DG cells')

        figure()
        matshow(transpose(data['center_mat_plot']), cmap=cm.YlOrRd, fignum=False)
        colorbar()
        title('Bump center location plotted across time')
        xlabel('x position')
        ylabel('y position')
        ax = gca()
        ax.set_xticks([0, len(data['center_mat_plot'][0,:])])
        ax.set_xticklabels([0, maze['L_cm']])
        ax.set_yticks([0, len(data['center_mat_plot'][:,0])])
        ax.set_yticklabels([0, maze['L_cm']])     
        savefig('mat_plot_id'+str(np.random.randint(100))+'_t'+str(int(net_full.get_time()))+'.png')

        matshow( np.reshape( net_full.get(S_context_home_DG).w, (netw['n_grid'], netw['n_grid']) ))
        colorbar()
        savefig('weight_plot_id'+str(np.random.randint(100))+'_t'+str(int(net_full.get_time()))+'.png')
        ioff()
        show()


#------------------------------------------------------------------------------------------------------------------------

if __name__ == '__main__':

    #compile()

    nTrials = 1 # 3
    #n_grid = 80 
    N = 1 # 

    identifier = 'recoded_ann_seqtest_N'+str(N)+'_'+str(nTrials)+'trials'

    #dataman = DataManager(identifier)
    n_processes = 3 # 10 #

    stim(False, nTrials, 0)
    #_lat, _seq_end, S_context_home_DG_w, _seq_count, _seq_start, _seq_endp, _random_nav_t, _goal_nav_t, _focal_search_t, data_occupancyMap, \
    #        _weight_array_home, _weight_array_away, data_center_mat_plot, _center_mat, _goal_ind = stim(False, nTrials, 0)
    #print("_weight_array_home.max(), _weight_array_home.min() = ", _weight_array_home.max(), _weight_array_home.min())


    '''#	
    while dataman.itemcount() < N:
        run_tasks(dataman, stim, [(False, nTrials, x) for x in range(dataman.itemcount(), min(N, n_processes + dataman.itemcount()) )], gui=False, poolsize=n_processes) # Creates a number of sub-processes defined by "n_processes"
    #run_tasks(dataman, stim, [(False, nTrials, x) for x in range(dataman.itemcount(), min(N, n_processes + dataman.itemcount()) )], gui=False, poolsize=n_processes)
    '''

    print("All tasks finished.")















    
