#from brian import *
from ANNarchy import *

from brian.tools.datamanager import *
from brian.tools.taskfarm import *
from random import *
from time import time as ttime
import pickle
import matplotlib.cm as cm
import gc
from random import seed as pyseed
from numpy.random import seed as npseed

from pylab import *
import numpy as np

# TODO:
# Learning at Context-to-DG synapses does not seem to work!
# - Is the DA variables correctly set to 1?
# - Do I need to implement weight changes as w = clip(w + dw) as in the Annarchy doc?!
 

setup(dt = 0.2)

import ann_InitLearn as net_init # xypos_maze, xy_index_maze, initConstants, quad_dist_grid, Exp_xyValue2
from ann_recoded_functions import rbc_wrapper, cpf_wrapper, nav_to_goal_wrapper, goal_check_wrapper, reward_delivery_wrapper, set_start_location, calc_focal_search_ann, do_focal_search, prepare_sequential_search, set_subgoal, subgoal_coords, subgoal_reached, update_startloc, set_search_goal, continue_search, prepare_sequence, prepare_navigation, reward_placement

from ann_init_vars import agent, task, mvc, maze, sim, data, netw

from ann_network_fullsim import poisson_inp, context_pop_home, context_pop_away, DG_neurons, S_Poi_cont_home, S_Poi_cont_away, S_context_home_DG, S_context_away_DG, Exc_neurons, Inh_neurons, S_Exc_Exc, S_Exc_Inh, S_Inh_Exc, S_Inh_Inh, simple_syn, S_DG_Exc

# TODO: Check bump decoding 

mon_te, mon_ne = array([]), array([])
mon_xest, mon_yest= array([]), array([])
mon_te_dg, mon_ne_dg = array([]), array([]) # TEST
mon_te_poi, mon_ne_poi = array([]), array([]) # TEST
mon_te_cont, mon_ne_cont = array([]), array([]) # TEST
mon_te_contaway, mon_ne_contaway = array([]), array([]) # TEST


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

	global mon_te_dg
	global mon_ne_dg
	global mon_te_poi
	global mon_ne_poi
	global mon_te_cont
	global mon_ne_cont
	global mon_te_contaway
	global mon_ne_contaway

	exc_spikes = net_full.get(m_exc).get(['spike'])
	te, ne = net_full.get(m_exc).raster_plot(exc_spikes)
	nz_spiking = nonzero(te)
	ni = ne[nz_spiking]

	dg_spikes = net_full.get(m_dg).get(['spike'])# TEST
	te_dg, ne_dg = net_full.get(m_dg).raster_plot(dg_spikes)
	poi_spikes = net_full.get(m_poi).get(['spike'])# TEST
	te_poi, ne_poi = net_full.get(m_poi).raster_plot(poi_spikes)
	cont_spikes = net_full.get(m_conthome).get(['spike'])# TEST
	te_cont, ne_cont = net_full.get(m_conthome).raster_plot(cont_spikes)
	cont_away_spikes = net_full.get(m_contaway).get(['spike'])# TEST
	te_contaway, ne_contaway = net_full.get(m_contaway).raster_plot(cont_away_spikes)




	if sim['ongoingSeq']==True:
	#if sim['ongoingSeq']==True or sim['ongoingSeq']==False: # TEST for place-cell recording
	    data = rbc_wrapper(ni, data, netw['n_grid'], maze['L_cm'], sim['trial'], net_full.get_time())

	    mon_te = append(mon_te, te)
	    mon_ne = append(mon_ne, ne)

	    mon_te_dg = append(mon_te_dg, te_dg) # TEST
	    mon_ne_dg = append(mon_ne_dg, ne_dg)
	    mon_te_poi = append(mon_te_poi, te_poi) # TEST
	    mon_ne_poi = append(mon_ne_poi, ne_poi)
	    mon_te_cont = append(mon_te_cont, te_cont) # TEST
	    mon_ne_cont = append(mon_ne_cont, ne_cont)
	    mon_te_contaway = append(mon_te_contaway, te_contaway) # TEST
	    mon_ne_contaway = append(mon_ne_contaway, ne_contaway)


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
		net_full.get(S_context_home_DG).DA = DG_cells_dopamine # Although dopamine is delivered globally, learning should be gated by activity
		net_full.get(S_context_away_DG).DA = DG_cells_dopamine

#-------------------------------------------------------------------------------------------------------------    
#'''#

def test_function(n):
	global agent
	#print "Time [sec], agent['hasGoal'], agent['focal_search'], agent['random_search'], sim['stopping'] = ", net_full.get_time()*1e-3, agent['hasGoal'], agent['focal_search'], agent['random_search'], sim['stopping']
	print("Time [sec] = ", net_full.get_time()*1e-3)


#'''
#-------------------------------------------------------------------------------------------------------------    

def resume_navigation(sim, context_pop_home):
	sim['stopping'] = False
	return sim

#-------------------------------------------------------------------------------------------------------------    
def simulate_wrapper(t_max_sim_ms, sim_step_ms, netw_id=Inf):
	t_init_ms = net_full.get_time()

	while(net_full.get_time() - t_init_ms < t_max_sim_ms - 0.5*sim_step_ms and sim['stopping']==False):
		# functions to be called between simulation blocks:
		running_bump_center(0) # TEST - clear memory 
		check_placefields(0) # disabled for NO SIMULATION
		navigateToGoal(0)
		goalCheck(0)
		reward_delivery(0)
		test_function(0)

		#net_full.simulate(min(sim_step_ms, t_max_sim_ms - (net_full.get_time() - t_init_ms))) # "Normal" simulation continues after reward delivery
		if sim['reward_found']==False: 
		    net_full.simulate(min(sim_step_ms, t_max_sim_ms - (net_full.get_time() - t_init_ms))) # TEST: Stop simulation after reward delivery

		#if mod(net_full.get_time() - t_init_ms, 1000) < 0.1 : # TEST
		#    plot_sequence(netw_id)

#-------------------------------------------------------------------------------------------------------------

def simulate_sequence(t_max_sim_ms, sim_step_ms):
	t_init_ms = net_full.get_time()
	while(net_full.get_time() - t_init_ms < t_max_sim_ms - 0.5*sim_step_ms):
		running_bump_center(0) # Deactivate to obtain a raster plot
		net_full.simulate(min(sim_step_ms, t_max_sim_ms - (net_full.get_time() - t_init_ms)))



#-------------------------------------------------------------------------------------------------------------    

def plot_sequence(netw_index):
    # What I see in this plot from the second trial on:
    # 100ms of simulation following reward delivery (might be switched off in simulate_wrapper() )
    # 100ms of learning phase
    # 50ms are missing / unexplained - where do they come from?!

    global net_full
    global mon_te 
    global mon_ne
    global mon_xest
    global mon_yest

    global mon_te_dg # TEST
    global mon_ne_dg
    global mon_te_poi # TEST
    global mon_ne_poi
    global mon_te_cont # TEST
    global mon_ne_cont
    global mon_te_contaway # TEST
    global mon_ne_contaway


    if len(mon_te) > 0 and len(mon_te_dg) > 0:
        tmin = min(min(mon_te), min(mon_te_dg))
        tmax = max(max(mon_te), min(mon_te_dg))
    else:
        tmin = 0
        tmax = 400

    '''#
    subplot(411)
    poi_spikes = net_full.get(m_poi).get(['spike'])
    te, ne = net_full.get(m_poi).raster_plot(poi_spikes)
    plot(mon_te_poi, mon_ne_poi, 'b.')
    plot(te, ne, 'r.')
    ylabel('Poisson inp.')
    axis([tmin, tmax, 0, 50])
    '''

    subplot(411)
    cont_spikes = net_full.get(m_conthome).get(['spike'])
    te, ne = net_full.get(m_conthome).raster_plot(cont_spikes)
    plot(mon_te_cont, mon_ne_cont, 'b.')
    plot(te, ne, 'r.')

    te, ne = net_full.get(m_contaway).raster_plot(cont_spikes)
    plot(mon_te_contaway, mon_ne_contaway, 'k.')
    plot(te, ne, 'g.')

    ylabel('Context pop.')
    axis([tmin, tmax, 0, 50])


    subplot(412)
    dg_spikes = net_full.get(m_dg).get(['spike'])
    te, ne = net_full.get(m_dg).raster_plot(dg_spikes)
    plot(mon_te_dg, mon_ne_dg, 'b.')
    plot(te, ne, 'r.')
    ylabel('DG')
    axis([tmin, tmax, 0, 7000])

    #subplot(413)
    #dg_gexc = net_full.get(m_dg).get(['g_exc'])
    #plot(dt() * arange(dg_gexc.shape[0]), 1e-3 * np.mean(dg_gexc, 1))
    #ylabel('g_exc (DG)')
    #axis([tmin, tmax, 0, axis()[3]])

    subplot(414)
    exc_spikes = net_full.get(m_exc).get(['spike'])
    te, ne = net_full.get(m_exc).raster_plot(exc_spikes)
    #plot(te, ne, '.')
    plot(mon_te, mon_ne, 'b.')
    plot(te, ne, 'r.')
    ylabel('CA3')
    axis([tmin, tmax, 0, 7000])

    #subplot(312)
    #xn, yn = net_init.xypos_maze(mon_ne, netw['n_grid'], maze['L_cm'])
    #tmax = max(mon_te)
    #for i in xrange(len(mon_te)):
    #    plot(xn[i], yn[i], '.', color=str(1 - mon_te[i]/tmax)) # VERY SLOW PLOT?!

    mon_te, mon_ne = array([]), array([])
    mon_te_dg, mon_ne_dg = array([]), array([]) # TEST
    mon_te_poi, mon_ne_poi = array([]), array([]) # TEST
    mon_te_cont, mon_ne_cont = array([]), array([])

    #subplot(414)
    #plot(mon_xest, mon_yest, 'r')
    #for i in xrange(len(mon_xest)):
    #    plot(mon_xest[i], mon_yest[i], '.', color=str(1 - float(i)/len(mon_xest)))
    #axis([0, 450, 0, 450])
    #mon_xest, mon_yest= array([]), array([])

    #savefig('exc_raster_id'+str(np.random.randint(100))+'.png')
    #savefig('exc_raster_id'+str(np.random.randint(100))+'_t'+str(int(net_full.get_time()))+'.png')
    savefig('exc_raster_netw'+str(netw_index)+'_t'+str(int(net_full.get_time()))+'.png')
    clf()


    

#i_sigma_dist_cm, i_wmax_Place_cells_recur, i_wmax_Place_cells_inh,\
#i_wmax_inh_inh, i_wmax_inh_Place_cells, i_pconn_recur, i_tau_membr, i_tau_exc, i_tau_inh, i_dt = range(10)

#-------------------------------------------------------------------------------------------------------------

#netw_params = 	[sigma_dist_cm, wmax_exc_exc, wmax_exc_inh, wmax_inh_inh, wmax_inh_exc, pconn_recur, tau_membr, tau_exc, tau_inh,   dt    ]
#netw_params = 	[50.0, 	        2*0.4114*nA, 12*0.8125*pA, 6*6*13.0*pA, 0.9*5*4.1*pA,  1, 10, 6,   2,  0.2] # 
#print "netw_params= ", netw_params

# Constants - neuron model and network

useMonitors =  True # False #
if useMonitors:
    print("Caution, using place cell monitors!")


global agent
global task
global sim
global data

data['occupancyMap'] = Inf * ones([40, maze['L_cm'] / maze['spatialBinSize_cm'], maze['L_cm'] / maze['spatialBinSize_cm']]) 

global placevalMat
placevalMat = zeros(netw['n_place_cells'])


data['start_index'] = netw['n_grid'] * (netw['n_grid'] - 1) # 45*44 # 45**2 - 1 #0 #44


m_poi = Monitor(poisson_inp[0:50], ['spike'])
m_conthome = Monitor(context_pop_home[0:50], ['spike'])
m_contaway = Monitor(context_pop_away[0:50], ['spike'])
m_dg = Monitor(DG_neurons, ['spike'])#, 'g_exc']) # TEST
m_exc = Monitor(Exc_neurons, ['spike']) 
#m_inh = Monitor(Inh_neurons, ['spike'])


#'''#
if useMonitors: 
    #net_move = Network(poisson_inp, #context_pop_home, context_pop_away, 
	#			  #Exc_neurons, DG_neurons, \
    #              #S_Poi_cont_home, S_Poi_cont_away, S_context_home_DG, S_context_away_DG,\
    #              #M_sp_DG, M_sp_Exc_neurons,\
    #              navigateToGoal, goalCheck, reward_delivery, check_placefields) 

    #net_seqs_move = Network(poisson_inp)#, context_pop_home, context_pop_away, DG_neurons, 
					   #Exc_neurons, Inh_neurons, \
                       #S_Poi_cont_home, S_Poi_cont_away, S_context_home_DG, S_context_away_DG, S_DG_Exc, S_Exc_Exc, S_Exc_Inh, S_Inh_Exc, S_Inh_Inh,\
                       #M_sp_DG, M_sp_Exc_neurons, M_sp_Inh, 
					   #M_bump_center)

    net_full = Network(everything=True)

else:
    #net_full = Network(poisson_inp, context_pop_home, context_pop_away, DG_neurons, Exc_neurons, Inh_neurons, \
    #                    S_Poi_cont_home, S_Poi_cont_away, S_context_home_DG, S_context_away_DG, S_DG_Exc, S_Exc_Exc, S_Exc_Inh, S_Inh_Exc, S_Inh_Inh)
    net_full = Network(everything=True) # remove monitor m_Exc?!

#'''


#compile()
net_full.compile()

print("Modifying synaptic weights...")
for dend in net_full.get(S_Exc_Exc).dendrites:
    dend.w *= (1.2*np.random.rand(dend.size)+0.4) 
print("Done")

report(filename = "./ann_recoded_model_report.tex")

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
    print("Network %i: Home reward location (x,y) = %f, %f" %(i_seed, task['x_reward_cm'], task['y_reward_cm'])) 
    task['goal_index'] = task['home_index']
    agent['random_search'] = False


    for _iTrial in range(nTrials):
        print("Network %i: Trial %i of %i" %(i_seed, _iTrial+1, nTrials))
        iStim = _iTrial
        sim['trial'] = _iTrial
        _start_loc = rand()
        #print "_goal_index_array[_iTrial], task['goal_index'], task['home_index'] = ", _goal_index_array[_iTrial], task['goal_index'], task['home_index']
        #print "_goal_index_array[_iTrial] = ", _goal_index_array[_iTrial]
        #print "task['home_index'] (calc) = ", net_init.xy_index_maze(maze['edge_cm'] + 0.5*maze['grid_factor'] + i_home * maze['grid_factor'], maze['edge_cm'] + 0.5*maze['grid_factor'] + j_home * maze['grid_factor'], netw['n_grid'], maze['L_cm']) # task['home_index']
        #print "task['goal_index'] = ", task['goal_index']

        _goal_index_array[_iTrial] = task['goal_index']    
        agent['visited_index_36'] = zeros(36) # reset for directed random search
        agent['visited_index_6x6'] = zeros([6,6])

        if _iTrial == 0: 
                agent['x_cm'], agent['y_cm'] = set_start_location(_start_loc, maze['edge_cm'], maze['L_cm'], agent['x_cm'], agent['y_cm'], task['x_reward_cm'], task['y_reward_cm'])
                print("Network %i: Start position (x,y) = %i, %i" %(i_seed, agent['x_cm'], agent['y_cm']))
                data['start_index'] = net_init.xy_index_maze(agent['x_cm'], agent['y_cm'], netw['n_grid'], maze['L_cm']) 
        if agent['x_cm'] < maze['edge_cm'] or agent['y_cm'] < maze['edge_cm'] or agent['x_cm'] > maze['L_cm'] - maze['edge_cm'] - 1 or agent['y_cm'] > maze['L_cm'] - maze['edge_cm'] - 1:
            print("Starting outside of maze area!")

        sim['reward_found'] = False
        agent['hasGoal'] = False 
        agent['focal_search'] = False
        sim['ongoingSeq'] = False
        _start_time_msec = net_full.get_time() # get_time()

        if sim['home_trial']:
            #for dend in S_context_home_DG.dendrites:
            for dend in net_full.get(S_context_home_DG).dendrites:
                #if dend.rank==0:
                #    print "rank 0: dend.w = ", dend.w
                placevalMat[dend.rank] = dend.w
        else:
            #for dend in S_context_away_DG.dendrites:
            for dend in net_full.get(S_context_away_DG).dendrites:
                #if dend.rank==0:
                #    print "rank 0: dend.w = ", dend.w
                placevalMat[dend.rank] = dend.w


        if sim['home_trial']:
            #S_Poi_cont_home.w = 21.0
            #S_Poi_cont_away.w = 0.0
            net_full.get(S_Poi_cont_home).w = 21.0
            net_full.get(S_Poi_cont_away).w = 0.0
        else:
            #S_Poi_cont_home.w = 0.0
            #S_Poi_cont_away.w = 21.0
            net_full.get(S_Poi_cont_home).w = 0.0
            net_full.get(S_Poi_cont_away).w = 21.0

        counter_0 = 0

        #while sim['reward_found'] == False and get_time()  < (mvc['time_out_sec']*1e3 - 1) : # Global timeout based on all trials 
        while sim['reward_found'] == False and net_full.get_time()  < (mvc['time_out_sec']*1e3 - 1) : # Global timeout based on all trials 
            counter_0 += 1
            if counter_0 > 1:
                print("Network %i: counter_0 = %i" %(i_seed, counter_0))
                
            sim['value_gain'] = 0

            # Sequence generation for goal-setting:
            sim, data, curr_pos_index, _seq_start_array = prepare_sequence(_iTrial, agent, netw, maze, sim, data, _seq_count_array, _seq_start_array)

            initiate_sequence(curr_pos_index, False, i_seed) #  # True # Disabled: HACK for NO SIMULATION
            print("Network %i: Agent pos. (x,y) = %i, %i" %(i_seed, agent['x_cm'], agent['y_cm']))
            sx, sy = net_init.xypos_maze(curr_pos_index, netw['n_grid'], maze['L_cm'])
            print("Network %i: Sequence start (x,y) = %i, %i" %(i_seed, agent['x_cm'], agent['y_cm']))

            _seq_count_array[_iTrial] += 1
            sim['ongoingSeq'] = False	    
            # Sequence has finished - determine the navigation goal as the end point of the sequence:
            agent['xGoal_cm'], agent['yGoal_cm'] = net_init.xypos_maze(data['center_mat'].argmax(), len(data['center_mat'][0,:]), maze['L_cm'])

            #-------------------------- HACK to shorten search!!!
            '''#
            i_nearhome, j_nearhome = net_init.xypos(mod(i_seed+1, 36), 6) # i_seed + 1 !!!
            nearhome_index = net_init.xy_index_maze(maze['edge_cm'] + 0.5*maze['grid_factor'] + i_nearhome * maze['grid_factor'], maze['edge_cm'] + 0.5*maze['grid_factor'] + j_nearhome * maze['grid_factor'], netw['n_grid'], maze['L_cm'])
            agent['xGoal_cm'], agent['yGoal_cm'] = net_init.xypos_maze(nearhome_index, netw['n_grid'], maze['L_cm'])
            print "Sequence endpoint (HACK): x,y = ", agent['xGoal_cm'], agent['yGoal_cm']  
            '''
            #-------------------------- END OF HACK

            print("Sequence endpoint: x,y = ", agent['xGoal_cm'], agent['yGoal_cm'])  

            #--------------------------
            #agent['xGoal_cm'], agent['yGoal_cm'] = 150, 250 # 150, 150 # HACK for NO SIMULATION !!!
            #if sim['home_trial']:
            #	agent['xGoal_cm'], agent['yGoal_cm'] = 150, 250 # HACK for NO SIMULATION !!!
            #else: 
            #	agent['xGoal_cm'], agent['yGoal_cm'] = 180, 220 # HACK for NO SIMULATION !!!
            #--------------------------



            data, _seq_endpoint_final_array, _seq_endpoint_array, agent, sim = prepare_navigation(_iTrial, i_seed, agent, netw, maze, sim, data, _seq_endpoint_final_array, _seq_count_array, _seq_start_array, _seq_endpoint_array)

            # Goal-directed navigation simulation:
            net_full.get(poisson_inp).rates = 10*ones(netw['n_place_cells'])

            if sim['value_gain'] > 0:
                print("Network %i: navigation to replay goal..." %(i_seed))
                movement_length_sec = 1.2*1.5 * maze['L_cm'] * np.sqrt(net_init.quad_dist_grid(curr_pos_index, data['endpoint'] , netw['n_grid'])) / mvc['speed_cm_s'] # error corrected 22.7.14
                #_start_time_navigate_msec = get_time()
                _start_time_navigate_msec = net_full.get_time()
                print("Total simulation time for movement [sec]: ", movement_length_sec)
                simulate_wrapper(movement_length_sec * 1000, 100, i_seed) # TEST for "every" decorator - functions are called!
                print("Replay goal navigation finished.")
                sim = resume_navigation(sim, context_pop_home)
                _goal_nav_time_array_sec[_iTrial] += 1e-3 * (get_time() - _start_time_navigate_msec)

            if agent['focal_search']:
                # Search at the nearest four reward wells
                time_focal_sec, x_feeder_cm, y_feeder_cm, nearestfour_36, _start_time_focal_msec = calc_focal_search_ann(mvc['speed_cm_s'], mvc['time_out_sec'], net_full.get_time(), _start_time_msec, maze['edge_cm'], maze['grid_factor'], agent['x_cm'], agent['y_cm'])
                for i_feeder in range(4): # Visit the nearest four feeders, the closest one first
                    agent['xGoal_cm'], agent['yGoal_cm'], agent['focal_search'], agent['random_search'], sim['value_gain'], agent['hasGoal'] = do_focal_search(x_feeder_cm, y_feeder_cm, nearestfour_36, i_feeder, i_seed)
                    #simulate_wrapper( 1e3 * (30.0/mvc['speed_cm_s'])* 4, 100) # Too short?!
                    simulate_wrapper( 1e3 * (30.0/mvc['speed_cm_s'])* 8, 100, i_seed)
                    sim = resume_navigation(sim, context_pop_home)

                    if agent['hasGoal'] == False:
                        agent['visited_index_36'][nearestfour_36[i_feeder]] = 1 # Don't search here again in the same trial!
                    if sim['reward_found']:
                        break
                    else: # new 7.12.15: "Prediction error" signal after each visited nearby feeder! ("DAdecr4")
                    #elif i_feeder == 0: # new 7.12.15: "Prediction error" signal only after the first feeder! ("DAdecr1st")
                    	#DG_neurons.DA = -0.5 # -1 
                    	net_full.get(DG_neurons).DA = -0.5
                    	net_full.get(S_context_home_DG).DA = -0.5
                    	net_full.get(S_context_away_DG).DA = -0.5

		        ## learning should now take place
		        net_full.simulate(100)
		        test_function(0) # TEST for plotting 

		        #DG_neurons.DA = 0  
		        net_full.get(DG_neurons).DA = 0
		        net_full.get(S_context_home_DG).DA = 0
		        net_full.get(S_context_away_DG).DA = 0

                #_focal_search_time_array_sec[_iTrial] += 1e-3 * (get_time() - _start_time_focal_msec)
                _focal_search_time_array_sec[_iTrial] += 1e-3 * (net_full.get_time() - _start_time_focal_msec)

            #_start_time_navigate_msec = get_time()
            _start_time_navigate_msec = net_full.get_time()
            while sim['reward_found']==False: # perform sequential search ...
                agent['random_search'], sim['value_gain'], search_counter, agent['visited_index_36'], agent['search_loc_index_36'], \
                	agent['visited_index_6x6'], visited_index_6x6_potential_subgoals, ix_6, jy_6, index_nearest_36, \
                	delta_x_6, delta_y_6, x_search_goal_cm, y_search_goal_cm, nmax_pot_subgoals, counter, ix_search_index_6, jy_search_index_6, i_subgoal_36 = \
					prepare_sequential_search(agent['x_cm'], agent['y_cm'], maze['edge_cm'], agent['visited_index_36'], sim['home_trial'], home_index_36, maze['grid_factor'], i_seed)
               
                while sim['reward_found']==False and sum(visited_index_6x6_potential_subgoals) < nmax_pot_subgoals and counter < 35-sum(agent['visited_index_36']):
                    counter += 1
                    #print "while loop, subgoal search: counter = ", counter
                    i_subgoal_36 = set_subgoal(delta_x_6, delta_y_6, ix_6, jy_6, agent['visited_index_36'])
        
                    if agent['visited_index_36'][int(i_subgoal_36)] == False:
                        #print "x, y, xGoal, yGoal, i_subgoal_36, agent['visited_index_36'] = ", agent['x_cm'], agent['y_cm'], agent['xGoal_cm'], agent['yGoal_cm'], i_subgoal_36, agent['visited_index_36']
                        task['goal_index'], agent['hasGoal'], search_length_sec, agent['xGoal_cm'], agent['yGoal_cm'] = subgoal_coords(i_subgoal_36, maze['edge_cm'], netw['n_grid'], maze['L_cm'], agent['x_cm'], agent['y_cm'], x_search_goal_cm, y_search_goal_cm, mvc['speed_cm_s'], i_seed, agent['visited_index_36'])
                        simulate_wrapper( search_length_sec * 1e3, 100, i_seed)
                        sim = resume_navigation(sim, context_pop_home)                    

                        agent['focal_search'], agent['visited_index_36'], agent['visited_index_6x6'] = subgoal_reached(agent['hasGoal'], agent['visited_index_36'], agent['visited_index_6x6'], i_subgoal_36)
                    else:
                        #print "subgoal search ended"
                        break # while loop

                    # updating start location information for the next "while" iteration
                    index_nearest_36, delta_x_6, delta_y_6, visited_index_6x6_potential_subgoals, nmax_pot_subgoals = update_startloc(agent['x_cm'], agent['y_cm'], maze['edge_cm'], ix_search_index_6, jy_search_index_6, agent['visited_index_6x6'])

                if sim['reward_found'] == False: # After "subgoal" search - proceed to the "search goal"
					agent['xGoal_cm'], agent['yGoal_cm'], search_length_sec, agent['hasGoal'], task['goal_index'] = set_search_goal(x_search_goal_cm, y_search_goal_cm, netw['n_grid'], maze['L_cm'], agent['x_cm'], agent['y_cm'], mvc['speed_cm_s'])
					print("Network %i: Searching random goal %i at %i, time [s]: %f" %(i_seed, sum(agent['visited_index_36']), agent['search_loc_index_36'], search_length_sec))
					simulate_wrapper( search_length_sec * 1e3, 100, i_seed)
					sim = resume_navigation(sim, context_pop_home)
					#print "x, y, agent['visited_index_36'] = ", agent['x_cm'], agent['y_cm'], agent['visited_index_36']
                   
					agent['focal_search'], agent['visited_index_36'], agent['visited_index_6x6'] =  continue_search(agent['hasGoal'], agent['visited_index_36'], agent['search_loc_index_36'], i_subgoal_36, i_seed, agent['visited_index_6x6']) 


            agent['random_search'] = False
            #_random_nav_time_array_sec[_iTrial] += 1e-3 * (get_time() - _start_time_navigate_msec)
            _random_nav_time_array_sec[_iTrial] += 1e-3 * (net_full.get_time() - _start_time_navigate_msec)

            if sim['reward_found']:
                #simulate(100)  # learning should now take place
                print("Network %i: ")
                print("net_full.get(S_context_home_DG).dendrite(0).DA = ", net_full.get(S_context_home_DG).dendrite(0).DA)
                print("max(net_full.get(S_context_home_DG).dendrite(0).w) = ", max(net_full.get(S_context_home_DG).dendrite(0).w))
                net_full.simulate(100)  # learning should now take place
                net_full.get(DG_neurons).DA = 0 # prevent learning from occurring during sequence generation
                net_full.get(S_context_home_DG).DA = 0  # Correct way to prevent learning from occurring during sequence generation
                net_full.get(S_context_away_DG).DA = 0


                print("net_full.get(S_context_home_DG).dendrite(0).DA = ", net_full.get(S_context_home_DG).dendrite(0).DA)
                print("max(net_full.get(S_context_home_DG).dendrite(0).w) = ", max(net_full.get(S_context_home_DG).dendrite(0).w))
                # NOW, the reward should be replaced. The activity of the context populations will be switched at the beginning of the next trial

                if sim['home_trial']: 
                    task = reward_placement(i_seed, task, sim, maze, netw)
                    if len(np.nonzero(_goal_index_array[0:_iTrial] == task['goal_index'])[0]) > 0:
                        task = reward_placement(i_seed, task, sim, maze, netw) # Prevent repeating Away reward locations
                    sim['home_trial'] = False # next trial will be an Away-trial
                else:
                    task = reward_placement(i_seed, task, sim, maze, netw)
                    sim['home_trial'] = True  # next trial will be a Home-trial
                #print "task['home_index'] = ", task['home_index']
                #print "task['goal_index'] = ", task['goal_index']

        #_start_array[_iTrial]    = iStim 
        _latency_array_sec[_iTrial] = 1e-3 * (net_full.get_time() - _start_time_msec)

        for dend in net_full.get(S_context_home_DG).dendrites:
		    _weight_array_home[_iTrial, dend.rank] = dend.w
        for dend in net_full.get(S_context_away_DG).dendrites:
		    _weight_array_away[_iTrial, dend.rank] = dend.w

        ix,jy = np.nonzero(data['center_mat_plot']==_iTrial)
        for i_pos in range(len(ix)):
            _center_mat_array[_iTrial][ix[i_pos], jy[i_pos]] = data['center_mat'][ix[i_pos], jy[i_pos]]

        print("Network %i: No. of sequences= %i, latency to reach reward= %f " %(i_seed, _seq_count_array[_iTrial], _latency_array_sec[_iTrial]))

    print("Network %i: Simulation time: %f" %(i_seed, ttime() - start_time))

    return _latency_array_sec, _seq_endpoint_final_array, S_context_home_DG.w, _seq_count_array, _seq_start_array, _seq_endpoint_array, _random_nav_time_array_sec,\
                _goal_nav_time_array_sec, _focal_search_time_array_sec, data['occupancyMap'], _weight_array_home, _weight_array_away, data['center_mat_plot'], _center_mat_array,\
                _goal_index_array

#------------------------------------------------------------------------------------------------------------------------
def initiate_sequence(start_index, plotting, netw_index):
    global placevalMat
    global sim
    global data

    global mon_te
    global mon_ne
    global mon_xest
    global mon_yest # 

    global mon_te_dg # TEST
    global mon_ne_dg
    global mon_te_poi # TEST
    global mon_ne_poi


    sim['ongoingSeq'] = True
    net_full.get(S_Exc_Exc).transmission = True
    start_bias = zeros(netw['n_place_cells'])
    start_bias[list(range(netw['n_place_cells']))] = net_init.Exp_xyValue2(list(range(netw['n_place_cells'])), netw['n_grid'], 0, start_index, 20*80/63.0) 

    #x_centers, y_centers = net_init.xypos_maze( range(netw['n_place_cells']), netw['n_grid'], maze['L_cm'] )
    #print "start_bias decoded: x,y = %f.1, %f.1" %( np.average(x_centers, weights=start_bias), np.average(y_centers, weights=start_bias))

    # initiation period
    #Exc_neurons.I_ext_pA = 0 # initial condition
    net_full.get(Exc_neurons).I_ext_pA = 0 # initial condition

    print("Network %i: Sequence generation... with Poisson rate= 200 Hz" %(netw_index))

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

    print("Network %i: Switching to place-value noise..." %(netw_index))

    #poisson_inp.rates = 200*ones(netw['n_place_cells'])
    net_full.get(poisson_inp).rates = 200*ones(netw['n_place_cells'])

    '''#
    if netw_index == 0:
		print "Poisson rate = ", poisson_inp.rates[0] 
		print "S_Poi_cont_home.w = ", S_Poi_cont_home.w
		print "S_Poi_cont_away.w = ", S_Poi_cont_away.w
		print "S_context_home_DG.w = ", max(S_context_home_DG.w)
		print "S_context_away_DG.w = ", max(S_context_away_DG.w)
		print "S_DG_Exc.w = ", max(max(S_DG_Exc.w))
		print "max. S_Exc_Exc.w= ", max(max(S_Exc_Exc.w))
		print "sum( S_Exc_Exc.w )= ", sum(S_Exc_Exc.w)
		print "const. S_Exc_Inh.w= ", S_Exc_Inh.w
		print "const. S_Inh_Inh.w= ", S_Inh_Inh.w
		print "const. S_Inh_Exc.w= ", S_Inh_Exc.w
    '''


    #print "Performing sequence generation..."
    if netw_index == 0:
        #print "max. S_context_home_DG.w = ", max(net_full.get(S_context_home_DG).dendrite(0).w)
        print("max. S_context_home_DG.w = ", max(net_full.get(S_context_home_DG).w))
        simulate_sequence(350, 2.0) # 2.0
    else:
        simulate_sequence(350, 2.0) # 2.0

    #poisson_inp.rates = zeros(netw['n_place_cells'])
    net_full.get(poisson_inp).rates = zeros(netw['n_place_cells'])
    
    # Network reset
    #Exc_neurons.I_exc = 0
    net_full.get(Exc_neurons).I_exc = 0
    data['call_counter'] = 0
    data['n_spikes'] = 0
    data['x_center_sum'], data['y_center_sum'] =  0, 0        

    #simulate(30)
    net_full.simulate(30)
    net_full.get(S_Exc_Exc).transmission = False # Is this in the correct order? Do I have to turn off transmission before the 30ms "fade-out" phase?
  
    plot_sequence(netw_index)

    #ioff()
    #show()


    if plotting:
        raster_plot(M_sp_DG, M_sp_Exc_neurons, M_sp_Inh)    

        # test:
        if useMonitors:
            figure()
            raster_plot(M_sp_Exc_neurons, M_sp_DG)
            title('Place cells and DG cells')

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


        ioff()
        show()



#------------------------------------------------------------------------------------------------------------------------

if __name__ == '__main__':

    #compile()

    nTrials = 3 # 3 
    #n_grid = 80 
    N = 3 # 

    #identifier = 'recoded_ann_dataman_noSim_N'+str(N)+'_'+str(nTrials)+'trials'
    #identifier = 'recoded_ann_dataman_seqplot_N'+str(N)+'_'+str(nTrials)+'trials'
    #identifier = 'recoded_ann_dataman_synV5_N'+str(N)+'_'+str(nTrials)+'trials' # Learning looks good, but decoding errors?!
    #identifier = 'recoded_ann_dataman_synV5a_N'+str(N)+'_'+str(nTrials)+'trials' # not bad - learning rate too high, "noisy" weight pattern
    #identifier = 'recoded_ann_dataman_synV5b_N'+str(N)+'_'+str(nTrials)+'trials' # learning rate somewhat reduced
    identifier = 'recoded_ann_dataman_synV5c_N'+str(N)+'_'+str(nTrials)+'trials' # learning rate further reduced, noise in CA3 reduced (sigma=45 instead of 63), tau_exc=12ms(DG). GOOD!
    # TODO: 
    # - Test with more trials / networks
    # - parallel_run ?

    dataman = DataManager(identifier)
    n_processes = 3 # 10 #

    #stim(False, nTrials, 0)
    #_lat, _seq_end, S_context_home_DG_w, _seq_count, _seq_start, _seq_endp, _random_nav_t, _goal_nav_t, _focal_search_t, data_occupancyMap, \
    #        _weight_array_home, _weight_array_away, data_center_mat_plot, _center_mat, _goal_ind = stim(False, nTrials, 0)
    #print "_weight_array_home.max(), _weight_array_home.min() = ", _weight_array_home.max(), _weight_array_home.min()


    #'''#	
    while dataman.itemcount() < N:
        run_tasks(dataman, stim, [(False, nTrials, x) for x in range(dataman.itemcount(), min(N, n_processes + dataman.itemcount()) )], gui=False, poolsize=n_processes) # Creates a number of sub-processes defined by "n_processes"
    #run_tasks(dataman, stim, [(False, nTrials, x) for x in range(dataman.itemcount(), min(N, n_processes + dataman.itemcount()) )], gui=False, poolsize=n_processes)
    #'''

    print("All tasks finished.")






    








    

