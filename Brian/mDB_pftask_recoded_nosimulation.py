from brian import *
from brian.tools.datamanager import *
from brian.tools.taskfarm import *
from random import *
from time import time
import pickle
import matplotlib.cm as cm
import gc
from random import seed as pyseed
from numpy.random import seed as npseed

set_global_preferences(usecodegenweave=True)

import mDB_azizi_newInitLearn as net_init
from mDB_brian_recoded_functions import rbc_wrapper, cpf_wrapper, nav_to_goal_wrapper, goal_check_wrapper, reward_delivery_wrapper, set_start_location, calc_focal_search, do_focal_search, prepare_sequential_search, set_subgoal, subgoal_coords, subgoal_reached, update_startloc, set_search_goal, continue_search, prepare_sequence, prepare_navigation

from init_vars import agent, task, mvc, maze, sim, data, netw
from network_setup import poisson_inp_uniform, context_pop_home, context_pop_away, DG_place_cells, Place_cells, Inh_neurons, S_Place_cells_recur, S_Place_cells_Inh, S_Inh_Place_cells, S_Inh_Inh, S_Poi_cont_home, S_Poi_cont_away, S_context_home_DG, S_context_away_DG, S_DG_Place_cells, M_sp_Place_cells, M_sp_Inh, M_sp_DG

movement_clock=Clock(dt=mvc['DeltaT_step_sec'] * second)

#--------------------------------------------------------------------------------------------------------
def running_bump_center(spikelist):
    global data
    data = rbc_wrapper(spikelist, data, netw['n_grid'], maze['L_cm'], sim['trial'], defaultclock.t)

#--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------    
@network_operation(movement_clock, when='start')             # to be called according to clock "movement_clock"
def check_placefields():
    global netw    
    Place_cells_addcurrent, DG_place_cells_addcurrent = cpf_wrapper(defaultclock, netw['n_place_cells'], netw['pf_center_x_cm'], netw['pf_center_y_cm'], netw['PlaceFieldRadius_cm'], agent)    
    Place_cells.I_exc[:] += Place_cells_addcurrent
    DG_place_cells.I_exc[:] += DG_place_cells_addcurrent

#--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
@network_operation(movement_clock) # to be called according to clock "movement_clock"
def navigateToGoal():
    global agent
    global sim
    global data
    agent['x_cm'], agent['y_cm'], data['occupancyMap'], agent['direction'] = nav_to_goal_wrapper(movement_clock, agent['x_cm'], agent['y_cm'], agent['xGoal_cm'], agent['yGoal_cm'], agent['hasGoal'], agent['focal_search'], sim['trial'], sim['value_gain'], agent['direction'], agent['new_direction'], sim['ongoingSeq'], mvc['speed_cm_s'], maze['edge_cm'], maze['L_cm'], maze['spatialBinSize_cm'], mvc['turning_prob'], mvc['turn_factor'], mvc['search_radius_cm'], data['occupancyMap']) 
    return
#--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------    
@network_operation(movement_clock) # to be called according to clock "movement_clock"
def goalCheck():
    global agent
    global sim
    agent['hasGoal'], agent['focal_search'], sim['stopping'] = goal_check_wrapper(agent['x_cm'], agent['y_cm'], agent['xGoal_cm'], agent['yGoal_cm'], agent['hasGoal'], agent['focal_search'], sim['value_gain'], agent['random_search'], sim['netw_index'])
    if sim['stopping']:
        net_move.stop()
    return
#--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------    
@network_operation(movement_clock)              # to be called according to clock "movement_clock"
def reward_delivery():
    global agent
    global sim
    global data 
    data['reward_counter'], sim['reward_found'], agent['hasGoal'], sim['stopping'], DG_cells_dopamine = reward_delivery_wrapper(agent['x_cm'], agent['y_cm'], task['x_reward_cm'], task['y_reward_cm'], data['reward_counter'], sim['ongoingSeq'], sim['reward_found'], sim['netw_index'], DG_place_cells, agent['hasGoal'])    
    DG_place_cells.DA = DG_cells_dopamine
    if sim['stopping']:
        net_move.stop()    
    return

#------------------------------------------------------------------------------------------------------------------------

def reward_placement(i_netw):
    global task
    global maze

    if sim['home_trial']==False: # called during an "away"-Trial
        task['goal_index'] = task['home_index']
        task['x_reward_cm'], task['y_reward_cm'] = net_init.xypos_maze(task['goal_index'], netw['n_grid'], maze['L_cm'])
        print "Network %i: New reward location is Home: (x,y) = %f, %f" %(i_netw, task['x_reward_cm'], task['y_reward_cm'])
    else:
        x_temp_cm, y_temp_cm = maze['edge_cm'] + 0.5*maze['grid_factor'] + randint(0, 5)*maze['grid_factor'], maze['edge_cm'] + 0.5*maze['grid_factor'] + randint(0, 5)*maze['grid_factor']
        task['goal_index'] = net_init.xy_index_maze(x_temp_cm, y_temp_cm, netw['n_grid'], maze['L_cm'])

        counter_rew = 0
        while task['goal_index'] == task['home_index']:
            x_temp_cm, y_temp_cm = maze['edge_cm'] + 0.5*maze['grid_factor'] + randint(0, 5)*maze['grid_factor'], maze['edge_cm'] + 0.5*maze['grid_factor'] + randint(0, 5)*maze['grid_factor']
            task['goal_index'] = net_init.xy_index_maze(x_temp_cm, y_temp_cm, netw['n_grid'], maze['L_cm'])
            counter_rew += 1
            if counter_rew > 100:
                print "Network %i: Stuck in while loop, reward_placement" %(i_netw)
        task['x_reward_cm'], task['y_reward_cm'] = net_init.xypos_maze(task['goal_index'], netw['n_grid'], maze['L_cm'])
        print "Network %i: New random reward location (x,y) = %f, %f" %(i_netw, task['x_reward_cm'], task['y_reward_cm'])






#-------------------------------------------------------------------------------------------------------------    

i_sigma_dist_cm, i_wmax_Place_cells_recur, i_wmax_Place_cells_inh,\
i_wmax_inh_inh, i_wmax_inh_Place_cells, i_pconn_recur, i_tau_membr, i_tau_exc, i_tau_inh, i_dt = range(10)

#-------------------------------------------------------------------------------------------------------------

#netw_params = 	[sigma_dist_cm, wmax_exc_exc, wmax_exc_inh, wmax_inh_inh, wmax_inh_exc, pconn_recur, tau_membr, tau_exc, tau_inh,   dt    ]
netw_params = 	[50.0, 	        2*0.4114*nA, 12*0.8125*pA, 6*6*13.0*pA, 0.9*5*4.1*pA,  1, 10*ms, 6*ms,   2*ms,  0.2*ms] # 
print "netw_params= ", netw_params

# Constants - neuron model and network
defaultclock.dt =  netw_params[i_dt]
tau_membr = netw_params[i_tau_membr]
tau_exc = netw_params[i_tau_exc] 
tau_inh = netw_params[i_tau_inh]
C, gL, EL, tauTrace = net_init.initConstants()

weightmod_DGCA3 = False # True

useMonitors = False # True #      
if useMonitors:
    print "Caution, using place cell monitors!"


global agent
global task
global sim
global data

data['occupancyMap'] = Inf * ones([40, maze['L_cm'] / maze['spatialBinSize_cm'], maze['L_cm'] / maze['spatialBinSize_cm']]) 

global placevalMat
placevalMat = zeros(netw['n_place_cells'])


data['start_index'] = netw['n_grid'] * (netw['n_grid'] - 1) # 45*44 # 45**2 - 1 #0 #44

M_bump_center = SpikeMonitor(Place_cells, function=running_bump_center) # custom monitor

if useMonitors: 
    net_move = Network(poisson_inp_uniform, #context_pop_home, context_pop_away, 
				  #Place_cells, DG_place_cells, \
                  #S_Poi_cont_home, S_Poi_cont_away, S_context_home_DG, S_context_away_DG,\
                  #M_sp_DG, M_sp_Place_cells,\
                  navigateToGoal, goalCheck, reward_delivery, check_placefields) 

    net_seqs = Network(poisson_inp_uniform)#, context_pop_home, context_pop_away, DG_place_cells, 
					   #Place_cells, Inh_neurons, \
                       #S_Poi_cont_home, S_Poi_cont_away, S_context_home_DG, S_context_away_DG, S_DG_Place_cells, S_Place_cells_recur, S_Place_cells_Inh, S_Inh_Place_cells, S_Inh_Inh,\
                       #M_sp_DG, M_sp_Place_cells, M_sp_Inh, 
					   #M_bump_center)

else:
    #net_move = Network(poisson_inp_uniform,\
	#					navigateToGoal, goalCheck, reward_delivery, check_placefields) 
    #net_seqs = Network(poisson_inp_uniform)

	# Don't forget to switch on navigation towards the sequence endpoint below!

    #'''# with simulation
    net_move = Network(poisson_inp_uniform, context_pop_home, context_pop_away, 
				  Place_cells, DG_place_cells, \
                  S_Poi_cont_home, S_Poi_cont_away, S_context_home_DG, S_context_away_DG,\
                  navigateToGoal, goalCheck, reward_delivery, check_placefields) 

    net_seqs = Network(poisson_inp_uniform, context_pop_home, context_pop_away, DG_place_cells, 
					   Place_cells, Inh_neurons, \
                       S_Poi_cont_home, S_Poi_cont_away, S_context_home_DG, S_context_away_DG, S_DG_Place_cells, S_Place_cells_recur, S_Place_cells_Inh, S_Inh_Place_cells, S_Inh_Inh,\
                       M_bump_center)
    #'''




def stim(saving, nTrials, i_seed): # main function for recall / sequence generation  
    global agent
    global task
    global sim
    global data

    global placevalMat
    global home_index_36

    sim['netw_index'] = i_seed
    
    _latency_array = zeros(nTrials)
    _seq_endpoint_final_array = zeros(nTrials)
    _seq_count_array = zeros(nTrials)
    _seq_start_array = zeros([nTrials, 30])
    _seq_endpoint_array = zeros([nTrials, 30])
    _random_nav_time_array = zeros(nTrials)
    _goal_nav_time_array = zeros(nTrials)
    _focal_search_time_array = zeros(nTrials)
    _weight_array_home = zeros([nTrials, netw['n_place_cells']])
    _weight_array_away = zeros([nTrials, netw['n_place_cells']])
    _center_mat_array = -Inf*ones([nTrials, 100, 100])
    _goal_index_array = zeros(nTrials)

    start_time = time()
    netLength = 0

    pyseed(i_seed + int(0.001 * time() + 324823)*i_seed)
    npseed(i_seed + int(0.001 * time() + 324823)*i_seed)  

    i_home, j_home = net_init.xypos(mod(i_seed, 36), 6)
    task['home_index'] = net_init.xy_index_maze(maze['edge_cm'] + 0.5*maze['grid_factor'] + i_home * maze['grid_factor'], maze['edge_cm'] + 0.5*maze['grid_factor'] + j_home * maze['grid_factor'], netw['n_grid'], maze['L_cm'])
    home_index_36 = i_seed # = 6*i_home + j_home
    sim['home_trial'] = True
    task['x_reward_cm'], task['y_reward_cm'] = net_init.xypos_maze(task['home_index'], netw['n_grid'], maze['L_cm'])
    print "Network %i: Home reward location (x,y) = %f, %f" %(i_seed, task['x_reward_cm'], task['y_reward_cm']) 
    task['goal_index'] = task['home_index']
    agent['random_search'] = False


    for _iTrial in xrange(nTrials):
        print "Network %i: Trial %i of %i" %(i_seed, _iTrial+1, nTrials)
        iStim = _iTrial
        sim['trial'] = _iTrial
        _start_loc = rand()
        _goal_index_array[_iTrial] = task['goal_index']    
        agent['visited_index_36'] = zeros(36) # reset for directed random search
        agent['visited_index_6x6'] = zeros([6,6])

        if _iTrial == 0: 
                agent['x_cm'], agent['y_cm'] = set_start_location(_start_loc, maze['edge_cm'], maze['L_cm'], agent['x_cm'], agent['y_cm'], task['x_reward_cm'], task['y_reward_cm'])
                print "Network %i: Start position (x,y) = %i, %i" %(i_seed, agent['x_cm'], agent['y_cm'])
                data['start_index'] = net_init.xy_index_maze(agent['x_cm'], agent['y_cm'], netw['n_grid'], maze['L_cm']) 
        if agent['x_cm'] < maze['edge_cm'] or agent['y_cm'] < maze['edge_cm'] or agent['x_cm'] > maze['L_cm'] - maze['edge_cm'] - 1 or agent['y_cm'] > maze['L_cm'] - maze['edge_cm'] - 1:
            print "Starting outside of maze area!"

        sim['reward_found'] = False
        agent['hasGoal'] = False 
        agent['focal_search'] = False
        sim['ongoingSeq'] = False
        _start_time = defaultclock.t

        if sim['home_trial']:
            placevalMat[:] = S_context_home_DG.w.data[:]
        else:
            placevalMat[:] = S_context_away_DG.w.data[:]

        if sim['home_trial']:
            S_Poi_cont_home.W = 21*mV
            S_Poi_cont_away.W = 0*mV
        else:
            S_Poi_cont_home.W = 0*mV
            S_Poi_cont_away.W = 21*mV

        counter_0 = 0

        while sim['reward_found'] == False and defaultclock.t  < (mvc['time_out']-1*second) : # Global timeout based on all trials 
            counter_0 += 1
            if counter_0 > 1:
                print "Network %i: counter_0 = %i" %(i_seed, counter_0)
                
            sim['value_gain'] = 0
            netLength = defaultclock.t
            clear(False)  
            reinit_default_clock()
            gc.collect()
            reinit_default_clock()
            defaultclock.dt = 0.2*ms # defaultclock.t must be corrected   
            defaultclock.t = netLength

            # Sequence generation for goal-setting:
            sim, data, curr_pos_index, _seq_start_array = prepare_sequence(_iTrial, agent, netw, maze, sim, data, _seq_count_array, _seq_start_array)
            initiate_sequence(curr_pos_index, False, i_seed) #  # True

            _seq_count_array[_iTrial] += 1
            sim['ongoingSeq'] = False	    
            # Sequence has finished - determine the navigation goal as the end point of the sequence:
            agent['xGoal_cm'], agent['yGoal_cm'] = net_init.xypos_maze(data['center_mat'].argmax(), len(data['center_mat'][0,:]), maze['L_cm'])
            #--------------------------
            #agent['xGoal_cm'], agent['yGoal_cm'] = 150, 150 # HACK for NO SIMULATION !!!
            #--------------------------

            data, _seq_endpoint_final_array, _seq_endpoint_array, agent, sim = prepare_navigation(_iTrial, i_seed, agent, netw, maze, sim, data, _seq_endpoint_final_array, _seq_count_array, _seq_start_array, _seq_endpoint_array)

            # Goal-directed navigation simulation:
            netLength = defaultclock.t
            clear(erase=False, all=True)
            gc.collect()
            poisson_inp_uniform.rate = 10*ones(netw['n_place_cells'])

            if sim['value_gain'] > 0:
                print "Network %i: navigation to replay goal..." %(i_seed)
                movement_length = 1.2*1.5 * maze['L_cm'] * sqrt(net_init.quad_dist_grid(curr_pos_index, data['endpoint'] , netw['n_grid'])) / mvc['speed_cm_s'] # error corrected 22.7.14
                _start_time_navigate = defaultclock.t

                net_move.run(movement_length * second) #, report='text')

                _goal_nav_time_array[_iTrial] += (defaultclock.t - _start_time_navigate) / second

            if agent['focal_search']:
                # Search at the nearest four reward wells
                time_focal, x_feeder_cm, y_feeder_cm, nearestfour_36, _start_time_focal = calc_focal_search(mvc['speed_cm_s'], mvc['time_out'], defaultclock, _start_time, maze['edge_cm'], maze['grid_factor'], agent['x_cm'], agent['y_cm'])
                print "Network %i: agent['hasGoal'] = %i, sim['value_gain'] = %f, agent['focal_search'] = %i" %(i_seed, agent['hasGoal'], sim['value_gain'], agent['focal_search'])
                for i_feeder in xrange(4): # Visit the nearest four feeders, the closest one first
                    agent['xGoal_cm'], agent['yGoal_cm'], agent['focal_search'], agent['random_search'], sim['value_gain'], agent['hasGoal'] = do_focal_search(x_feeder_cm, y_feeder_cm, nearestfour_36, i_feeder, i_seed)
                    net_move.run((30.0/mvc['speed_cm_s'])* 4 * second) 
                    if agent['hasGoal'] == False:
                        agent['visited_index_36'][nearestfour_36[i_feeder]] = 1 # Don't search here again in the same trial!
                    if sim['reward_found']:
                        break
                    else: # new 7.12.15: "Prediction error" signal after each visited nearby feeder! ("DAdecr4")
                    #elif i_feeder == 0: # new 7.12.15: "Prediction error" signal only after the first feeder! ("DAdecr1st")
                    	DG_place_cells.DA = -0.5 # -1 

		        net_move.run(100*ms)  # learning should now take place
		        DG_place_cells.DA = 0  

                _focal_search_time_array[_iTrial] += (defaultclock.t - _start_time_focal) / second

            _start_time_navigate = defaultclock.t
            while sim['reward_found']==False: # perform sequential search ...
                agent['random_search'], sim['value_gain'], search_counter, agent['visited_index_36'], agent['search_loc_index_36'], \
                	agent['visited_index_6x6'], visited_index_6x6_potential_subgoals, ix_6, jy_6, index_nearest_36, \
                	delta_x_6, delta_y_6, x_search_goal_cm, y_search_goal_cm, nmax_pot_subgoals, counter, ix_search_index_6, jy_search_index_6, i_subgoal_36 = \
					prepare_sequential_search(agent['x_cm'], agent['y_cm'], maze['edge_cm'], agent['visited_index_36'], sim['home_trial'], home_index_36, maze['grid_factor'], i_seed)
               
                while sim['reward_found']==False and sum(visited_index_6x6_potential_subgoals) < nmax_pot_subgoals and counter < 35-sum(agent['visited_index_36']):
                    counter += 1
                    i_subgoal_36 = set_subgoal(delta_x_6, delta_y_6, ix_6, jy_6, agent['visited_index_36'])
        
                    if agent['visited_index_36'][int(i_subgoal_36)] == False:
                        task['goal_index'], agent['hasGoal'], search_length, agent['xGoal_cm'], agent['yGoal_cm'] = subgoal_coords(i_subgoal_36, maze['edge_cm'], netw['n_grid'], maze['L_cm'], agent['x_cm'], agent['y_cm'], x_search_goal_cm, y_search_goal_cm, mvc['speed_cm_s'], i_seed, agent['visited_index_36'])
                        net_move.run(search_length * second)
                        agent['focal_search'], agent['visited_index_36'], agent['visited_index_6x6'] = subgoal_reached(agent['hasGoal'], agent['visited_index_36'], agent['visited_index_6x6'], i_subgoal_36)
                    else:
                        break # while loop

                    # updating start location information for the next "while" iteration
                    index_nearest_36, delta_x_6, delta_y_6, visited_index_6x6_potential_subgoals, nmax_pot_subgoals = update_startloc(agent['x_cm'], agent['y_cm'], maze['edge_cm'], ix_search_index_6, jy_search_index_6, agent['visited_index_6x6'])

                if sim['reward_found'] == False: # After "subgoal" search - proceed to the "search goal"
					agent['xGoal_cm'], agent['yGoal_cm'], search_length, agent['hasGoal'], task['goal_index'] = set_search_goal(x_search_goal_cm, y_search_goal_cm, netw['n_grid'], maze['L_cm'], agent['x_cm'], agent['y_cm'], mvc['speed_cm_s'])
					print "Network %i: Searching random goal %i at %i, time [s]: %f" %(i_seed, sum(agent['visited_index_36']), agent['search_loc_index_36'], search_length)
					net_move.run(search_length * second)
					agent['focal_search'], agent['visited_index_36'], agent['visited_index_6x6'] = continue_search(agent['hasGoal'], agent['visited_index_36'], agent['search_loc_index_36'], i_subgoal_36, i_seed, agent['visited_index_6x6']) 


            agent['random_search'] = False
            _random_nav_time_array[_iTrial] += (defaultclock.t - _start_time_navigate) / second

            if sim['reward_found']:
                net_move.run(100*ms)  # learning should now take place
                DG_place_cells.DA = 0 # prevent learning from occurring during sequence generation
                # NOW, the reward should be replaced. The activity of the context populations will be switched at the beginning of the next trial

                if sim['home_trial']: 
                    reward_placement(i_seed)
                    if len(nonzero(_goal_index_array[0:_iTrial] == task['goal_index'])[0]) > 0:
                        reward_placement(i_seed) # Prevent repeating Away reward locations
                    sim['home_trial'] = False # next trial will be an Away-trial
                else:
                    reward_placement(i_seed)
                    sim['home_trial'] = True  # next trial will be a Home-trial

            # test:
            if useMonitors and _iTrial == nTrials-1:
                ion()
                subplot(2,1,1)
                raster_plot(M_sp_Place_cells)
                title('Place cells')
                subplot(2,1,2)
                raster_plot(M_sp_DG)
                title('DG cells')
                savefig('raster_plot_'+str(int(time())))


                show()


	#_start_array[_iTrial]    = iStim 
	_latency_array[_iTrial] = (defaultclock.t - _start_time) / second
        _weight_array_home[_iTrial] = S_context_home_DG.w.data
        _weight_array_away[_iTrial] = S_context_away_DG.w.data

        ix,jy = nonzero(data['center_mat_plot']==_iTrial)
        for i_pos in xrange(len(ix)):
            _center_mat_array[_iTrial][ix[i_pos], jy[i_pos]] = data['center_mat'][ix[i_pos], jy[i_pos]]

        print "Network %i: No. of sequences= %i, latency to reach reward= %f " %(i_seed, _seq_count_array[_iTrial], _latency_array[_iTrial])

    print "Network %i: Simulation time: %f" %(i_seed, time() - start_time)


    return _latency_array, _seq_endpoint_final_array, S_context_home_DG.w.data, _seq_count_array, _seq_start_array, _seq_endpoint_array, _random_nav_time_array,\
                _goal_nav_time_array, _focal_search_time_array, data['occupancyMap'], _weight_array_home, _weight_array_away, data['center_mat_plot'], _center_mat_array,\
                _goal_index_array

#------------------------------------------------------------------------------------------------------------------------
def initiate_sequence(start_index, plotting, netw_index):
    global placevalMat
    global sim
    global data

    sim['ongoingSeq'] = True
    start_bias = zeros(netw['n_place_cells'])
    start_bias[range(netw['n_place_cells'])] = net_init.Exp_xyValue2(range(netw['n_place_cells']), netw['n_grid'], 0, start_index, 20*80/63.0) 

    # initiation period
    Place_cells.I_ext = 0 # initial condition

    print "Network %i: Sequence generation... with Poisson rate= 2000 Hz" %(netw_index)

    Place_cells.I_ext = start_bias* 9e-10*amp 

    stim_time_ms = 35
    init_time_ms = 50
    
    net_seqs.run(stim_time_ms *ms) 

    # end of initiation period
    Place_cells.I_ext = 0
    net_seqs.run((init_time_ms - stim_time_ms) *ms)

    print "Network %i: Switching to place-value noise..." %(netw_index)

    poisson_inp_uniform.rate = 200*ones(netw['n_place_cells'])

    if netw_index == 0:
		print "Poisson rate = ", poisson_inp_uniform.rate[0] 
		print "S_Poi_cont_home.W = ", S_Poi_cont_home.W
		print "S_Poi_cont_away.W = ", S_Poi_cont_away.W
		print "S_context_home_DG.w = ", S_context_home_DG.w.data.max()
		print "S_context_away_DG.w = ", S_context_away_DG.w.data.max()
		print "S_DG_Place_cells.w = ", S_DG_Place_cells.w.data.max()
		print "max. S_Place_cells_recur.w= ", S_Place_cells_recur.w.data.max()
		print "sum( S_Place_cells_recur.w )= ", S_Place_cells_recur.w.data.sum()
		print "S_Place_cells_Inh.w= ", S_Place_cells_Inh.w.data.max()
		print "S_Inh_Inh.w= ", S_Inh_Inh.w.data.max()
		print "S_Inh_Place_cells.w= ", S_Inh_Place_cells.w.data.max()


    #print "Performing sequence generation..."
    if netw_index == 0:

        net_seqs.run(0.35*second, report='text')
    else:
        net_seqs.run(0.35*second)

    poisson_inp_uniform.rate = zeros(netw['n_place_cells'])
    
    # Network reset
    Place_cells.I_exc = 0
    data['call_counter'] = 0
    data['n_spikes'] = 0
    data['x_center_sum'], data['y_center_sum'] =  0, 0        
    net_seqs.run(30*ms)
  
    ion()

    if plotting:
        raster_plot(M_sp_DG, M_sp_Place_cells, M_sp_Inh)    

        # test:
        if useMonitors:
            figure()
            raster_plot(M_sp_Place_cells, M_sp_DG)
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
    nTrials = 3 # 1 
    #n_grid = 80 
    N = 10 # 99 # 144 # 

    #identifier = 'recoded_noSim_N'+str(N)+'_'+str(nTrials)+'trials'
    #identifier = 'recoded_withSim_N'+str(N)+'_'+str(nTrials)+'trials'
    identifier = 'recoded_withSim_v3_N'+str(N)+'_'+str(nTrials)+'trials'
    #identifier = 'recoded_noSim_v2_N'+str(N)+'_'+str(nTrials)+'trials'
    dataman = DataManager(identifier)
    n_processes = 10 # 11 # 12

    #stim(False, nTrials, 0)
    #'''#	
    while dataman.itemcount() < N:
        run_tasks(dataman, stim, [(False, nTrials, x) for x in range(dataman.itemcount(), min(N, n_processes + dataman.itemcount()) )], gui=False, poolsize=n_processes) # Creates a number of sub-processes defined by "n_processes"
    #'''

    print "All tasks finished."
















