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

import mDB_azizi_newInitLearn as net_init

from mDB_brian_recoded_functions import rbc_wrapper, cpf_wrapper, nav_to_goal_wrapper, goal_check_wrapper, reward_delivery_wrapper, set_start_location, calc_focal_search, do_focal_search, prepare_sequential_search, set_subgoal, subgoal_coords, subgoal_reached, update_startloc, set_search_goal, continue_search

set_global_preferences(usecodegenweave=True)

DeltaT_step_sec, speed_cm_s, turning_prob, turn_factor, spatialBinSize_cm, spatialBinSize_firingMap_cm = net_init.initConstants_movement()
movement_clock=Clock(dt=DeltaT_step_sec * second)
bump_clock=Clock(dt=0.004 * second) #  

#--------------------------------------------------------------------------------------------------------
def running_bump_center(spikelist):
    global center_mat
    global center_mat_plot
    global call_counter
    global n_spikes
    global x_center_sum
    global y_center_sum

    global x_center_sum_ringbuffer # new
    global y_center_sum_ringbuffer
    global n_spikes_ringbuffer

    center_mat, center_mat_plot, call_counter, n_spikes, x_center_sum, y_center_sum, x_center_sum_ringbuffer, y_center_sum_ringbuffer, n_spikes_ringbuffer = \
        rbc_wrapper(spikelist, center_mat, center_mat_plot, call_counter, n_spikes, x_center_sum, y_center_sum, x_center_sum_ringbuffer, y_center_sum_ringbuffer, n_spikes_ringbuffer, n_grid, L_maze_cm, trial, defaultclock.t)

    return
#--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------    
@network_operation(movement_clock, when='start')             # to be called according to clock "movement_clock"
def check_placefields():    
    #global x_cm
    #global y_cm
    #global hasGoal
    #global focal_search

    Place_cells_addcurrent, DG_place_cells_addcurrent = cpf_wrapper(defaultclock, n_place_cells, pf_center_x_cm, pf_center_y_cm, PlaceFieldRadius_cm, x_cm, y_cm)    
    Place_cells.I_exc[:] += Place_cells_addcurrent
    DG_place_cells.I_exc[:] += DG_place_cells_addcurrent

    return

#--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
@network_operation(movement_clock) # to be called according to clock "movement_clock"
def navigateToGoal():
    global x_cm
    global y_cm    
    global xGoal_cm
    global yGoal_cm
    global hasGoal
    global focal_search
    #global seqCounter
    global trial
    global value_gain
    global direction 	 # for random navigation
    global new_direction # for random navigation
    global pathRecord_x_cm
    global pathRecord_y_cm
    global occupancyMap

    x_cm, y_cm, occupancyMap, pathRecord_x_cm, pathRecord_y_cm, direction = nav_to_goal_wrapper(movement_clock, x_cm, y_cm, xGoal_cm, yGoal_cm, hasGoal, focal_search, trial, value_gain, direction, new_direction, pathRecord_x_cm, pathRecord_y_cm, ongoingSeq, speed_cm_s, maze_edge_cm, L_maze_cm, spatialBinSize_cm, turning_prob, turn_factor, search_radius_cm, occupancyMap)

    #if new_x_cm == x_cm and new_y_cm == y_cm:
    #    print "Function navigateToGoal: new_x_cm, new_y_cm = ", new_x_cm, new_y_cm

    return
#--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------    
@network_operation(movement_clock) # to be called according to clock "movement_clock"
def goalCheck():
    global x_cm
    global y_cm
    global xGoal_cm
    global yGoal_cm
    global hasGoal
    global focal_search
    #global netLength
    global value_gain
    global random_search
    global netw_index

    hasGoal, focal_search, stopping = goal_check_wrapper(x_cm, y_cm, xGoal_cm, yGoal_cm, hasGoal, focal_search, value_gain, random_search, netw_index)
    if stopping:
        net_move.stop()

    return
#--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------    
@network_operation(movement_clock)              # to be called according to clock "movement_clock"
def reward_delivery():
    global x_cm    
    global y_cm
    global reward_counter
    global ongoingSeq
    global reward_found
    global netw_index
    global hasGoal

    reward_counter, reward_found, hasGoal, stopping, DG_cells_dopamine = reward_delivery_wrapper(x_cm, y_cm, x_reward_cm, y_reward_cm, reward_counter, ongoingSeq, reward_found, netw_index, DG_place_cells, hasGoal)    
    DG_place_cells.DA = DG_cells_dopamine
    if stopping:
        net_move.stop()    
    return
#-------------------------------------------------------------------------------------------------------------    

i_sigma_dist_cm, i_wmax_Place_cells_recur, i_wmax_Place_cells_inh,\
i_wmax_inh_inh, i_wmax_inh_Place_cells, i_pconn_recur, i_tau_membr, i_tau_exc, i_tau_inh, i_dt = range(10)

#-------------------------------------------------------------------------------------------------------------
n_grid = 80 
n_place_cells = int(n_grid**2)
n_inh =  int(0.2*36**2) 

#netw_params = 	[sigma_dist_cm, wmax_exc_exc, wmax_exc_inh, wmax_inh_inh, wmax_inh_exc, pconn_recur, tau_membr, tau_exc, tau_inh,   dt    ]
netw_params = 	[50.0, 	        2*0.4114*nA, 12*0.8125*pA, 6*6*13.0*pA, 0.9*5*4.1*pA,  1, 10*ms, 6*ms,   2*ms,  0.2*ms] # 
print "netw_params= ", netw_params

# Constants - neuron model and network
defaultclock.dt =  netw_params[i_dt]
tau_membr = netw_params[i_tau_membr]
tau_exc = netw_params[i_tau_exc] 
tau_inh = netw_params[i_tau_inh]
C, gL, EL, tauTrace = net_init.initConstants()

tauTrace = 0.1*second 
weightmod_DGCA3 = False # True

# Constants - maze and movement
maze_edge_cm = 110 # for n_grid = 80
L_maze_cm = 200 + 2 * maze_edge_cm

search_radius_cm = 25.0 
nSteps = 1000 
saveWeights = False # True # 
plotFull = True # False #
useMonitors = False # True #      
if useMonitors:
    print "Caution, using place cell monitors!"

time_out = 120*60*second 

nWeightDisplays = 4
display_steps = rint(nSteps / nWeightDisplays)
simlength_sec = nSteps * DeltaT_step_sec
display_sec = display_steps * DeltaT_step_sec
global occupancyMap
occupancyMap = Inf * ones([40, L_maze_cm / spatialBinSize_cm, L_maze_cm / spatialBinSize_cm]) 

global pathRecord_x_cm
global pathRecord_y_cm
pathRecord_x_cm = zeros(nSteps)
pathRecord_y_cm = zeros(nSteps)

global x_reward_cm
global y_reward_cm
global reward_found
global xGoal_cm
global yGoal_cm
global hasGoal
global focal_search
global ongoingSeq
global endpoint
global seqCounter
#global net
global netLength
global trial

global value_gain

global goal_index
global home_index
global home_trial

global call_counter
call_counter = 0
global n_spikes
n_spikes = 0
global x_center_sum
x_center_sum = 0
global y_center_sum
y_center_sum = 0

global x_center_sum_ringbuffer
x_center_sum_ringbuffer = zeros(20)
global y_center_sum_ringbuffer
y_center_sum_ringbuffer = zeros(20)
global n_spikes_ringbuffer
n_spikes_ringbuffer = zeros(20)

global search_loc_index_36
global visited_index_36
visited_index_36 = zeros(36)
global visited_index_6x6
visited_index_6x6 = zeros([6,6])

global random_search
random_search = False
global netw_index
netw_index = -1

trial = 1 # default
hasGoal = False
focal_search = False
value_gain = 0

seqCounter = 0

# Initialize the starting location and direction:
x_cm = 200*random() + maze_edge_cm # random starting value in 0..200 cm
y_cm = 200*random() + maze_edge_cm   
direction = 2 * pi * random()   # random starting direction
new_direction = direction
# init predefined place cell map:
PlaceFieldRadius_cm = 90 # Note: Radius is divided by 3.0 !!!
pf_center_x_cm = zeros(n_place_cells)
pf_center_y_cm = zeros(n_place_cells)
pf_center_x_cm[:], pf_center_y_cm[:] = net_init.xypos_maze(range(n_place_cells), n_grid, L_maze_cm)    

reward_counter = 0


eqs_if_scaled_new, eqs_if_scaled_new_DG, eqs_if_trace = net_init.get_neuron_model()
Place_cells = NeuronGroup(N=n_place_cells, model=eqs_if_scaled_new, threshold='u > (EL + 20*mV)', reset="u = EL", refractory=3*ms, clock = defaultclock) # test for n_grid=80
Inh_neurons = NeuronGroup(N=n_inh, model=eqs_if_scaled_new, threshold='u > (EL + 20*mV)', reset="u = EL", refractory=4*ms, clock = defaultclock) # test for n_grid=80
 
DG_place_cells = NeuronGroup(N=n_place_cells, model=eqs_if_scaled_new_DG, threshold='u > (EL + 20*mV)', reset="u = EL; spiketrace =1", clock = defaultclock)  
poisson_inp_uniform = PoissonGroup(N=n_place_cells, rates=0*Hz, clock = defaultclock)
context_pop_home = NeuronGroup(N=n_place_cells, model=eqs_if_trace, threshold='u > (EL + 20*mV)', reset="u = EL; spiketrace =1", clock = defaultclock) # Standard version - time since last spike
context_pop_away = NeuronGroup(N=n_place_cells, model=eqs_if_trace, threshold='u > (EL + 20*mV)', reset="u = EL; spiketrace =1", clock = defaultclock)

Place_cells.u = EL
Inh_neurons.u = EL
DG_place_cells.u = EL
context_pop_home.u = EL
context_pop_away.u = EL

S_Place_cells_recur, S_Place_cells_Inh, S_Inh_Place_cells, S_Inh_Inh = net_init.get_synapses(Place_cells, Inh_neurons, netw_params, n_grid, L_maze_cm, netw_params[i_sigma_dist_cm])

print "Learning enabled"

S_Poi_cont_home = IdentityConnection(poisson_inp_uniform, context_pop_home, 'u', weight=0*mV)
#S_Poi_cont_home = IdentityConnection(poisson_inp_uniform, context_pop_home, 'u', weight=21*mV) # TEST: constant "HOME" condition!
S_Poi_cont_away = IdentityConnection(poisson_inp_uniform, context_pop_away, 'u', weight=0*mV)

S_context_home_DG = Synapses(context_pop_home, DG_place_cells, model='''dw/dt = 5*nA/(100*ms) * ((spiketrace_pre * spiketrace_post - w/nA > 0) * (spiketrace_pre * spiketrace_post - w/nA) * (DA_post > 0) - spiketrace_pre * spiketrace_post * (DA_post < 0) ) : amp''', 
                                                               pre ='''w = clip(w, 0, Inf)
                                                                       I_exc += w''') # pos. + neg. DA - perfect!



S_context_home_DG[:,:] = 'i==j' 
S_context_home_DG.w = 'rand()*0.15*nA' 

S_context_away_DG =Synapses(context_pop_away, DG_place_cells, model='''dw/dt = 5*nA/(100*ms) * ((spiketrace_pre * spiketrace_post - w/nA > 0) * (spiketrace_pre * spiketrace_post - w/nA) * (DA_post > 0) - spiketrace_pre * spiketrace_post * (DA_post < 0) ) : amp''', 
                                                               pre ='''w = clip(w, 0, Inf)
                                                                       I_exc += w''') # pos. + neg. DA - perfect!

S_context_away_DG[:,:] = 'i==j' 
S_context_away_DG.w = 'rand()*0.15*nA' 

S_DG_Place_cells = Synapses(DG_place_cells, Place_cells, model='''w : 1''', pre ='''I_exc += w''')
fan_out = 2500 
S_DG_Place_cells[:,:] = 'net_init.quad_dist_maze(i, j, n_grid, L_maze_cm) <= fan_out' # connect to CA3 cells within 50cm distance of place field centers

S_DG_Place_cells.w[:,:] = '0.9 * (10000.0 / fan_out) * 10 * 7.3e-13*amp * exp(-net_init.quad_dist_maze(i, j, n_grid, L_maze_cm) / fan_out)' # for tau_inh = 2ms

M_sp_Place_cells = SpikeMonitor(Place_cells)
M_sp_Inh = SpikeMonitor(Inh_neurons)
#SpikeCount_Place_cells = SpikeCounter(Place_cells)
M_sp_DG = SpikeMonitor(DG_place_cells) 
#M_rate_Place_cells=PopulationRateMonitor(Place_cells, bin=10*ms)

global center_mat_plot
global center_mat
center_mat_plot = Inf*ones([100, 100])
center_mat = zeros([100, 100])

M_bump_center = SpikeMonitor(Place_cells, function=running_bump_center) # custom monitor

global placevalMat
placevalMat = zeros(n_place_cells)
start_bias = zeros(n_place_cells)

start_index = n_grid * (n_grid - 1) # 45*44 # 45**2 - 1 #0 #44

if useMonitors: 
    net_move = Network(poisson_inp_uniform, context_pop_home, context_pop_away, Place_cells, DG_place_cells, \
                  S_Poi_cont_home, S_Poi_cont_away, S_context_home_DG, S_context_away_DG,\
                  M_sp_DG, M_sp_Place_cells,\
                  navigateToGoal, goalCheck, reward_delivery, check_placefields) 

    net_seqs = Network(poisson_inp_uniform, context_pop_home, context_pop_away, DG_place_cells, Place_cells, Inh_neurons, \
                       S_Poi_cont_home, S_Poi_cont_away, S_context_home_DG, S_context_away_DG, S_DG_Place_cells, S_Place_cells_recur, S_Place_cells_Inh, S_Inh_Place_cells, S_Inh_Inh,\
                       M_sp_DG, M_sp_Place_cells, M_sp_Inh, M_bump_center) # , check_bump_movement

else:
    net_move = Network(poisson_inp_uniform, context_pop_home, context_pop_away, Place_cells, DG_place_cells, \
                  S_Poi_cont_home, S_Poi_cont_away, S_context_home_DG, S_context_away_DG,\
                  navigateToGoal, goalCheck, reward_delivery, check_placefields) 

    net_seqs = Network(poisson_inp_uniform, context_pop_home, context_pop_away, DG_place_cells, Place_cells, Inh_neurons, \
                       S_Poi_cont_home, S_Poi_cont_away, S_context_home_DG, S_context_away_DG, S_DG_Place_cells, S_Place_cells_recur, S_Place_cells_Inh, S_Inh_Place_cells, S_Inh_Inh,\
                       M_bump_center) # , check_bump_movement





def stim(saving, nTrials, i_seed): # main function for recall / sequence generation  
    global hasGoal
    global focal_search
    global xGoal_cm
    global yGoal_cm
    global x_cm
    global y_cm
    global ongoingSeq
    global endpoint
    global netLength
    global reward_found
    global trial
    global value_gain
    global seqCounter
    global placevalMat
    global pathRecord_x_cm
    global pathRecord_y_cm
    global home_trial
    global start_index

    global search_loc_index_36
    global visited_index_36
    global visited_index_6x6
    global random_search
    global netw_index
    global home_index
    global home_index_36
    global goal_index
    global x_reward_cm
    global y_reward_cm

    visited_index_6x6 = zeros([6,6])

    netw_index = i_seed
    

    _start_array = zeros(nTrials)
    _latency_array = zeros(nTrials)
    _seq_endpoint_final_array = zeros(nTrials)
    _seq_count_array = zeros(nTrials)
    _seq_start_array = zeros([nTrials, 30])
    _seq_endpoint_array = zeros([nTrials, 30])
    _random_nav_time_array = zeros(nTrials)
    _goal_nav_time_array = zeros(nTrials)
    _focal_search_time_array = zeros(nTrials)
    _weight_array_home = zeros([nTrials, n_place_cells])
    _weight_array_away = zeros([nTrials, n_place_cells])
    _center_mat_array = -Inf*ones([nTrials, 100, 100])
    _goal_index_array = zeros(nTrials)

    start_time = time()

    pathRecord_x_cm = zeros( int(nTrials * (time_out/second) / DeltaT_step_sec) )
    pathRecord_y_cm = zeros( int(nTrials * (time_out/second) / DeltaT_step_sec) )

    #pyseed(i_seed + 324823*i_seed)
    #npseed(i_seed + 324823*i_seed)
    pyseed(i_seed + int(0.001 * time() + 324823)*i_seed)
    npseed(i_seed + int(0.001 * time() + 324823)*i_seed)  


    grid_factor = round( (L_maze_cm - 2*maze_edge_cm)/6.0 ) # in cm
    i_home, j_home = net_init.xypos(mod(i_seed, 36), 6)
    home_index = net_init.xy_index_maze(maze_edge_cm + 0.5*grid_factor + i_home * grid_factor, maze_edge_cm + 0.5*grid_factor + j_home * grid_factor, n_grid, L_maze_cm)
    home_index_36 = i_seed # = 6*i_home + j_home
    home_trial = True
    x_reward_cm, y_reward_cm = net_init.xypos_maze(home_index, n_grid, L_maze_cm)
    print "Network %i: Home reward location (x,y) = %f, %f" %(i_seed, x_reward_cm, y_reward_cm) 
    goal_index = home_index
    random_search = False


    for _iTrial in xrange(nTrials):
        print "Network %i: Trial %i of %i" %(i_seed, _iTrial+1, nTrials)
        iStim = _iTrial
        trial = _iTrial
        _start_loc = rand()
        _goal_index_array[_iTrial] = goal_index    
        visited_index_36 = zeros(36) # reset for directed random search
        visited_index_6x6 = zeros([6,6])

        if _iTrial == 0: 
                x_cm, y_cm = set_start_location(_start_loc, maze_edge_cm, L_maze_cm, x_cm, y_cm, x_reward_cm, y_reward_cm)
                print "Network %i: Start position (x,y) = %i, %i" %(i_seed, x_cm, y_cm)
                start_index = net_init.xy_index_maze(x_cm, y_cm, n_grid, L_maze_cm) 
        if x_cm < maze_edge_cm or y_cm < maze_edge_cm or x_cm > L_maze_cm - maze_edge_cm - 1 or y_cm > L_maze_cm - maze_edge_cm - 1:
            print "Starting outside of maze area!"

        reward_found = False
        hasGoal = False 
        focal_search = False
        ongoingSeq = False
        _start_time = defaultclock.t

        if home_trial:
            placevalMat[:] = S_context_home_DG.w.data[:]
        else:
            placevalMat[:] = S_context_away_DG.w.data[:]

        if home_trial:
            S_Poi_cont_home.W = 21*mV
            S_Poi_cont_away.W = 0*mV
        else:
            S_Poi_cont_home.W = 0*mV
            S_Poi_cont_away.W = 21*mV

        counter_0 = 0

        while reward_found == False and defaultclock.t  < (time_out-1*second) : # Global timeout based on all trials 
            counter_0 += 1
            if counter_0 > 1:
                print "Network %i: counter_0 = %i" %(i_seed, counter_0)
                
            value_gain = 0
            netLength = defaultclock.t
            clear(False)  
            reinit_default_clock()
            gc.collect()
            reinit_default_clock()
            defaultclock.dt = 0.2*ms # defaultclock.t must be corrected   
            defaultclock.t = netLength

            # Sequence generation for goal-setting:
            ongoingSeq = True
            seqCounter += 1
            curr_pos_index = net_init.xy_index_maze(x_cm, y_cm, n_grid, L_maze_cm)
            if _seq_count_array[_iTrial] < len(_seq_start_array[0, :]):
                _seq_start_array[_iTrial, int(_seq_count_array[_iTrial])] = curr_pos_index

            initiate_sequence(curr_pos_index, False, i_seed) #  # True

            _seq_count_array[_iTrial] += 1
            ongoingSeq = False	    
            # Sequence has finished - determine the navigation goal as the end point of the sequence:
            xGoal_cm, yGoal_cm = net_init.xypos_maze(center_mat.argmax(), len(center_mat[0,:]), L_maze_cm)
            print "Network %i: Navigation goal (x,y) = %f, %f" %(i_seed, xGoal_cm, yGoal_cm)
            endpoint = net_init.xy_index_maze(xGoal_cm, yGoal_cm, n_grid, L_maze_cm)
            _seq_endpoint_final_array[_iTrial] = endpoint
            if _seq_count_array[_iTrial] < len(_seq_start_array[0, :]):
                _seq_endpoint_array[_iTrial, int(_seq_count_array[_iTrial])-1] = endpoint
            hasGoal = True
            # Difference between sequence start and end place-value:
            value_gain = (sqrt((xGoal_cm - x_cm)**2 + (yGoal_cm - y_cm)**2) > 30.0) # value_gain = 1 if sequence travels at least 30 cm

            # Goal-directed navigation simulation:
            netLength = defaultclock.t
            clear(erase=False, all=True)
            gc.collect()

            poisson_inp_uniform.rate = 10*ones(n_place_cells)

            if value_gain > 0:
                print "Network %i: navigation to replay goal..." %(i_seed)
                movement_length = 1.2*1.5 * L_maze_cm * sqrt(net_init.quad_dist_grid(curr_pos_index, endpoint, n_grid)) / speed_cm_s # error corrected 22.7.14
                _start_time_navigate = defaultclock.t

                net_move.run(movement_length * second) #, report='text')

                _goal_nav_time_array[_iTrial] += (defaultclock.t - _start_time_navigate) / second

            if focal_search:
                # Search at the nearest four reward wells
                time_focal, x_feeder_cm, y_feeder_cm, nearestfour_36, _start_time_focal = calc_focal_search(speed_cm_s, time_out, defaultclock, _start_time, maze_edge_cm, grid_factor, x_cm, y_cm)
                print "Network %i: hasGoal = %i, value_gain = %f, focal_search = %i" %(i_seed, hasGoal, value_gain, focal_search)
                for i_feeder in xrange(4): # Visit the nearest four feeders, the closest one first
                    xGoal_cm, yGoal_cm, focal_search, random_search, value_gain, hasGoal = do_focal_search(x_feeder_cm, y_feeder_cm, nearestfour_36, i_feeder, i_seed)
                    net_move.run((30.0/speed_cm_s)* 4 * second) 
                    if hasGoal == False:
                        visited_index_36[nearestfour_36[i_feeder]] = 1 # Don't search here again in the same trial!
                    if reward_found:
                        break
                    else: # new 7.12.15: "Prediction error" signal after each visited nearby feeder! ("DAdecr4")
                    #elif i_feeder == 0: # new 7.12.15: "Prediction error" signal only after the first feeder! ("DAdecr1st")
                    	DG_place_cells.DA = -0.5 # -1 

		        net_move.run(100*ms)  # learning should now take place
		        DG_place_cells.DA = 0  

                _focal_search_time_array[_iTrial] += (defaultclock.t - _start_time_focal) / second

            _start_time_navigate = defaultclock.t
            while reward_found==False: # perform sequential search ...
                random_search, value_gain, search_counter, visited_index_36, search_loc_index_36, \
                	visited_index_6x6, visited_index_6x6_potential_subgoals, ix_6, jy_6, index_nearest_36, \
                	delta_x_6, delta_y_6, x_search_goal_cm, y_search_goal_cm, nmax_pot_subgoals, counter, ix_search_index_6, jy_search_index_6, i_subgoal_36 = \
					prepare_sequential_search(x_cm, y_cm, maze_edge_cm, visited_index_36, home_trial, home_index_36, grid_factor, i_seed)
               
                while reward_found==False and sum(visited_index_6x6_potential_subgoals) < nmax_pot_subgoals and counter < 35-sum(visited_index_36):
                    counter += 1
                    i_subgoal_36 = set_subgoal(delta_x_6, delta_y_6, ix_6, jy_6, visited_index_36)
        
                    if visited_index_36[int(i_subgoal_36)] == False:
                        #print "Before subgoal_coords: goal_index, hasGoal, search_length, xGoal_cm, yGoal_cm = ", goal_index, hasGoal, search_length, xGoal_cm, yGoal_cm
                        goal_index, hasGoal, search_length, xGoal_cm, yGoal_cm = subgoal_coords(i_subgoal_36, maze_edge_cm, n_grid, L_maze_cm, x_cm, y_cm, x_search_goal_cm, y_search_goal_cm, speed_cm_s, i_seed, visited_index_36)
                        #print "After subgoal_coords: goal_index, hasGoal, search_length, xGoal_cm, yGoal_cm = ", goal_index, hasGoal, search_length, xGoal_cm, yGoal_cm
                        net_move.run(search_length * second)
                        #print "After net_move.run(): x_cm, y_cm = ", x_cm, y_cm
                        focal_search, visited_index_36, visited_index_6x6 = subgoal_reached(hasGoal, visited_index_36, visited_index_6x6, i_subgoal_36)
                        #print "After subgoal_reached: focal_search, visited_index_36, visited_index_6x6 = ", focal_search, visited_index_36, visited_index_6x6
                    else:
                        break # while loop

                    # updating start location information for the next "while" iteration
                    index_nearest_36, delta_x_6, delta_y_6, visited_index_6x6_potential_subgoals, nmax_pot_subgoals = update_startloc(x_cm, y_cm, maze_edge_cm, ix_search_index_6, jy_search_index_6, visited_index_6x6)

                if reward_found == False: # After "subgoal" search - proceed to the "search goal"
					xGoal_cm, yGoal_cm, search_length, hasGoal, goal_index = set_search_goal(x_search_goal_cm, y_search_goal_cm, n_grid, L_maze_cm, x_cm, y_cm, speed_cm_s)
					print "Network %i: Searching random goal %i at %i, time [s]: %f" %(i_seed, sum(visited_index_36), search_loc_index_36, search_length)
					net_move.run(search_length * second)
					focal_search, visited_index_36, visited_index_6x6 = continue_search(hasGoal, visited_index_36, search_loc_index_36, i_subgoal_36, i_seed, visited_index_6x6) 


            random_search = False
            _random_nav_time_array[_iTrial] += (defaultclock.t - _start_time_navigate) / second

            if reward_found:
                net_move.run(100*ms)  # learning should now take place
                DG_place_cells.DA = 0 # prevent learning from occurring during sequence generation
                # NOW, the reward should be replaced. The activity of the context populations will be switched at the beginning of the next trial

                if home_trial: 
                    reward_placement(i_seed)
                    if len(nonzero(_goal_index_array[0:_iTrial] == goal_index)[0]) > 0:
                        reward_placement(i_seed) # Prevent repeating Away reward locations
                    home_trial = False # next trial will be an Away-trial
                else:
                    reward_placement(i_seed)
                    home_trial = True  # next trial will be a Home-trial

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


	_start_array[_iTrial]    = iStim 
	_latency_array[_iTrial] = (defaultclock.t - _start_time) / second
        _weight_array_home[_iTrial] = S_context_home_DG.w.data
        _weight_array_away[_iTrial] = S_context_away_DG.w.data

        ix,jy = nonzero(center_mat_plot==_iTrial)
        for i_pos in xrange(len(ix)):
            _center_mat_array[_iTrial][ix[i_pos], jy[i_pos]] = center_mat[ix[i_pos], jy[i_pos]]

        print "Network %i: No. of sequences= %i, latency to reach reward= %f " %(i_seed, _seq_count_array[_iTrial], _latency_array[_iTrial])

    print "Network %i: Simulation time: %f" %(i_seed, time() - start_time)


    return _latency_array, _seq_endpoint_final_array, S_context_home_DG.w.data, _seq_count_array, _seq_start_array, _seq_endpoint_array, _random_nav_time_array,\
                _goal_nav_time_array, _focal_search_time_array, occupancyMap, _weight_array_home, _weight_array_away, center_mat_plot, _center_mat_array,\
                _goal_index_array

#------------------------------------------------------------------------------------------------------------------------
def initiate_sequence(start_index, plotting, netw_index):
    global placevalMat
    global ongoingSeq
    global trial

    global call_counter
    global n_spikes
    global x_center_sum
    global y_center_sum


    ongoingSeq = True
    start_bias[range(n_place_cells)] = net_init.Exp_xyValue2(range(n_place_cells), n_grid, 0, start_index, 20*80/63.0) 

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

    poisson_inp_uniform.rate = 200*ones(n_place_cells)

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

    poisson_inp_uniform.rate = zeros(n_place_cells)
    
    # Network reset
    Place_cells.I_exc = 0
    call_counter = 0
    n_spikes = 0
    x_center_sum, y_center_sum =  0, 0        
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
        matshow(transpose(center_mat_plot), cmap=cm.YlOrRd, fignum=False)
        colorbar()
        title('Bump center location plotted across time')
        xlabel('x position')
        ylabel('y position')
        ax = gca()
        ax.set_xticks([0, len(center_mat_plot[0,:])])
        ax.set_xticklabels([0, L_maze_cm])
        ax.set_yticks([0, len(center_mat_plot[:,0])])
        ax.set_yticklabels([0, L_maze_cm])     


        ioff()
        show()

#------------------------------------------------------------------------------------------------------------------------

def reward_placement(i_netw):
    global goal_index
    global x_reward_cm
    global y_reward_cm
    global home_index

    if home_trial==False: # called during an "away"-Trial
        goal_index = home_index
        x_reward_cm, y_reward_cm = net_init.xypos_maze(goal_index, n_grid, L_maze_cm)
        print "Network %i: New reward location is Home: (x,y) = %f, %f" %(i_netw, x_reward_cm, y_reward_cm)
    else:
        grid_factor = round( (L_maze_cm - 2*maze_edge_cm)/6.0 )
        x_temp_cm, y_temp_cm = maze_edge_cm + 0.5*grid_factor + randint(0, 5)*grid_factor, maze_edge_cm + 0.5*grid_factor + randint(0, 5)*grid_factor
        goal_index = net_init.xy_index_maze(x_temp_cm, y_temp_cm, n_grid, L_maze_cm)
        x_home_cm, y_home_cm = net_init.xypos_maze(home_index, n_grid, L_maze_cm)          

        counter_rew = 0
        while goal_index == home_index:
            x_temp_cm, y_temp_cm = maze_edge_cm + 0.5*grid_factor + randint(0, 5)*grid_factor, maze_edge_cm + 0.5*grid_factor + randint(0, 5)*grid_factor
            goal_index = net_init.xy_index_maze(x_temp_cm, y_temp_cm, n_grid, L_maze_cm)
            counter_rew += 1
            if counter_rew > 100:
                print "Network %i: Stuck in while loop, reward_placement" %(i_netw)
        x_reward_cm, y_reward_cm = net_init.xypos_maze(goal_index, n_grid, L_maze_cm)
        print "Network %i: New random reward location (x,y) = %f, %f" %(i_netw, x_reward_cm, y_reward_cm)




#------------------------------------------------------------------------------------------------------------------------

if __name__ == '__main__':
    nTrials = 3 # 6 # 16 # 8 ## 40 
    n_grid = 80 
    N = 10 # 99 # 144 # 

    identifier = 'recoded_v1_N'+str(N)+'_'+str(nTrials)+'trials'
    dataman = DataManager(identifier)
    n_processes = 10 # 11 # 12

    #stim(False, nTrials, 0)
    #'''#	
    while dataman.itemcount() < N:
        run_tasks(dataman, stim, [(False, nTrials, x) for x in range(dataman.itemcount(), min(N, n_processes + dataman.itemcount()) )], gui=False, poolsize=n_processes) # Creates a number of sub-processes defined by "n_processes"
    #'''

    print "All tasks finished."
















