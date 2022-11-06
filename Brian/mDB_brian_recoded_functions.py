from brian import *
import mDB_azizi_newInitLearn as net_init
from random import *
from time import time

def rbc_wrapper(spikelist, data, n_grid, L_maze_cm, trial, t_curr):
    # Called by function running_bump_center()
    if spikelist <> []:
        data['call_counter'] += 1
        x_centers, y_centers = net_init.xypos_maze(spikelist, n_grid, L_maze_cm) 
        data['n_spikes'] += len(spikelist)
        data['x_center_sum'] += sum(x_centers)
        data['y_center_sum'] += sum(y_centers)
        data['n_spikes_ringbuffer'][data['call_counter'] % 20] = len(spikelist)
        data['x_center_sum_ringbuffer'][data['call_counter'] % 20] = sum(x_centers)
        data['y_center_sum_ringbuffer'][data['call_counter'] % 20] = sum(y_centers)
    if data['call_counter'] >= 20 and sum(data['x_center_sum_ringbuffer']) > 0 :
        scale = len(data['center_mat'][0,:]) / float(L_maze_cm)
        data['center_mat_plot'][int(scale * sum(data['x_center_sum_ringbuffer']) / sum(data['n_spikes_ringbuffer'])), int(scale * sum(data['y_center_sum_ringbuffer']) / sum(data['n_spikes_ringbuffer']))] = trial
        data['center_mat'][int(scale * sum(data['x_center_sum_ringbuffer']) / sum(data['n_spikes_ringbuffer'])), int(scale * sum(data['y_center_sum_ringbuffer']) / sum(data['n_spikes_ringbuffer']))] = t_curr
        data['n_spikes'] = 0
        data['x_center_sum'], data['y_center_sum'] =  0, 0
    return data

#--------------------------------------------------------------------------------------------------------

def cpf_wrapper(defaultclock, n_place_cells, pf_center_x_cm, pf_center_y_cm, PlaceFieldRadius_cm, agent):
    # Called by function check_placefields()
    x_cm, y_cm = agent['x_cm'], agent['y_cm']
    net_time = defaultclock.t
    defaultclock.dt = 0.2*ms 
    defaultclock.t = net_time # Correction seems to be necessary after accessing defaultclock.dt 
    square_dist_rat_PFcenter_cm = zeros(n_place_cells)
    insidePF = zeros(n_place_cells)
    bell_func = zeros(n_place_cells)
    insidePF_DG = zeros(n_place_cells)    

    square_dist_rat_PFcenter_cm[:] = (x_cm - pf_center_x_cm[:]) ** 2 + (y_cm - pf_center_y_cm[:]) ** 2    
    insidePF[:] = sqrt(square_dist_rat_PFcenter_cm[:]) <= PlaceFieldRadius_cm    
    bell_func[:] = exp(-square_dist_rat_PFcenter_cm[:] / ((PlaceFieldRadius_cm/3.0)**2) )     
    # Gaussian firing rate map inside the place field:
    Place_cells_addcurrent = 3e-8*amp * (rand(n_place_cells) < 0.5*bell_func[:])
    insidePF_DG = sqrt(square_dist_rat_PFcenter_cm[:]) <= 25
    DG_place_cells_addcurrent = 1e-8*amp * bell_func[:] * insidePF_DG[:] 
    return Place_cells_addcurrent, DG_place_cells_addcurrent

#--------------------------------------------------------------------------------------------------------

def nav_to_goal_wrapper(movement_clock, x_cm, y_cm, xGoal_cm, yGoal_cm, hasGoal, focal_search, trial, value_gain, direction, new_direction, ongoingSeq, speed_cm_s, maze_edge_cm, L_maze_cm, spatialBinSize_cm, turning_prob, turn_factor, search_radius_cm, occupancyMap): # , pathRecord_x_cm, pathRecord_y_cm
    #print "trial = ", trial   
    # Called by function navigateToGoal()
    timestep_sec =  movement_clock.dt / second
    #print "nav_to_goal_wrapper: hasGoal, value_gain, focal_search, ongoingSeq = ", hasGoal, value_gain, focal_search, ongoingSeq
    if hasGoal==True and value_gain > 0 and focal_search==False:
	    # navigate towards the goal defined by replay
	    dist = sqrt( (x_cm - xGoal_cm)**2 + (y_cm - yGoal_cm)**2 )
	    if dist > 0:	    
		    new_x_cm = x_cm + timestep_sec * speed_cm_s * (xGoal_cm - x_cm) / dist + 1.5*randn() 
		    new_y_cm = y_cm + timestep_sec * speed_cm_s * (yGoal_cm - y_cm) / dist + 1.5*randn()
	    else:
		    new_x_cm = x_cm + 0.5*randn()
		    new_y_cm = y_cm + 0.5*randn()	       
            new_x_cm = max(maze_edge_cm-1, min(new_x_cm, L_maze_cm-maze_edge_cm-1)) # limit to the accessible part of the environment
            new_y_cm = max(maze_edge_cm-1, min(new_y_cm, L_maze_cm-maze_edge_cm-1))
	    occupancyMap[trial, int(rint(new_x_cm / spatialBinSize_cm)), int(rint(new_y_cm / spatialBinSize_cm))] = defaultclock.t # new 29.8.14
	    x_cm = new_x_cm
	    y_cm = new_y_cm
            #pathRecord_x_cm[int(rint(movement_clock.t / movement_clock.dt))] = x_cm
            #pathRecord_y_cm[int(rint(movement_clock.t / movement_clock.dt))] = y_cm
    elif focal_search:
	    if random() > turning_prob:          # don't change direction in every step
		Delta_direction = turn_factor * 2 * pi * (-1 + 2*random())
		new_direction = mod(direction + Delta_direction, 2*pi)	    
	    new_x_cm = x_cm + timestep_sec * speed_cm_s * cos(new_direction)
	    new_y_cm = y_cm + timestep_sec * speed_cm_s * sin(new_direction)
            while (new_x_cm - xGoal_cm)**2 + (new_y_cm - yGoal_cm)**2 > search_radius_cm**2 or (new_x_cm > L_maze_cm-maze_edge_cm-1 or new_y_cm > L_maze_cm-maze_edge_cm-1 or new_x_cm < maze_edge_cm or new_y_cm < maze_edge_cm):
                new_direction = 2 * pi * (-1 + 2*random())		 
	    new_x_cm = x_cm + timestep_sec * speed_cm_s * cos(new_direction)
	    new_y_cm = y_cm + timestep_sec * speed_cm_s * sin(new_direction)
	    direction = new_direction		
	    occupancyMap[trial, int(rint(new_x_cm / spatialBinSize_cm)), int(rint(new_y_cm / spatialBinSize_cm))] = defaultclock.t # new 29.8.14 
	    x_cm = new_x_cm
	    y_cm = new_y_cm
            #pathRecord_x_cm[int(rint(movement_clock.t / movement_clock.dt))] = x_cm
            #pathRecord_y_cm[int(rint(movement_clock.t / movement_clock.dt))] = y_cm
    elif value_gain <= 0 and ongoingSeq == False:
	    if random() > turning_prob:          # don't change direction in every step
		Delta_direction = turn_factor * 2 * pi * (-1 + 2*random())
		new_direction = mod(direction + Delta_direction, 2*pi)	    
	    new_x_cm = x_cm + timestep_sec * speed_cm_s * cos(new_direction)
	    new_y_cm = y_cm + timestep_sec * speed_cm_s * sin(new_direction)    
	    while min(new_x_cm, new_y_cm) < maze_edge_cm or max(new_x_cm, new_y_cm) > L_maze_cm-maze_edge_cm-1: # navigation in a square maze
                #print "avoiding the wall"
		if new_x_cm < maze_edge_cm or new_x_cm > L_maze_cm-maze_edge_cm-1:
		    new_direction = mod(-1 * (new_direction - pi), 2*pi)
		else:
		    new_direction = mod(-1 * new_direction, 2*pi)
                new_direction = 2 * pi * (-1 + 2*random())
	    	new_x_cm = x_cm + timestep_sec * speed_cm_s * cos(new_direction)
	    	new_y_cm = y_cm + timestep_sec * speed_cm_s * sin(new_direction)
	    direction = new_direction		
	    occupancyMap[trial, int(rint(new_x_cm / spatialBinSize_cm)), int(rint(new_y_cm / spatialBinSize_cm))] = defaultclock.t
	    x_cm = new_x_cm
	    y_cm = new_y_cm
            #pathRecord_x_cm[int(rint(movement_clock.t / movement_clock.dt))] = x_cm
            #pathRecord_y_cm[int(rint(movement_clock.t / movement_clock.dt))] = y_cm
    #print "x_cm, y_cm, new_x_cm, new_y_cm, direction = ", x_cm, y_cm, new_x_cm, new_y_cm, direction

    return x_cm, y_cm, occupancyMap, direction # , pathRecord_x_cm, pathRecord_y_cm

#--------------------------------------------------------------------------------------------------------

def goal_check_wrapper(x_cm, y_cm, xGoal_cm, yGoal_cm, hasGoal, focal_search, value_gain, random_search, netw_index): 
    stopping = False        
    if hasGoal == True and value_gain > 0 and sqrt( (x_cm - xGoal_cm)**2 + (y_cm - yGoal_cm)**2 ) < 3: 
        # Navigation goal is reached
        print "Network %i: Navigation goal reached, Position: (x,y) = %f, %f" %(netw_index, x_cm, y_cm)
        hasGoal = False
        if random_search==False:
            print "Network %i: initiating focal search..." %(netw_index)
            focal_search = True        
        stopping = True
    return hasGoal, focal_search, stopping

#--------------------------------------------------------------------------------------------------------

def reward_delivery_wrapper(x_cm, y_cm, x_reward_cm, y_reward_cm, reward_counter, ongoingSeq, reward_found, netw_index, DG_place_cells, hasGoal):
    stopping = False
    DG_cells_dopamine = 0.0
    
    if ongoingSeq == False:
            quad_dist_reward = (x_cm - x_reward_cm)**2 + (y_cm - y_reward_cm)**2
            if quad_dist_reward < 49 and reward_found==False: # sqrt(...) < 7 - located within 7cm of the reward location
                DG_cells_dopamine = 1
                reward_counter += 1
                print "Network %i: Encountered Reward at time %f" %(netw_index, defaultclock.t)
                reward_found = True
                hasGoal = False
                stopping = True
            else:
                DG_cells_dopamine = 0 # necessary if DA levels are fixed
    return reward_counter, reward_found, hasGoal, stopping, DG_cells_dopamine

#--------------------------------------------------------------------------------------------------------

def set_start_location(_start_loc, maze_edge_cm, L_maze_cm, x_cm, y_cm, x_reward_cm, y_reward_cm):
    init_offset_cm = 10  # Start 10cm "inside" the maze

    if _start_loc < 0.25:
        x_cm, y_cm = maze_edge_cm + init_offset_cm, maze_edge_cm + init_offset_cm                   # Bottom left corner
    elif _start_loc < 0.5:
        x_cm, y_cm = L_maze_cm-1 - maze_edge_cm - init_offset_cm, maze_edge_cm + init_offset_cm     # Bottom right corner
    elif _start_loc < 0.75:
        x_cm, y_cm = maze_edge_cm + init_offset_cm, L_maze_cm-1 - maze_edge_cm - init_offset_cm     # Top left corner
    else:
        x_cm, y_cm = L_maze_cm - maze_edge_cm- 1 - init_offset_cm, L_maze_cm - maze_edge_cm- 1 - init_offset_cm # Top right corner

    while (x_cm - x_reward_cm)**2 + (y_cm - y_reward_cm)**2 < 2500: # Ensure min. distance of 50 cm
        _start_loc = rand()
        if _start_loc < 0.25:
            x_cm, y_cm = maze_edge_cm + init_offset_cm, maze_edge_cm + init_offset_cm               # Bottom left corner
        elif _start_loc < 0.5:
            x_cm, y_cm = L_maze_cm-1 - maze_edge_cm - init_offset_cm, maze_edge_cm + init_offset_cm # Bottom right corner
        elif _start_loc < 0.75:
            x_cm, y_cm = maze_edge_cm + init_offset_cm, L_maze_cm-1 - maze_edge_cm - init_offset_cm # Top left corner
        else:
            x_cm, y_cm = L_maze_cm - maze_edge_cm- 1 - init_offset_cm, L_maze_cm - maze_edge_cm- 1 - init_offset_cm # Top right corner
    return x_cm, y_cm

#--------------------------------------------------------------------------------------------------------

def calc_focal_search(speed_cm_s, time_out, defaultclock, _start_time, maze_edge_cm, grid_factor, x_cm, y_cm):

	time_focal = min( (30.0/speed_cm_s)* 10 * second, time_out - (defaultclock.t - _start_time))
	x_feeder_cm = maze_edge_cm + 0.5*grid_factor + floor(arange(36)/6.0)*grid_factor # This doesn't have to be done every time anew?!
	y_feeder_cm = maze_edge_cm + 0.5*grid_factor +   mod(arange(36),6  )*grid_factor 
	nearestfour_36 = argsort( (x_cm - x_feeder_cm)**2 + (y_cm - y_feeder_cm)**2 )[0:4]
	_start_time_focal = defaultclock.t
	return time_focal, x_feeder_cm, y_feeder_cm, nearestfour_36, _start_time_focal

#--------------------------------------------------------------------------------------------------------

def do_focal_search(x_feeder_cm, y_feeder_cm, nearestfour_36, i_feeder, i_seed):
	xGoal_cm, yGoal_cm = x_feeder_cm[nearestfour_36[i_feeder]], y_feeder_cm[nearestfour_36[i_feeder]] 
	focal_search = False
	random_search = True
	value_gain = 1 # Required for "goal navigation" in the corresponding function
	hasGoal = True # Required for "goal navigation" in the corresponding function
	print "Network %i: Searching nearest feeder %i at %i" %(i_seed, i_feeder, nearestfour_36[i_feeder])

	return xGoal_cm, yGoal_cm, focal_search, random_search, value_gain, hasGoal

#--------------------------------------------------------------------------------------------------------

def prepare_sequential_search(x_cm, y_cm, maze_edge_cm, visited_index_36, home_trial, home_index_36, grid_factor, i_seed):

    random_search = True
    value_gain = 1 # Required for "goal navigation" in the corresponding function
    search_loc_index_36 = randint(0,35)
    x_search_goal_cm, y_search_goal_cm = net_init.xypos_maze(search_loc_index_36, 6, 200)
    x_search_goal_cm += maze_edge_cm
    y_search_goal_cm += maze_edge_cm

    search_counter = 0
    while len( nonzero(nonzero(visited_index_36==1)[0] == search_loc_index_36)[0] ) > 0 or (home_trial==False and search_loc_index_36 == home_index_36):                    
        inz_free = nonzero(visited_index_36==0)[0]
        if len(inz_free) > 1:
            ri = clip(randint(0, len(inz_free)), 0, len(inz_free)-1)
            search_loc_index_36 = inz_free[ri]
        elif len(inz_free) == 1:
            if inz_free[0] == home_index_36:
                print "Network %i: No unvisited locations remaining !!! Restarting random search... " %(i_seed)
                visited_index_36 = zeros(36) 
            else:
                ri = clip(randint(0, len(inz_free)), 0, len(inz_free)-1)
                search_loc_index_36 = inz_free[ri]
        else: 
            print "Network %i: No unvisited locations remaining !!! Restarting random search... "
            visited_index_36 = zeros(36)
        search_counter += 1
        if search_counter > 100:
            print "Network %i: Stuck in while loop, search_counter" %(i_seed)
            print "Network %i: visited_index_36 = %s" %(i_seed, visited_index_36)
        if search_loc_index_36 > 35:
            print "Network %i: search_loc_index_36 exceeds bounds!" %(i_seed)

    x_search_goal_cm = maze_edge_cm + 0.5*grid_factor + floor(search_loc_index_36/6.0)*grid_factor
    y_search_goal_cm = maze_edge_cm + 0.5*grid_factor +   mod(search_loc_index_36, 6) *grid_factor 

    visited_index_6x6 = zeros([6, 6])
    i_subgoal_36 = search_loc_index_36 # FALLBACK case
    i6, j6 = net_init.xypos(range(36), 6)
    if min(i6) < 0 or min(j6)<0 or max(i6)>5 or max(j6)>5:
        print "Network %i: Error: i_6 or j_6 out of range" %(i_seed)
    else:
        for k in xrange(36):
            visited_index_6x6[int(i6[k]), int(j6[k])] = visited_index_36[k] 

    ix_search_index_6, jy_search_index_6 = net_init.xypos(search_loc_index_36, 6) # (x,y) indices at the search goal                
    index_nearest_36 = net_init.xy_index_maze(x_cm - maze_edge_cm, y_cm - maze_edge_cm, 6, 200) 
    ix_6, jy_6 = net_init.xypos(index_nearest_36, 6) # (x,y) indices closest to the current location
    delta_x_6 = int(ix_search_index_6 - ix_6)
    delta_y_6 = int(jy_search_index_6 - jy_6)       
    if ix_6 + (delta_x_6<0)*delta_x_6 < 0 or ix_6 + (delta_x_6>0)*delta_x_6 + 1 > 6:
        print "Network %i: Error: ix_6 + delta_x_6 out of range" %(i_seed)
    if jy_6 + (delta_y_6<0)*delta_y_6 < 0 or jy_6 + (delta_y_6>0)*delta_y_6 + 1 > 6:
        print "Network %i: Error: jy_6 + delta_y_6 out of range" %(i_seed)
    visited_index_6x6_potential_subgoals = visited_index_6x6[int(ix_6) + (delta_x_6<0)*delta_x_6 : int(ix_6) + (delta_x_6>0)*delta_x_6 + 1, int(jy_6) + (delta_y_6<0)*delta_y_6: int(jy_6) + (delta_y_6>0)*delta_y_6 + 1]
    nmax_pot_subgoals = (abs(delta_x_6) + 1) * (abs(delta_y_6) + 1) - 2 # subtracting start and end locations
    counter = 0

    return random_search, value_gain, search_counter, visited_index_36, search_loc_index_36, visited_index_6x6, visited_index_6x6_potential_subgoals, ix_6, jy_6, index_nearest_36, delta_x_6, delta_y_6, x_search_goal_cm, y_search_goal_cm, nmax_pot_subgoals, counter, ix_search_index_6, jy_search_index_6, i_subgoal_36

#--------------------------------------------------------------------------------------------------------

def set_subgoal(delta_x_6, delta_y_6, ix_6, jy_6, visited_index_36):
	for n_dist_subgoal in xrange(1, max(abs(delta_x_6), abs(delta_y_6)) + 1):

		for x_step in xrange(min(n_dist_subgoal, abs(delta_x_6)) + 1):
		    for y_step in xrange(min(n_dist_subgoal, abs(delta_y_6)) + 1):
		        ix_subgoal_6 = ix_6 + sign(delta_x_6) * x_step
		        jy_subgoal_6 = jy_6 + sign(delta_y_6) * y_step 
		        i_subgoal_36 = net_init.xy_index_maze(ix_subgoal_6, jy_subgoal_6, 6, 6)
		        if i_subgoal_36 < 0 or i_subgoal_36 > 35:
		            # Subgoals outside of maze area - cycle to next subgoal
		            print "Network %i: Error: i_subgoal out of range" %(i_seed)
		            print "Network %i: ix_6 = %i, jy_6 = %i, delta_x_6 = %i, delta_y_6 = %i, ix_subgoal_6 = %i, jy_subgoal_6 = %i, i_subgoal_36 = %i" %(i_seed, ix_6, jy_6, delta_x_6, delta_y_6, ix_subgoal_6, jy_subgoal_6, i_subgoal_36)
		        elif min(x_step, y_step)==1 and visited_index_36[int(i_subgoal_36)] == False:
		            #if min(x_step, y_step)==1 and visited_index_36[i_subgoal_36] == False:
		            # A subgoal not visited before has been found - navigate there now!
		            break # for y_step
		    if min(x_step, y_step)==1 and visited_index_36[int(i_subgoal_36)] == False:
		        break # for x_step
		if min(x_step, y_step)==1 and visited_index_36[int(i_subgoal_36)] == False:
		    break # for n_dist_subgoal

	return i_subgoal_36

#--------------------------------------------------------------------------------------------------------

def subgoal_coords(i_subgoal_36, maze_edge_cm, n_grid, L_maze_cm, x_cm, y_cm, x_search_goal_cm, y_search_goal_cm, speed_cm_s, i_seed, visited_index_36):
	x_subgoal_cm, y_subgoal_cm = net_init.xypos_maze(int(i_subgoal_36), 6, 200)
	x_subgoal_cm += maze_edge_cm
	y_subgoal_cm += maze_edge_cm
	goal_index = net_init.xy_index_maze(x_subgoal_cm, y_subgoal_cm, n_grid, L_maze_cm)
	hasGoal = True
	search_length = 1.5 * sqrt( (x_search_goal_cm - x_cm)**2 + (y_search_goal_cm - y_cm)**2 )/ speed_cm_s
	print "Network %i: Searching subgoal %i at %i, time [s]: %f" %(i_seed, sum(visited_index_36), i_subgoal_36, search_length)
	xGoal_cm, yGoal_cm = x_subgoal_cm, y_subgoal_cm # Assigning search goal coordinates used by the navigation function
	return goal_index, hasGoal, search_length, xGoal_cm, yGoal_cm

#--------------------------------------------------------------------------------------------------------

def subgoal_reached(hasGoal, visited_index_36, visited_index_6x6, i_subgoal_36):
	focal_search = False # Ensure that goal-directed navigation continues
	if hasGoal == False:
		visited_index_36[int(i_subgoal_36)] = 1
		i6, j6 = net_init.xypos(i_subgoal_36, 6)
		if min(i6,j6) < 0 or max(i6, j6) > 5:
		    print "Network %i: Error: i_6 or j_6 out of range (2)" %(i_seed)
		else:
		    visited_index_6x6[int(i6), int(j6)] = 1
	return focal_search, visited_index_36, visited_index_6x6

#--------------------------------------------------------------------------------------------------------

def update_startloc(x_cm, y_cm, maze_edge_cm, ix_search_index_6, jy_search_index_6, visited_index_6x6):
	index_nearest_36 = net_init.xy_index_maze(x_cm - maze_edge_cm, y_cm - maze_edge_cm, 6, 200) 
	ix_6, jy_6 = net_init.xypos(index_nearest_36, 6) # (x,y) indices closest to the current location
	delta_x_6 = int(ix_search_index_6 - ix_6)
	delta_y_6 = int(jy_search_index_6 - jy_6)                
	visited_index_6x6_potential_subgoals = visited_index_6x6[int(ix_6) + (delta_x_6<0)*delta_x_6 : int(ix_6) + (delta_x_6>0)*delta_x_6 + 1, int(jy_6) + (delta_y_6<0)*delta_y_6: int(jy_6) + (delta_y_6>0)*delta_y_6 + 1]
	nmax_pot_subgoals = (abs(delta_x_6) + 1) * (abs(delta_y_6) + 1) - 2 # subtracting start and end locations
	return index_nearest_36, delta_x_6, delta_y_6, visited_index_6x6_potential_subgoals, nmax_pot_subgoals

#--------------------------------------------------------------------------------------------------------

def set_search_goal(x_search_goal_cm, y_search_goal_cm, n_grid, L_maze_cm, x_cm, y_cm, speed_cm_s):
	goal_index = net_init.xy_index_maze(x_search_goal_cm, y_search_goal_cm, n_grid, L_maze_cm)
	hasGoal = True
	search_length = 2 * sqrt( (x_search_goal_cm - x_cm)**2 + (y_search_goal_cm - y_cm)**2 )/ speed_cm_s 
	xGoal_cm, yGoal_cm = x_search_goal_cm, y_search_goal_cm # Assigning search goal coordinates used by the navigation function
	return xGoal_cm, yGoal_cm, search_length, hasGoal, goal_index

#--------------------------------------------------------------------------------------------------------

def continue_search(hasGoal, visited_index_36, search_loc_index_36, i_subgoal_36, i_seed, visited_index_6x6):
	focal_search = False # Ensure that goal-directed navigation continues
	if hasGoal == False:
		visited_index_36[search_loc_index_36] = 1
		i6, j6 = net_init.xypos(i_subgoal_36, 6)
		if min(i6, j6) < 0 or max(i6, j6) > 5:
		    print "Network %i: Error: i_6 or j_6 out of range (3)" %(i_seed)
		else:
		    visited_index_6x6[int(i6), int(j6)] = 1
	return focal_search, visited_index_36, visited_index_6x6

#--------------------------------------------------------------------------------------------------------

def prepare_sequence(iTrial, agent, netw, maze, sim, data, seq_count_array, seq_start_array):
    sim['ongoingSeq'] = True
    data['seqCounter'] += 1
    curr_pos_index = net_init.xy_index_maze(agent['x_cm'], agent['y_cm'], netw['n_grid'], maze['L_cm'])
    if seq_count_array[iTrial] < len(seq_start_array[0, :]):
        seq_start_array[iTrial, int(seq_count_array[iTrial])] = curr_pos_index
	return sim, data, curr_pos_index, seq_start_array

#--------------------------------------------------------------------------------------------------------

def prepare_navigation(iTrial, i_seed, agent, netw, maze, sim, data, seq_endpoint_final_array, seq_count_array, seq_start_array, seq_endpoint_array):
	print "Network %i: Navigation goal (x,y) = %f, %f" %(i_seed, agent['xGoal_cm'], agent['yGoal_cm'])
	data['endpoint']  = net_init.xy_index_maze(agent['xGoal_cm'], agent['yGoal_cm'], netw['n_grid'], maze['L_cm'])
	seq_endpoint_final_array[iTrial] = data['endpoint'] 
	if seq_count_array[iTrial] < len(seq_start_array[0, :]):
		seq_endpoint_array[iTrial, int(seq_count_array[iTrial])-1] = data['endpoint'] 
	agent['hasGoal'] = True
	# Difference between sequence start and end place-value:
	sim['value_gain'] = (sqrt((agent['xGoal_cm'] - agent['x_cm'])**2 + (agent['yGoal_cm'] - agent['y_cm'])**2) > 30.0) # sim['value_gain'] = 1 if sequence travels at least 30 cm
	return data, seq_endpoint_final_array, seq_endpoint_array, agent, sim

#--------------------------------------------------------------------------------------------------------

























