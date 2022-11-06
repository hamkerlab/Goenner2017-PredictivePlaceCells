#from brian import *
import ann_InitLearn as net_init
from random import *
from time import time
import numpy as np

def rbc_wrapper(spikelist, data, n_grid, L_maze_cm, trial, t_curr):
    # Called by function running_bump_center()
    if spikelist != []:
        data['call_counter'] += 1
        x_centers, y_centers = net_init.xypos_maze(spikelist, n_grid, L_maze_cm) 
        #print "function rbc_wrapper: xm, ym = %.1f, %.1f, std_x = %.1f" %(np.nanmean(x_centers), np.nanmean(y_centers), np.nanstd(x_centers))
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

def cpf_wrapper(n_place_cells, pf_center_x_cm, pf_center_y_cm, PlaceFieldRadius_cm, agent):
    # Called by function check_placefields()
    x_cm, y_cm = agent['x_cm'], agent['y_cm']
    square_dist_rat_PFcenter_cm = np.zeros(n_place_cells)
    insidePF = np.zeros(n_place_cells)
    bell_func = np.zeros(n_place_cells)
    insidePF_DG = np.zeros(n_place_cells)    

    square_dist_rat_PFcenter_cm[:] = (x_cm - pf_center_x_cm[:]) ** 2 + (y_cm - pf_center_y_cm[:]) ** 2    
    insidePF[:] = np.sqrt(square_dist_rat_PFcenter_cm[:]) <= PlaceFieldRadius_cm    
    bell_func[:] = np.exp(-square_dist_rat_PFcenter_cm[:] / ((PlaceFieldRadius_cm/3.0)**2) )     
    # Gaussian firing rate map inside the place field:
    Place_cells_addcurrent = 3e4 * (np.random.rand(n_place_cells) < 0.5*bell_func[:]) # 3e-8*amp = 30*nA (Brian) = 30*1e3*pA (ANNarchy)
    insidePF_DG = np.sqrt(square_dist_rat_PFcenter_cm[:]) <= 25
    DG_place_cells_addcurrent = 1e4 * bell_func[:] * insidePF_DG[:]  # 1e-8*amp = 10*nA (Brian) = 10*1e3*pA (ANNarchy)
    return Place_cells_addcurrent, DG_place_cells_addcurrent

#--------------------------------------------------------------------------------------------------------

def nav_to_goal_wrapper(mvc, agent, sim, maze, data, t_curr_ms, movement_clock_dt): 
    # Called by function navigateToGoal()
    speed_cm_s = mvc['speed_cm_s']
    turning_prob = mvc['turning_prob'] 
    turn_factor = mvc['turn_factor']
    search_radius_cm = mvc['search_radius_cm'] 
    x_cm, y_cm = agent['x_cm'], agent['y_cm']
    xGoal_cm, yGoal_cm = agent['xGoal_cm'], agent['yGoal_cm']
    hasGoal = agent['hasGoal']
    focal_search = agent['focal_search']
    direction = agent['direction']
    new_direction = agent['new_direction']
    trial = sim['trial']
    value_gain = sim['value_gain'] 
    ongoingSeq = sim['ongoingSeq'] 
    maze_edge_cm = maze['edge_cm']
    L_maze_cm = maze['L_cm']
    spatialBinSize_cm = maze['spatialBinSize_cm']
    occupancyMap = data['occupancyMap']

    timestep_sec =  movement_clock_dt
    #print "nav_to_goal_wrapper: x_cm, y_cm, hasGoal, value_gain, focal_search, ongoingSeq = ", agent['x_cm'], agent['y_cm'], hasGoal, value_gain, focal_search, ongoingSeq
    #print "nav_to_goal_wrapper: x_cm, y_cm = ", agent['x_cm'], agent['y_cm']

    if hasGoal==True and value_gain > 0 and focal_search==False:
        # navigate towards the goal defined by replay
        dist = np.sqrt( (x_cm - xGoal_cm)**2 + (y_cm - yGoal_cm)**2 )
        #print "dist, timestep_sec, speed_cm_s, xGoal_cm, yGoal_cm = ", dist, timestep_sec, speed_cm_s, xGoal_cm, yGoal_cm
        #print "x_cm, y_cm, dist, timestep_sec, speed_cm_s, xGoal_cm, yGoal_cm = ", x_cm, y_cm, dist, timestep_sec, speed_cm_s, xGoal_cm, yGoal_cm
        if dist > 0:	    
            new_x_cm = x_cm + timestep_sec * speed_cm_s * (xGoal_cm - x_cm) / dist + 1.5*np.random.randn() 
            new_y_cm = y_cm + timestep_sec * speed_cm_s * (yGoal_cm - y_cm) / dist + 1.5*np.random.randn()
            #print "new_x_cm, new_y_cm (a) = ", new_x_cm, new_y_cm
        else:
            new_x_cm = x_cm + 0.5*np.random.randn()
            new_y_cm = y_cm + 0.5*np.random.randn()	       
            #print "new_x_cm, new_y_cm (b) = ", new_x_cm, new_y_cm
            new_x_cm = max(maze_edge_cm-1, min(new_x_cm, L_maze_cm-maze_edge_cm-1)) # limit to the accessible part of the environment
            new_y_cm = max(maze_edge_cm-1, min(new_y_cm, L_maze_cm-maze_edge_cm-1))
        occupancyMap[trial, int(np.rint(new_x_cm / spatialBinSize_cm)), int(np.rint(new_y_cm / spatialBinSize_cm))] = t_curr_ms # new 29.8.14
        x_cm = new_x_cm
        y_cm = new_y_cm
        #print "x_cm, y_cm (1) = ", x_cm, y_cm
    elif focal_search:
        print("nav_to_goal_wrapper, focal_search")
        if random() > turning_prob:          # don't change direction in every step
            Delta_direction = turn_factor * 2 * np.pi * (-1 + 2*random())
            new_direction = np.mod(direction + Delta_direction, 2*np.pi)	    
        new_x_cm = x_cm + timestep_sec * speed_cm_s * np.cos(new_direction)
        new_y_cm = y_cm + timestep_sec * speed_cm_s * np.sin(new_direction)
        while (new_x_cm - xGoal_cm)**2 + (new_y_cm - yGoal_cm)**2 > search_radius_cm**2 or (new_x_cm > L_maze_cm-maze_edge_cm-1 or new_y_cm > L_maze_cm-maze_edge_cm-1 or new_x_cm < maze_edge_cm or new_y_cm < maze_edge_cm):
                if np.mod(t_curr_ms, 100) == 0:
                    print(("nav_to_goal_wrapper, focal_search, while loop: new_x_cm, xGoal_cm, new_y_cm, yGoal_cm = ", new_x_cm, xGoal_cm, new_y_cm, yGoal_cm))
                new_direction = 2 * np.pi * (-1 + 2*random())		 
                new_x_cm = x_cm + timestep_sec * speed_cm_s * np.cos(new_direction)
                new_y_cm = y_cm + timestep_sec * speed_cm_s * np.sin(new_direction)
        direction = new_direction		
        occupancyMap[trial, int(np.rint(new_x_cm / spatialBinSize_cm)), int(np.rint(new_y_cm / spatialBinSize_cm))] = t_curr_ms # new 29.8.14 
        x_cm = new_x_cm
        y_cm = new_y_cm
    elif value_gain <= 0 and ongoingSeq == False:
        if random() > turning_prob:          # don't change direction in every step
            Delta_direction = turn_factor * 2 * np.pi * (-1 + 2*random())
            new_direction = np.mod(direction + Delta_direction, 2*np.pi)	    
            new_x_cm = x_cm + timestep_sec * speed_cm_s * np.cos(new_direction)
            new_y_cm = y_cm + timestep_sec * speed_cm_s * np.sin(new_direction)    
            while min(new_x_cm, new_y_cm) < maze_edge_cm or max(new_x_cm, new_y_cm) > L_maze_cm-maze_edge_cm-1: # navigation in a square maze
                #print("avoiding the wall")
                if new_x_cm < maze_edge_cm or new_x_cm > L_maze_cm-maze_edge_cm-1:
                    new_direction = np.mod(-1 * (new_direction - np.pi), 2*np.pi)
                else:
                    new_direction = np.mod(-1 * new_direction, 2*np.pi)
                    new_direction = 2 * np.pi * (-1 + 2*random())
            new_x_cm = x_cm + timestep_sec * speed_cm_s * np.cos(new_direction)
            new_y_cm = y_cm + timestep_sec * speed_cm_s * np.sin(new_direction)
            direction = new_direction		
            occupancyMap[trial, int(np.rint(new_x_cm / spatialBinSize_cm)), int(np.rint(new_y_cm / spatialBinSize_cm))] = t_curr_ms
            x_cm = new_x_cm
            y_cm = new_y_cm
    return x_cm, y_cm, occupancyMap, direction # , pathRecord_x_cm, pathRecord_y_cm

#--------------------------------------------------------------------------------------------------------

#def goal_check_wrapper(x_cm, y_cm, xGoal_cm, yGoal_cm, hasGoal, focal_search, value_gain, random_search, netw_index, t_curr_ms):
def goal_check_wrapper(agent, sim, t_curr_ms):  
    x_cm = agent['x_cm']
    y_cm = agent['y_cm']
    xGoal_cm = agent['xGoal_cm']
    yGoal_cm = agent['yGoal_cm']
    hasGoal = agent['hasGoal']
    focal_search = agent['focal_search']
    value_gain = sim['value_gain']
    random_search = agent['random_search']
    netw_index = sim['netw_index']
    #stopping = sim['stopping']
    sim_out = sim

    #sim['stopping'] = False # redundant?
    if hasGoal == True and value_gain > 0 and np.sqrt( (x_cm - xGoal_cm)**2 + (y_cm - yGoal_cm)**2 ) < 3: 
        # Navigation goal is reached
        print(("Network %i: Navigation goal reached, Position: (x,y) = %f, %f" %(netw_index, x_cm, y_cm)))
        hasGoal = False
        if random_search==False:
            print(("Network %i: initiating focal search..." %(netw_index)))
            print(("Network %i: Time [ms] = %f" %(netw_index, t_curr_ms)))
            focal_search = True
        sim_out['stopping'] = True
    return hasGoal, focal_search, sim_out['stopping']
    #return agent, sim
    #return agent, sim_out

#--------------------------------------------------------------------------------------------------------

def reward_delivery_wrapper(agent, task, data, sim, t_curr_ms):
    reward_counter = data['reward_counter']
    reward_found = sim['reward_found']
    hasGoal = agent['hasGoal']

    #stopping = False # ???
    #stopping = sim['stopping'] # ???
    DG_cells_dopamine = 0.0
    
    if sim['ongoingSeq'] == False:
            quad_dist_reward = (agent['x_cm'] - task['x_reward_cm'])**2 + (agent['y_cm'] - task['y_reward_cm'])**2
            if quad_dist_reward < 49 and sim['reward_found']==False: # sqrt(...) < 7 - located within 7cm of the reward location
                DG_cells_dopamine = 1
                reward_counter += 1
                print(("Network %i: Encountered Reward at time %f sec, position (x,y) = %f,%f" %(sim['netw_index'], 1e-3*t_curr_ms, agent['x_cm'], agent['y_cm'])))
                reward_found = True
                hasGoal = False
                #stopping = True
                sim['stopping'] = True
            else:
                DG_cells_dopamine = 0 # necessary if DA levels are fixed
    #return reward_counter, reward_found, hasGoal, stopping, DG_cells_dopamine
    return reward_counter, reward_found, hasGoal, sim['stopping'], DG_cells_dopamine

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
        _start_loc = np.random.rand()
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

def calc_focal_search_ann(speed_cm_s, time_out, t_curr_ms, _start_time_ms, maze_edge_cm, grid_factor, x_cm, y_cm):

    time_focal_sec = min( (30.0/speed_cm_s)* 10, time_out - (t_curr_ms - _start_time_ms)) 
    x_feeder_cm = maze_edge_cm + 0.5*grid_factor + np.floor(np.arange(36)/6.0)*grid_factor # This doesn't have to be done every time anew?!
    y_feeder_cm = maze_edge_cm + 0.5*grid_factor +   np.mod(np.arange(36),6)*grid_factor 
    nearestfour_36 = np.argsort( (x_cm - x_feeder_cm)**2 + (y_cm - y_feeder_cm)**2 )[0:4]
    _start_time_focal = t_curr_ms
    return time_focal_sec, x_feeder_cm, y_feeder_cm, nearestfour_36, _start_time_focal

#--------------------------------------------------------------------------------------------------------

def do_focal_search(x_feeder_cm, y_feeder_cm, nearestfour_36, i_feeder, i_seed):
    xGoal_cm, yGoal_cm = x_feeder_cm[nearestfour_36[i_feeder]], y_feeder_cm[nearestfour_36[i_feeder]] 
    focal_search = False
    random_search = True
    value_gain = 1 # Required for "goal navigation" in the corresponding function
    hasGoal = True # Required for "goal navigation" in the corresponding function
    print(("Network %i: Searching nearest feeder %i at %i" %(i_seed, i_feeder, nearestfour_36[i_feeder])))

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
    while len( np.nonzero(np.nonzero(visited_index_36==1)[0] == search_loc_index_36)[0] ) > 0 or (home_trial==False and search_loc_index_36 == home_index_36):                    
        inz_free = np.nonzero(visited_index_36==0)[0]
        if len(inz_free) > 1:
            ri = np.clip(randint(0, len(inz_free)), 0, len(inz_free)-1)
            search_loc_index_36 = inz_free[ri]
        elif len(inz_free) == 1:
            if inz_free[0] == home_index_36:
                print(("Network %i: No unvisited locations remaining !!! Restarting random search... " %(i_seed)))
                visited_index_36 = np.zeros(36) 
            else:
                ri = np.clip(randint(0, len(inz_free)), 0, len(inz_free)-1)
                search_loc_index_36 = inz_free[ri]
        else: 
            print("Network %i: No unvisited locations remaining !!! Restarting random search... ")
            visited_index_36 = np.zeros(36)
        search_counter += 1
        if search_counter > 100:
            print(("Network %i: Stuck in while loop, search_counter" %(i_seed)))
            print(("Network %i: visited_index_36 = %s" %(i_seed, visited_index_36)))
        if search_loc_index_36 > 35:
            print(("Network %i: search_loc_index_36 exceeds bounds!" %(i_seed)))

    x_search_goal_cm = maze_edge_cm + 0.5*grid_factor + np.floor(search_loc_index_36/6.0)*grid_factor
    y_search_goal_cm = maze_edge_cm + 0.5*grid_factor +   np.mod(search_loc_index_36, 6) *grid_factor 

    visited_index_6x6 = np.zeros([6, 6])
    i_subgoal_36 = search_loc_index_36 # FALLBACK case
    i6, j6 = net_init.xypos(list(range(36)), 6)
    if min(i6) < 0 or min(j6)<0 or max(i6)>5 or max(j6)>5:
        print(("Network %i: Error: i_6 or j_6 out of range" %(i_seed)))
    else:
        for k in range(36):
            visited_index_6x6[int(i6[k]), int(j6[k])] = visited_index_36[k] 

    ix_search_index_6, jy_search_index_6 = net_init.xypos(search_loc_index_36, 6) # (x,y) indices at the search goal                
    index_nearest_36 = net_init.xy_index_maze(x_cm - maze_edge_cm, y_cm - maze_edge_cm, 6, 200) 
    ix_6, jy_6 = net_init.xypos(index_nearest_36, 6) # (x,y) indices closest to the current location
    delta_x_6 = int(ix_search_index_6 - ix_6)
    delta_y_6 = int(jy_search_index_6 - jy_6)       
    if ix_6 + (delta_x_6<0)*delta_x_6 < 0 or ix_6 + (delta_x_6>0)*delta_x_6 + 1 > 6:
        print(("Network %i: Error: ix_6 + delta_x_6 out of range" %(i_seed)))
    if jy_6 + (delta_y_6<0)*delta_y_6 < 0 or jy_6 + (delta_y_6>0)*delta_y_6 + 1 > 6:
        print(("Network %i: Error: jy_6 + delta_y_6 out of range" %(i_seed)))
    visited_index_6x6_potential_subgoals = visited_index_6x6[int(ix_6) + (delta_x_6<0)*delta_x_6 : int(ix_6) + (delta_x_6>0)*delta_x_6 + 1, int(jy_6) + (delta_y_6<0)*delta_y_6: int(jy_6) + (delta_y_6>0)*delta_y_6 + 1]
    nmax_pot_subgoals = (abs(delta_x_6) + 1) * (abs(delta_y_6) + 1) - 2 # subtracting start and end locations
    counter = 0

    return random_search, value_gain, search_counter, visited_index_36, search_loc_index_36, visited_index_6x6, visited_index_6x6_potential_subgoals, ix_6, jy_6, index_nearest_36, delta_x_6, delta_y_6, x_search_goal_cm, y_search_goal_cm, nmax_pot_subgoals, counter, ix_search_index_6, jy_search_index_6, i_subgoal_36

#--------------------------------------------------------------------------------------------------------

def set_subgoal(delta_x_6, delta_y_6, ix_6, jy_6, visited_index_36, i_seed):
    for n_dist_subgoal in range(1, max(abs(delta_x_6), abs(delta_y_6)) + 1):

        for x_step in range(min(n_dist_subgoal, abs(delta_x_6)) + 1):
            for y_step in range(min(n_dist_subgoal, abs(delta_y_6)) + 1):
                ix_subgoal_6 = ix_6 + np.sign(delta_x_6) * x_step
                jy_subgoal_6 = jy_6 + np.sign(delta_y_6) * y_step 
                i_subgoal_36 = net_init.xy_index_maze(ix_subgoal_6, jy_subgoal_6, 6, 6)
                if i_subgoal_36 < 0 or i_subgoal_36 > 35:
                    # Subgoals outside of maze area - cycle to next subgoal
                    print(("Network %i: Error: i_subgoal out of range" %(i_seed)))
                    print(("Network %i: ix_6 = %i, jy_6 = %i, delta_x_6 = %i, delta_y_6 = %i, ix_subgoal_6 = %i, jy_subgoal_6 = %i, i_subgoal_36 = %i" %(i_seed, ix_6, jy_6, delta_x_6, delta_y_6, ix_subgoal_6, jy_subgoal_6, i_subgoal_36)))
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
    search_length = 1.5 * np.sqrt( (x_search_goal_cm - x_cm)**2 + (y_search_goal_cm - y_cm)**2 )/ speed_cm_s
    print(("Network %i: Searching subgoal %i at %i, time [s]: %f" %(i_seed, sum(visited_index_36), i_subgoal_36, search_length)))
    xGoal_cm, yGoal_cm = x_subgoal_cm, y_subgoal_cm # Assigning search goal coordinates used by the navigation function
    return goal_index, hasGoal, search_length, xGoal_cm, yGoal_cm

#--------------------------------------------------------------------------------------------------------

def subgoal_reached(hasGoal, visited_index_36, visited_index_6x6, i_subgoal_36, i_seed):
    focal_search = False # Ensure that goal-directed navigation continues
    if hasGoal == False:
        visited_index_36[int(i_subgoal_36)] = 1
        i6, j6 = net_init.xypos(i_subgoal_36, 6)
        if min(i6,j6) < 0 or max(i6, j6) > 5:
            print(("Network %i: Error: i_6 or j_6 out of range (2)" %(i_seed)))
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
    search_length = 2 * np.sqrt( (x_search_goal_cm - x_cm)**2 + (y_search_goal_cm - y_cm)**2 )/ speed_cm_s 
    xGoal_cm, yGoal_cm = x_search_goal_cm, y_search_goal_cm # Assigning search goal coordinates used by the navigation function
    return xGoal_cm, yGoal_cm, search_length, hasGoal, goal_index

#--------------------------------------------------------------------------------------------------------

def continue_search(hasGoal, visited_index_36, search_loc_index_36, i_subgoal_36, i_seed, visited_index_6x6):
    focal_search = False # Ensure that goal-directed navigation continues
    if hasGoal == False:
        visited_index_36[search_loc_index_36] = 1
        i6, j6 = net_init.xypos(i_subgoal_36, 6)
        if min(i6, j6) < 0 or max(i6, j6) > 5:
            print(("Network %i: Error: i_6 or j_6 out of range (3)" %(i_seed)))
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
    print(("Network %i: Navigation goal (x,y) = %f, %f" %(i_seed, agent['xGoal_cm'], agent['yGoal_cm'])))
    data['endpoint']  = net_init.xy_index_maze(agent['xGoal_cm'], agent['yGoal_cm'], netw['n_grid'], maze['L_cm'])
    seq_endpoint_final_array[iTrial] = data['endpoint'] 
    if seq_count_array[iTrial] < len(seq_start_array[0, :]):
        seq_endpoint_array[iTrial, int(seq_count_array[iTrial])-1] = data['endpoint'] 
    agent['hasGoal'] = True
    # Difference between sequence start and end place-value:
    sim['value_gain'] = (np.sqrt((agent['xGoal_cm'] - agent['x_cm'])**2 + (agent['yGoal_cm'] - agent['y_cm'])**2) > 30.0) # sim['value_gain'] = 1 if sequence travels at least 30 cm
    return data, seq_endpoint_final_array, seq_endpoint_array, agent, sim

#--------------------------------------------------------------------------------------------------------

def reward_placement(i_netw, task, sim, maze, netw):
    #global task
    #global maze

    if sim['home_trial']==False: # called during an "away"-Trial
        task['goal_index'] = task['home_index']
        task['x_reward_cm'], task['y_reward_cm'] = net_init.xypos_maze(task['goal_index'], netw['n_grid'], maze['L_cm'])
        print(("Network %i: New reward location is Home: (x,y) = %f, %f, goal index= %i, home index= %i " %(i_netw, task['x_reward_cm'], task['y_reward_cm'], task['goal_index'], task['home_index'])))
    else:
        x_temp_cm, y_temp_cm = maze['edge_cm'] + 0.5*maze['grid_factor'] + randint(0, 5)*maze['grid_factor'], maze['edge_cm'] + 0.5*maze['grid_factor'] + randint(0, 5)*maze['grid_factor']
        task['goal_index'] = net_init.xy_index_maze(x_temp_cm, y_temp_cm, netw['n_grid'], maze['L_cm'])

        counter_rew = 0
        while task['goal_index'] == task['home_index']:
            x_temp_cm, y_temp_cm = maze['edge_cm'] + 0.5*maze['grid_factor'] + randint(0, 5)*maze['grid_factor'], maze['edge_cm'] + 0.5*maze['grid_factor'] + randint(0, 5)*maze['grid_factor']
            task['goal_index'] = net_init.xy_index_maze(x_temp_cm, y_temp_cm, netw['n_grid'], maze['L_cm'])
            counter_rew += 1
            if counter_rew > 100:
                print(("Function reward_placement: i_netw = ", i_netw))
                print(("Network %i: Stuck in while loop, reward_placement" %(i_netw)))
        task['x_reward_cm'], task['y_reward_cm'] = net_init.xypos_maze(task['goal_index'], netw['n_grid'], maze['L_cm'])
        print(("Network %i: New random reward location (x,y) = %f, %f, goal index= %i" %(i_netw, task['x_reward_cm'], task['y_reward_cm'], task['goal_index'])))

    print(("Network %i: end of function reward_placement" %(i_netw)))
    print(("task = ", task))

    return task
























