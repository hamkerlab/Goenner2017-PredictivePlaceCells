from brian import *
from pylab import *
import mDB_azizi_newInitLearn as net_init

agent = {}
agent['x_cm'] = 0.0
agent['y_cm'] = 0.0
agent['xGoal_cm'] = 0.0
agent['yGoal_cm'] = 0.0
agent['direction'] =  2*pi * random()   # random starting direction
agent['new_direction'] = agent['direction']
agent['search_loc_index_36'] = 0
agent['visited_index_36'] = zeros(36)
agent['visited_index_6x6'] = zeros([6,6])
agent['hasGoal'] = False
agent['focal_search'] = False
agent['random_search'] = False

task = {}
task['x_reward_cm'] = 0.0
task['y_reward_cm'] = 0.0
task['goal_index'] = 0
task['home_index'] = 0


# Simulation control variables
sim = {}
sim['trial'] = 0 # used only in center_mat
sim['value_gain'] = 0
sim['ongoingSeq'] = 0
sim['netw_index'] = -1
sim['stopping'] = 0
sim['reward_found'] = 0
sim['home_trial'] = 0
sim['n_steps'] = 1000


# Analysis variables
data = {}
data['center_mat'] = zeros([100, 100]) 		# Bump movement, separately for each trial
data['center_mat_plot'] = Inf * ones([100, 100]) # Bump movement, collapsed across all trials
data['call_counter'] = 0 					# Used in decoding the bump location
data['n_spikes'] = 0  						# Used in decoding the bump location 
data['n_spikes_ringbuffer'] = zeros(20)  	# ...
data['x_center_sum'] = 0
data['y_center_sum'] = 0
data['x_center_sum_ringbuffer'] = zeros(20)
data['y_center_sum_ringbuffer'] = zeros(20) # Used in decoding the bump location 
data['endpoint'] = 0						# Sequence endpoint (index)
data['occupancyMap'] = []
data['seqCounter'] = 0
data['reward_counter'] = 0
data['start_index'] = 0




# Constants:
# ---------

# Maze parameter constants
maze = {}
maze['edge_cm'] = 110
maze['L_cm'] = 200 + 2 * maze['edge_cm']
maze['grid_factor'] = round( (maze['L_cm'] - 2*maze['edge_cm'])/6.0 )

# Movement parameter constants
mvc = {}
mvc['search_radius_cm'] = 25.0
mvc['time_out'] = 120*60*second 
mvc['DeltaT_step_sec'], mvc['speed_cm_s'], mvc['turning_prob'], mvc['turn_factor'], maze['spatialBinSize_cm'] , maze['spatialBinSize_firingMap_cm'] = net_init.initConstants_movement()

# Network constants
netw = {}
netw['n_grid'] = 80
netw['n_place_cells'] = int(netw['n_grid']**2)
netw['n_inh'] = int(0.2*36**2)
netw['PlaceFieldRadius_cm'] = 90 # Note: Radius is divided by 3.0 !!!
netw['pf_center_x_cm'] = zeros(netw['n_place_cells'])
netw['pf_center_y_cm'] = zeros(netw['n_place_cells'])
netw['pf_center_x_cm'][:], netw['pf_center_y_cm'][:] = net_init.xypos_maze(range(netw['n_place_cells']), netw['n_grid'], maze['L_cm'])    




















