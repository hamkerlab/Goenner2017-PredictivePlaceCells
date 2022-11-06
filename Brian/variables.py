# Constants
# ---------

# Network & simulation parameter constants
defaultclock.dt
movement_clock.dt
n_grid
n_place_cells
n_inh
netw_params
tau_membr
tau_exc
tau_inh
C, gL, EL, tauTrace
pf_center_x_cm
pf_center_y_cm
PlaceFieldRadius_cm
fan_out
stim_time_ms
init_time_ms
Learning_rate_IMPLICIT # hard-coded as numbers
w_init_context_DG_IMPLICIT # hard-coded as numbers
low_rate_Poi_IMPLICIT
high_rate_Poi_IMPLICIT


# Analysis parameter constants
useMonitors




# Obsolete / unused / to be removed:
reward_counter
bump_clock
weightmod_DGCA3
saveWeights
plotFull
nWeightDisplays
display_steps
simlength_sec
display_sec
seq_counter
placevalMat
start_bias # only local
netLength  # only local
pathRecord_x_cm
pathRecord_y_cm
nSteps
reward_counter


# Variables
# ---------

# Network objects
poisson_inp_uniform
context_pop_home
context_pop_away
DG_place_cells
Place_cells
Inh_neurons
S_Place_cells_recur
S_Place_cells_Inh
S_Inh_Place_cells
S_Inh_Inh
S_Poi_cont_home
S_Poi_cont_away
S_context_home_DG
S_context_away_DG
S_DG_Place_cells
M_sp_Place_cells
M_sp_Inh
M_sp_DG
M_bump_center


# Network variables
defaultclock.t
movement_clock.t


# Results / output variables; mostly local in function stim()
_latency_array
_seq_endpoint_final_array
S_context_home_DG.w.data # Obsolete / redundant?! See: _weight_array_home (below)
_seq_count_array
_seq_start_array
_seq_endpoint_array
_random_nav_time_array
_goal_nav_time_array
_focal_search_time_array
occupancyMap			# global var.
_weight_array_home
_weight_array_away
center_mat_plot			# global var. - Bump movement, integrating all trials
_center_mat_array
_goal_index_array






# DONE:
# ----

# Analysis variables
center_mat				# Bump movement, separately for each trial
call_counter			# Used in decoding the bump location
call_counter			# Used in decoding the bump location
n_spikes_ringbuffer		# Used in decoding the bump location
x_center_sum			# Used in decoding the bump location
y_center_sum			# Used in decoding the bump location
x_center_sum_ringbuffer	# Used in decoding the bump location
y_center_sum_ringbuffer	# Used in decoding the bump location
n_spikes_ringbuffer		# Used in decoding the bump location
endpoint				# Sequence endpoint

# Analysis constants
spatialBinSize_cm
spatialBinSize_firingMap_cm

# Simulation control variables
trial
hasGoal
focal_search
random_search
value_gain
ongoingSeq
netw_index
stopping
reward_found
time_out
home_trial


# Maze parameter constants
L_maze_cm
maze_edge_cm
grid_factor

# Movement parameter constants
DeltaT_step_sec	# Simulated movement every ... sec (e.g., 0.1 sec)
speed_cm_s		# Simulated moement speed
turning_prob	# Probability of making a turn during goal-directed navigation
turn_factor		# Size of the turn
search_radius_cm

# Movement control variables
x_cm
y_cm
xGoal_cm
yGoal_cm
direction
new_direction
x_reward_cm
y_reward_cm
goal_index
home_index
x_home_cm
y_home_cm
search_loc_index_36
visited_index_36
visited_index_6x6




