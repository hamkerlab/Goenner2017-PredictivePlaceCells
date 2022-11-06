# Strictly speaking, I should separate constant global parameters
# from control variables (hasGoal, focal_search) and result values (x,y,occupancyMap)!!!
# (Actually, this is almost the case)

import numpy as np

#netw_params = 	[50.0,	2*0.4114e3,	12*0.8125, 	6*6*13.0, 	5*4.1,  1, 10, 12,    2,  0.2] # test: tau_inh =2ms, tau-exc=6ms - cool! 
#netw_params = 	[50.0, 	0.8228e3,       9.75, 		468, 		20.5,  	1, 10, 12,    2,  0.2] # orig. params - good

# Constant network parameters
netw_param = {}
netw_param['n_grid'] = 80
netw_param['n_exc'] = int(netw_param['n_grid']**2)
netw_param['n_inh'] = int(0.2*36**2)
netw_param['sigma_dist_cm'] = 50.0
netw_param['wmax_exc_exc_pA'] = 0.8228e3
netw_param['wmax_exc_inh_pA'] = 9.75
netw_param['wmax_inh_inh_pA'] = 468
netw_param['wmax_inh_exc_pA'] = 20.5
netw_param['t_refr_exc'] = 3
netw_param['t_refr_inh'] = 4
netw_param['fan_out'] = 2500.0

# Constant parameters for navigation
maze = {}
maze['edge_cm'] = 110.0
maze['L_cm'] = 200 + 2 * maze['edge_cm']
maze['scale_dist'] = (350.0 / float(netw_param['n_grid'])) / (300.0 / 72.0)
maze['center_len'] = 100
maze['n_edge'] = np.ceil(maze['edge_cm'] / np.float(maze['L_cm']) * maze['center_len'])
maze['turning_prob'] = 0.1
maze['turn_factor'] = 0.1                   # fraction of 2 pi - corresponding to turns in the range turn_factor * [-180 deg, 180 deg]
maze['speed_cm_s'] = 15.0

# Template for a dictionary containing navigation variables - each network instance needs its own copy of this dict!
nav = {}
nav['x_cm'] = 0.0
nav['y_cm'] = 0.0    
nav['xGoal_cm'] = 0.0
nav['yGoal_cm'] = 0.0
nav['hasGoal'] = False
nav['focal_search'] = False
nav['ongoingSeq'] = False
nav['seqCounter'] = 0.0
nav['trial'] = 0
nav['value_gain'] = 0.0
nav['direction'] = 0.0 	 # for random navigation
#nav['new_direction'] = 0.0 # for random navigation
nav['pathRecord_x_cm'] = 0.0
nav['pathRecord_y_cm'] = 0.0

























