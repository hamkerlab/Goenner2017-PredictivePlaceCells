# TODO:
# Create projections S_context_home_DG and S_context_away_DG with synaptic plasticity!

from ANNarchy import *
from ann_neuron_model import eqs_if_scaled_new, eqs_if_scaled_new_DG, eqs_if_trace

#from ann_init_vars import netw
from ann_define_params import netw_param, maze

import numpy as np
from scipy.sparse import lil_matrix

Exc_neurons = Population(geometry=(netw_param['n_grid'], netw_param['n_grid']), neuron = eqs_if_scaled_new, name = "Exc_neurons")
Exc_neurons.compute_firing_rate(window=5.0) # New 27.2.19: Perform decoding by accessing Exc_neurons.r
Inh_neurons = Population(netw_param['n_inh'], neuron = eqs_if_scaled_new, name = "Inh_neurons")
Inh_neurons.t_refr = netw_param['t_refr_inh']
DG_neurons = Population(geometry=(netw_param['n_grid'], netw_param['n_grid']), neuron = eqs_if_scaled_new_DG, name = "DG_neurons")
#DG_neurons = Population(geometry=(netw_param['n_grid'], netw_param['n_grid']), neuron = eqs_if_scaled_new, name = "DG_neurons")

poisson_inp = PoissonPopulation(geometry=(netw_param['n_grid'], netw_param['n_grid']), rates=0.1 * 100)
#poisson_inp = PoissonPopulation(geometry=(netw_param['n_grid'], netw_param['n_grid']), rates=200) # TEST

context_pop_home = Population(geometry=(netw_param['n_grid'], netw_param['n_grid']), neuron = eqs_if_trace, name = "context_pop_home") # , stop_condition = "stopping == 1 : any"
context_pop_away = Population(geometry=(netw_param['n_grid'], netw_param['n_grid']), neuron = eqs_if_trace, name = "context_pop_away")
context_pop_home.u_mV = context_pop_home.EL_mV
context_pop_away.u_mV = context_pop_away.EL_mV

simple_syn =Synapse(pre_spike=""" g_target += w """)


max_synapse_exc = netw_param['wmax_exc_exc_pA'] * 1/(np.sqrt(2 * np.pi) * netw_param['sigma_dist_cm'])
min_synapse_exc = 0.0 
sigma_dist_sheet_units =  netw_param['sigma_dist_cm'] * maze['scale_dist'] / maze['L_cm'] # Distances between neurons are normalized to the [0,1] range by ANNarchy

S_Exc_Exc = Projection(
	pre=Exc_neurons, post=Exc_neurons, target='excCA3', synapse=simple_syn
).connect_gaussian( amp = max_synapse_exc, sigma = sigma_dist_sheet_units, limit = min_synapse_exc, allow_self_connections = False, delays = 2.5) # 
# If I want to "exclude" connections (to simulate a "block" in the middle),
# I'll probably have to calculate each neuron's position, for each dendrite!


S_Exc_Inh = Projection(
    pre=Exc_neurons, post=Inh_neurons, target='excCA3', synapse=simple_syn
).connect_all_to_all( weights = netw_param['wmax_exc_inh_pA'], delays = 2.5)

S_Inh_Inh = Projection(
    pre=Inh_neurons, post=Inh_neurons, target='inh', synapse=simple_syn
).connect_all_to_all( weights = netw_param['wmax_inh_inh_pA'], allow_self_connections = False, delays = 2.5)

S_Inh_Exc = Projection(
    pre=Inh_neurons, 
    post=Exc_neurons, 
    target='inh',
    synapse=simple_syn
).connect_all_to_all( weights = netw_param['wmax_inh_exc_pA'], delays = 2.5)

S_Poi_cont_home = Projection(
	pre = poisson_inp, post = context_pop_home, target = 'copy_mV', synapse = simple_syn
).connect_one_to_one( weights = 21.0 )

S_Poi_cont_away = Projection(
	pre = poisson_inp, post = context_pop_away, target = 'copy_mV', synapse = simple_syn
).connect_one_to_one( weights = 21.0 )

# TODO:
# Create projections S_context_home_DG and S_context_away_DG with synaptic plasticity!

plastic_syn_rewardmod = Synapse(
parameters = 	"""
				lrate_pA_per_ms = 50.0  : projection
				tau_trace_ms = 100.0 			: projection
				tau_DA_ms = 10000.0 			: projection 
				""", # learning_rate = (5000 pA)/(100.0 ms)
equations = 	"""
				dspiketrace_pre /dt = - spiketrace_pre / tau_trace_ms	: event-driven, init = 0.0
				dspiketrace_post/dt = - spiketrace_post/ tau_trace_ms	: event-driven, init = 0.0
				dw / dt = if DA>0: 
							     if spiketrace_pre*spiketrace_post - 1e-3*w > 0:
							         lrate_pA_per_ms * (spiketrace_pre*spiketrace_post - 1e-3*w)
							     else:
							         0.0
							 else:
							     if DA<0:
							         -lrate_pA_per_ms * spiketrace_pre*spiketrace_post
							     else:
							         0.0 : event-driven
				dDA/dt = - DA / tau_DA_ms 								: init = 0.0
				""", # Caution: Weight values in pA! -->  w_pA is equal to 1/1000 * w_nA
pre_spike =		"""
				g_target += w 
				spiketrace_pre = 1.0
				""",
post_spike = 	"""
				spiketrace_post = 1.0
				"""
)

'''#
plastic_syn_rewardmod_v2 = Synapse(
parameters = 	"""
				lrate_pA_per_ms = 50.0          : projection
				tau_trace_ms = 100.0 			: projection
				tau_DA_ms = 10000.0 			: projection 
				wmax_pA = 1e3                   : projection
				""", # learning_rate = (5000 pA)/(100.0 ms)
equations = 	"""
				dspiketrace_pre /dt = - spiketrace_pre / tau_trace_ms	: event-driven, init = 0.0
				dspiketrace_post/dt = - spiketrace_post/ tau_trace_ms	: event-driven, init = 0.0
				dDA/dt = - DA / tau_DA_ms 								: init = 0.0
				""", # Caution: Weight values in pA! -->  w_pA is equal to 1/1000 * w_nA
pre_spike =		"""
				g_target += w
				dw = (DA>0) * (spiketrace_pre*spiketrace_post - 1e-3*w > 0) * (lrate_pA_per_ms * (spiketrace_pre*spiketrace_post - 1e-3*w)) - (DA<0) * (lrate_pA_per_ms * spiketrace_pre*spiketrace_post)
				w = clip(w + dw, 0.0, wmax_pA) 
				spiketrace_pre = 1.0
				""",
post_spike = 	"""
				spiketrace_post = 1.0
				dw = (DA>0) * (spiketrace_pre*spiketrace_post - 1e-3*w > 0) * (lrate_pA_per_ms * (spiketrace_pre*spiketrace_post - 1e-3*w)) - (DA<0) * (lrate_pA_per_ms * spiketrace_pre*spiketrace_post)
				w = clip(w + dw, 0.0, wmax_pA) 
				"""
) # If the presynaptic neurons spike at 10Hz, there will be on average only one spike per neuron during the 100msec learning window!!!
'''


plastic_syn_rewardmod_v5 = Synapse(
parameters = 	"""
				lrate_pA_per_ms = 100.0  : projection
				tau_trace_ms = 100.0 			: projection
				tau_DA_ms = 10000.0 			: projection
				wmax_pA = 1e3                   : projection 
				""",
                # lrate_pA_per_ms = 50.0  : projection # First (original) version - very slow learning
                # lrate_pA_per_ms = 500.0  : projection # Second try (v5a) - learning is too fast
                # lrate_pA_per_ms = 200.0  : projection # Third version (v5b) - learning still a bit fast
equations = 	"""
				dspiketrace_pre /dt = - spiketrace_pre / tau_trace_ms	: event-driven, init = 0.0
				dspiketrace_post/dt = - spiketrace_post/ tau_trace_ms	: event-driven, init = 0.0
				dw / dt = if DA>0: 
							     if spiketrace_pre*spiketrace_post - 1e-3*w > 0:
							         lrate_pA_per_ms * (spiketrace_pre*spiketrace_post - 1e-3*w)
							     else:
							         0.0
							 else: 
							     if DA < 0:
							         (-1) * lrate_pA_per_ms * spiketrace_pre*spiketrace_post
							 else:
							     0.0
				dDA/dt = - DA / tau_DA_ms 								: init = 0.0
				""", # Caution: Weight values in pA! -->  w_pA is equal to 1/1000 * w_nA
pre_spike =		"""
				w = clip(w, 0.0, wmax_pA)
				g_target += w 
				spiketrace_pre = 1.0
				""",
post_spike = 	"""
				spiketrace_post = 1.0
				"""
) 
# v5: Learning rate increased, CAUTION- "spillover" of learned Home weights to Away weights!



#equations = 	"""
#				dspiketrace_pre /dt = - spiketrace_pre / tau_trace_ms	: event-driven, init = 0.0
#				dspiketrace_post/dt = - spiketrace_post/ tau_trace_ms	: event-driven, init = 0.0
#				dw / dt = if DA>0: 
#							     (spiketrace_pre*spiketrace_post - 1e-3*w > 0) * lrate_pA_per_ms * (spiketrace_pre*spiketrace_post - 1e-3*w)
#							 else: 
#							     (DA < 0) * (-1) * lrate_pA_per_ms * spiketrace_pre*spiketrace_post
#				dDA/dt = - DA / tau_DA_ms 								: init = 0.0
#				""", # Caution: Weight values in pA! -->  w_pA is equal to 1/1000 * w_nA





# Define weights in units of pA 
# Init value: rand()*0.15 nA = rand()*150pA
lil_w = lil_matrix(np.diag(150 * np.random.rand(netw_param['n_exc'])) ) # [0, 150pA] instead of [0,300 pA] ?! tau_Exc = 6 or 12 ms?
S_context_home_DG = Projection(
	#pre = context_pop_home, post = DG_neurons, target = 'exc', synapse = plastic_syn_rewardmod
	#pre = context_pop_home, post = DG_neurons, target = 'exc', synapse = plastic_syn_rewardmod_v2
	pre = context_pop_home, post = DG_neurons, target = 'exc', synapse = plastic_syn_rewardmod_v5 # Correct sign, learning rate increased
).connect_from_sparse( weights = lil_w )


lil_w = lil_matrix(np.diag(150 * np.random.rand(netw_param['n_exc'])) )
S_context_away_DG = Projection(
	#pre = context_pop_away, post = DG_neurons, target = 'exc', synapse = simple_syn
	pre = context_pop_away, post = DG_neurons, target = 'exc', synapse = plastic_syn_rewardmod_v5
).connect_from_sparse( weights = lil_w )


sigmafan_sheet_units = np.sqrt(0.5 * netw_param['fan_out']) / maze['L_cm'] 
max_synapse_dg_exc = 7.3 * (10000.0 / netw_param['fan_out']) 
min_synapse_dg_exc = 0.1 

S_DG_Exc = Projection(
	pre = DG_neurons, post = Exc_neurons, target = 'excDG', synapse = simple_syn
).connect_gaussian( amp = max_synapse_dg_exc, sigma = sigmafan_sheet_units, limit = min_synapse_dg_exc, delays = 2.0)
















