# TODO:
# Create projections S_context_home_DG and S_context_away_DG with synaptic plasticity!

from ANNarchy import *
from ann_neuron_model import eqs_if_scaled_new, eqs_if_scaled_new_DG, eqs_if_trace

#from ann_init_vars import netw
from ann_define_params import netw_param, maze

import numpy as np
from scipy.sparse import lil_matrix


context_pop_home = Population(geometry=(netw_param['n_grid'], netw_param['n_grid']), neuron = eqs_if_trace, name = "context_pop_home") 
context_pop_away = Population(geometry=(netw_param['n_grid'], netw_param['n_grid']), neuron = eqs_if_trace, name = "context_pop_away")


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




# Define weights in units of pA 
# Init value: rand()*0.15 nA = rand()*150pA
lil_w = lil_matrix(np.diag(150 * np.random.rand(netw_param['n_exc'])) ) # [0, 150pA] instead of [0,300 pA] ?! tau_Exc = 6 or 12 ms?
S_context_home_DG = Projection(
	pre = context_pop_home, post = DG_neurons, target = 'exc', synapse = plastic_syn_rewardmod_v5 # Correct sign, learning rate increased
).connect_from_sparse( weights = lil_w )


lil_w = lil_matrix(np.diag(150 * np.random.rand(netw_param['n_exc'])) )
S_context_away_DG = Projection(
	pre = context_pop_away, post = DG_neurons, target = 'exc', synapse = plastic_syn_rewardmod_v5
).connect_from_sparse( weights = lil_w )
















