# TODO:
# Add dopamine variable to DG neurons (or to context-to-DG projections)

from ANNarchy import Neuron

eqs_if_scaled_new = Neuron(
	parameters = '''gL_nS = 30 : population
        		C_pF = 300 : population
        		EL_mV = -70.6 : population
		        sigma_mV = 2.0* 22.5 : population
			tau_exc_ms = 6.0 : population
			tau_inh_ms = 2.0 : population
			I_ext_pA = 0.0
			t_refr = 3.0 : population
                        movement_mode = 0.0 : population''', # 
	equations = ''' noise = Normal(0.0, 1.0) 
        		C_pF/gL_nS * du_mV/dt = - (u_mV - EL_mV) + 1/gL_nS * (g_excCA3 + g_excDG - g_inh + I_ext_pA) + sigma_mV * noise : init = -70.6
			dg_excCA3/dt = - g_excCA3 / tau_exc_ms : init = 0.0
			dg_excDG/dt = - g_excDG / tau_exc_ms : init = 0.0
			dg_inh/dt = - g_inh / tau_inh_ms : init = 0.0
                        ext_input = I_ext_pA''', 
	name = 'eqs_if_scaled_new', spike = 'u_mV > (EL_mV + 20)', reset = 'u_mV = EL_mV', refractory = 't_refr') 
		        #sigma_mV = 2.0* 31.62 : population # ORIG - too much noise in CA3!!!

eqs_if_scaled_new_DG = Neuron(
	parameters = '''gL_nS = 30 : population
        		C_pF = 300 : population
        		EL_mV = -70.6 : population
        		tau_exc_ms = 12.0 : population
				DA = 0 : population''',
	equations = ''' C_pF/gL_nS * du_mV/dt = - (u_mV - EL_mV) + 1/gL_nS * (g_exc - g_inh)  : init = -70.6
			dg_exc/dt = - g_exc / tau_exc_ms : init = 0.0
			dg_inh/dt = - g_inh / tau_exc_ms : init = 0.0''', 
	name = 'eqs_if_scaled_new', spike = 'u_mV > (EL_mV + 20)', reset = 'u_mV = EL_mV') # Keine Refraktaerzeit!
# tau_exc_ms = 12.0 : population

eqs_if_trace = Neuron(
	parameters = '''EL_mV = -70.6 : population
        		tau_membr = 10.0 : population
				 stopping = 0 : population''',
        equations = ''' u_sp = g_copy_mV : init = 0.0  
			dg_copy_mV / dt = -g_copy_mv / 1000.0''', 
	name = 'eqs_if_trace', 	spike = 'u_sp > 0', reset = '''g_copy_mV = 0.0''')  # if u_sp > 0 : 




# simple_synapse = ...

# learning_synapse = ...























