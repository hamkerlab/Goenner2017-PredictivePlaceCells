from brian import *
from pylab import *
import mDB_azizi_newInitLearn as net_init
from init_vars import netw, maze

i_sigma_dist_cm, i_wmax_Place_cells_recur, i_wmax_Place_cells_inh,\
i_wmax_inh_inh, i_wmax_inh_Place_cells, i_pconn_recur, i_tau_membr, i_tau_exc, i_tau_inh, i_dt = range(10)

netw_params = 	[50.0, 	        2*0.4114*nA, 12*0.8125*pA, 6*6*13.0*pA, 0.9*5*4.1*pA,  1, 10*ms, 6*ms,   2*ms,  0.2*ms] # 

tau_membr = netw_params[i_tau_membr]
tau_exc = netw_params[i_tau_exc] 
tau_inh = netw_params[i_tau_inh]

C, gL, EL, tauTrace = net_init.initConstants()


eqs_if_scaled_new, eqs_if_scaled_new_DG, eqs_if_trace = net_init.get_neuron_model()
Place_cells = NeuronGroup(N=netw['n_place_cells'], model=eqs_if_scaled_new, threshold='u > (EL + 20*mV)', reset="u = EL", refractory=3*ms, clock = defaultclock) # test for n_grid=80
Inh_neurons = NeuronGroup(N=netw['n_inh'], model=eqs_if_scaled_new, threshold='u > (EL + 20*mV)', reset="u = EL", refractory=4*ms, clock = defaultclock) # test for n_grid=80
 
DG_place_cells = NeuronGroup(N=netw['n_place_cells'], model=eqs_if_scaled_new_DG, threshold='u > (EL + 20*mV)', reset="u = EL; spiketrace =1", clock = defaultclock)  
poisson_inp_uniform = PoissonGroup(N=netw['n_place_cells'], rates=0*Hz, clock = defaultclock)
context_pop_home = NeuronGroup(N=netw['n_place_cells'], model=eqs_if_trace, threshold='u > (EL + 20*mV)', reset="u = EL; spiketrace =1", clock = defaultclock) # Standard version - time since last spike
context_pop_away = NeuronGroup(N=netw['n_place_cells'], model=eqs_if_trace, threshold='u > (EL + 20*mV)', reset="u = EL; spiketrace =1", clock = defaultclock)

Place_cells.u = EL
Inh_neurons.u = EL
DG_place_cells.u = EL
context_pop_home.u = EL
context_pop_away.u = EL


S_Place_cells_recur, S_Place_cells_Inh, S_Inh_Place_cells, S_Inh_Inh = net_init.get_synapses(Place_cells, Inh_neurons, netw_params, netw['n_grid'], maze['L_cm'], netw_params[i_sigma_dist_cm])

S_Poi_cont_home = IdentityConnection(poisson_inp_uniform, context_pop_home, 'u', weight=0*mV)
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

print "Learning enabled"


S_DG_Place_cells = Synapses(DG_place_cells, Place_cells, model='''w : 1''', pre ='''I_exc += w''')
fan_out = 2500 
L_maze_cm, n_grid = maze['L_cm'], netw['n_grid'] # local workaround - dict. access doesn't work inside the string in the following line
S_DG_Place_cells[:,:] = 'net_init.quad_dist_maze(i, j, n_grid, L_maze_cm) <= fan_out' # connect to CA3 cells within 50cm distance of place field centers

S_DG_Place_cells.w[:,:] = '0.9 * (10000.0 / fan_out) * 10 * 7.3e-13*amp * exp(-net_init.quad_dist_maze(i, j, n_grid, L_maze_cm) / fan_out)' # for tau_inh = 2ms

# Monitors
M_sp_Place_cells = SpikeMonitor(Place_cells)
M_sp_Inh = SpikeMonitor(Inh_neurons)
M_sp_DG = SpikeMonitor(DG_place_cells) 


