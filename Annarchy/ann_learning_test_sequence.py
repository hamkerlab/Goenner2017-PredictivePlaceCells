# Implementation of the network described in 
# Azizi et al., Front. Comp. Neurosci. 2013

# TODO: Test for membrane potential oscillations ?! (Cf. English et al. 2014)

#from random import *

import pickle
import matplotlib.cm as cm
import gc

from ANNarchy import *
import numpy as np
from pylab import *
from scipy.sparse import lil_matrix
from time import time
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import gridspec
import scipy.stats as st
from scipy.signal import hilbert

setup(dt = 0.2)
#setup(dt = 1.0) # test 9.11.15


#--------------------------------------------------------------------------------------------------------
def xypos(mat_index, L):
    # Given that an array of L^2 neurons is arranged on a toroidal LxL grid,  returns x and y grid coordinates for a given index, values are in [0, L] 
    # Inverse (arrangement) formula: mat_index = x*L + y    
    mat_index = array(mat_index)    
    L = float(L)
    x = floor(mat_index / L)
    y = mat_index - L * floor(mat_index / L)    
    return x,y
#--------------------------------------------------------------------------------------------------------
def xypos_maze(mat_index, nGrid, L_maze):
    # Given that an array of nGrid^2 neurons is arranged on a square (nGrid x nGrid) grid, 
    # returns x and y grid coordinates corresponding to the centers of the "checkerboard fields"
    # Inverse (arrangement) formula: mat_index =  nGrid*(x/DeltaL - 0.5) + (y/DeltaL - 0.5)    
    mat_index = array(mat_index)
    nGrid = float(nGrid)
    DeltaL = L_maze / nGrid    
    x = (floor(mat_index / nGrid) + 0.5) * DeltaL   #floor(mat_index / L)
    y = (mat_index - nGrid * floor(mat_index / nGrid) + 0.5) * DeltaL    
    return x,y
#--------------------------------------------------------------------------------------------------------
def xy_index_maze(x, y, nGrid, L_maze):
    DeltaL = L_maze / float(nGrid)
    mat_index = floor(y/DeltaL) + nGrid * floor(x/DeltaL)    
    return mat_index
#--------------------------------------------------------------------------------------------------------
def plot_sequence(nGrid, nValueCells, M_sp_PlaceCells, placevalMat, dt):    
    # plot a matrix showing which neurons spiked:
    print("Creating sequence plot...")
    L = nGrid
    spCountMat = zeros([L, L])
    spTimeMat = nan*ones([L,L]) 
    placeValueMat = zeros([L, L])
    for iNeuron in range(nValueCells):
        x,y = xypos(iNeuron, L)        
        placeValueMat[x, y] = placevalMat[iNeuron]
        #times = M_sp_PlaceCells.spiketimes[iNeuron] / ms
        times = np.array(M_sp_PlaceCells[iNeuron]) * dt
        if len(times) > 0:
            spTimeMat[x, y]  = times[len(times) -1]            
    figure()        
    subplot(1,2,1)
    matshow(transpose(spTimeMat), origin='lower', cmap=cm.YlOrRd, fignum=False)

    colorbar()
    title('Spike times of place cells [ms]')
    xlabel('x position')
    ylabel('y position')    
    subplot(1,2,2)
    matshow(transpose(placeValueMat), origin='lower',fignum=False)
    colorbar()
    title('Place-value')
    xlabel('x position')
    ylabel('y position')
    return
#--------------------------------------------------------------------------------------------------------
def Exp_xyValue(mat_index, L, scale_noise, goal_index):
    # Given that an array of L^2 neurons is arranged on a plain LxL grid, 
    # returns "place-value" for a given MATRIX index
    # EXPONENTIALLY increasing in x and y directions    
    x,y = xypos(mat_index, L)
    xGoal, yGoal = xypos(goal_index, L)
    L = float(L)    

    val = exp(- sqrt( (x-xGoal)**2 + (y-yGoal)**2) / 20 ) + scale_noise * rand(size(mat_index), size(mat_index)) # test for smaller DG place fields: Works with (large) fan-out of 4000, scaling x1
    #val = exp(- sqrt( (x-xGoal)**2 + (y-yGoal)**2) / (20*90/63.0 ) ) + scale_noise * rand(size(mat_index), size(mat_index)) # test for n_grid = 90
    val = exp(- sqrt( (x-xGoal)**2 + (y-yGoal)**2) / (20*80/63.0 ) ) + scale_noise * rand(size(mat_index), size(mat_index)) # test for n_grid = 80

    val /= (1 + scale_noise)    
    return val
#-------------------------------------------------------------------------------------------------------
def Exp_xyValue_normalized(mat_index, L, scale_noise, goal_index):
    # Given that an array of L^2 neurons is arranged on a toroidal LxL grid, 
    # returns "place-value" for a given MATRIX index
    # EXPONENTIALLY increasing in x and y directions    

    x,y = xypos(mat_index, L)
    xGoal, yGoal = xypos(goal_index, L)
    L = float(L)    
    val = exp(- sqrt( (x-xGoal)**2 + (y-yGoal)**2) / (20*80/63.0 ) ) + scale_noise * rand(size(mat_index)) # n_grid=70
    val /= (1 + scale_noise)

    refsum = 1.0 * 3913.66 # For n_grid=80, max. sum of values (for goal_index=3199).
    valsum = sum(val)

    val *= refsum / valsum
    
    return val

#-------------------------------------------------------------------------------------------------------------

i_sigma_dist_cm, i_wmax_exc_exc, i_wmax_exc_inh,\
i_wmax_inh_inh, i_wmax_inh_exc, i_pconn_recur, i_tau_membr, i_tau_exc, i_tau_inh, i_dt = list(range(10))

#-------------------------------------------------------------------------------------------------------------

t_init = time()

n_grid = 80 # 90 # 72 # test!!!
print("n_grid = ", n_grid)
n_exc = int(n_grid**2)
n_inh = int(0.2*36**2) # works also with n_grid=70 - 1:20 ratio


scale_length = 2

#netw_params = 	[sigma_dist_cm, wmax_exc_exc, wmax_exc_inh, wmax_inh_inh, wmax_inh_exc,         pconn_recur,    tau_membr, tau_exc, tau_inh,   dt]
netw_params = 	[50.0, 	        0.8228e3,             9.75,      468,      0.9*20.5,  1, 10, 12,    2,  0.2]


t_refr_exc = 3
t_refr_inh = 4

print("netw_params= ", netw_params)

# Constants

tau_membr = netw_params[i_tau_membr]
tau_exc = netw_params[i_tau_exc] 
tau_inh = netw_params[i_tau_inh]

maze_edge_cm = 110 # 150 # 115 # for n_grid=80, sigma=50
L_maze_cm = 200 + 2 * maze_edge_cm

print("L_maze_cm = ", L_maze_cm)
print("maze_edge_cm = ", maze_edge_cm)
scale_dist = (350.0 / float(n_grid)) / (300.0 / 72.0)

sigma_dist_cm = netw_params[i_sigma_dist_cm] # factor scale_length is applied in the weight formula
sigma_dist_new_cm = netw_params[i_sigma_dist_cm] # 30 # scale_length * sigma_dist_cm

eqs_if_scaled_new = Neuron(
	parameters = '''gL_nS = 30 : population
        		C_pF = 300 : population
        		EL_mV = -70.6 : population
		        sigma_mV = 2.0* 31.62 : population
			tau_exc_ms = 6.0 : population
			tau_inh_ms = 2.0 : population
			I_ext_pA = 0.0
			t_refr = 3.0 : population''', # 
	equations = ''' noise = Normal(0.0, 1.0) 
        		C_pF/gL_nS * du_mV/dt = - (u_mV - EL_mV) + 1/gL_nS * (g_exc + g_excDG - g_inh + I_ext_pA) + sigma_mV * noise : init = -70.6
			dg_exc/dt = - g_exc / tau_exc_ms : init = 0.0
			dg_excDG/dt = - g_excDG / tau_exc_ms : init = 0.0
			dg_inh/dt = - g_inh / tau_inh_ms : init = 0.0
                        ext_input = I_ext_pA''', 
	name = 'eqs_if_scaled_new', spike = 'u_mV > (EL_mV + 20)', reset = 'u_mV = EL_mV', refractory = 't_refr') # *0.2  #### multiply sigma_mV with 0.2 for dt=1 ms ?!

eqs_if_scaled_new_DG = Neuron(
	parameters = '''gL_nS = 30 : population
        		C_pF = 300 : population
        		EL_mV = -70.6 : population
			tau_exc_ms = 6.0 : population ''',
	equations = ''' C_pF/gL_nS * du_mV/dt = - (u_mV - EL_mV) + 1/gL_nS * (g_exc - g_inh)  : init = -70.6
			dg_exc/dt = - g_exc / tau_exc_ms : init = 0.0
			dg_inh/dt = - g_inh / tau_exc_ms : init = 0.0 ''', 
	name = 'eqs_if_scaled_new', spike = 'u_mV > (EL_mV + 20)', reset = 'u_mV = EL_mV') 

eqs_if_trace = Neuron(
	parameters = '''EL_mV = -70.6 : population
        		tau_membr = 10.0 : population''',
        equations = ''' u_sp = g_copy_mV : init = 0.0  
			dg_copy_mV / dt = -g_copy_mv / 1000.0''', 
	name = 'eqs_if_trace', 	spike = 'u_sp > 0', reset = '''g_copy_mV = 0.0''') # if u_sp > 0 :   # dg_copy_mV/dt = - g_copy_mV / 2.0 : init = 0.0 # du_mV/dt = - (u_mV - EL_mV) / tau_membr + g_copy_mV : init = -70.6



Exc_neurons = Population(geometry=(n_grid, n_grid), neuron = eqs_if_scaled_new, name = "Exc_neurons")
Inh_neurons = Population(n_inh, neuron = eqs_if_scaled_new, name = "Inh_neurons")
Inh_neurons.t_refr = t_refr_inh
DG_neurons = Population(geometry=(n_grid, n_grid), neuron = eqs_if_scaled_new_DG, name = "DG_neurons")
rate_factor_test = 1 #  Test 6.8.2015
poisson_inp_hi = PoissonPopulation(geometry=(n_grid, n_grid), rates=200) # '1.5 * 100.0 * (1.0 + 1.0*sin(2*pi*25*t/1000.0 - 0.5*pi))' ) # 25 Hz 
poisson_inp_lo = PoissonPopulation(geometry=(n_grid, n_grid), rates=10) # '5.0 * (1.0 + 1.0*sin(2*pi*25*t/1000.0 - 0.5*pi))' ) 

context_pop_home = Population(geometry=(n_grid, n_grid), neuron = eqs_if_trace, name = "context_pop_home")
context_pop_home.u_mV = context_pop_home.EL_mV

placevalMat = zeros(n_exc)
start_bias = zeros(n_exc)
start_index = xy_index_maze(maze_edge_cm,  maze_edge_cm, n_grid, L_maze_cm)
print("start_index = ", start_index)
print("start location (x,y) = ", xypos_maze(start_index, n_grid, L_maze_cm))

goal_index = xy_index_maze(200+maze_edge_cm, maze_edge_cm, n_grid, L_maze_cm) # center?
print("goal_index = ", goal_index)

nonfiring_index = xy_index_maze(75, 275, n_grid, L_maze_cm) # 

xr,yr = xypos(goal_index, n_grid)
print("reward location (x,y) = ", xypos_maze(goal_index, n_grid, L_maze_cm)) # L_maze_cm / float(n_grid)*xr, L_maze_cm / float(n_grid)*yr

placevalMat[list(range(n_exc))] = np.random.rand(n_exc) * 0.3 # * 0.4 # defined in nA - caution, tau_exc is now 4x larger than previously!!! # 0.4e-9

filename = 'learned_weights_DGinp'


#'''#
file_wdata_VC_PC = open('data/'+filename, 'rb') 
Wdata_VC_PC=pickle.load(file_wdata_VC_PC, encoding='latin1') 
file_wdata_VC_PC.close()
print("file = ", filename)
placevalMat = Wdata_VC_PC * 1e9 #* 0.5 # convert to nA - wdata from file are in ampere, max. ca. 1e-9 - caution, tau_exc is now 4x larger than previously!!!
placevalMat = (Wdata_VC_PC * 1e9)**0.635 # test 11.7.16
#'''
print("wmax (placevalMat) = ", np.max(placevalMat))

simple_syn =Synapse(pre_spike=""" g_target += w """) 
copy_syn =Synapse(pre_spike=""" g_target += w """)
plastic_syn_rewardmod = Synapse(
	parameters = 	"""lrate_pA_per_ms = 50.0  : projection
					tau_trace_ms = 100.0 			: projection
					tau_DA_ms = 10000.0 			: projection 
					""", # learning_rate = (5000 pA)/(100.0 ms)
	equations = 	"""
					dtrace_pre /dt = - trace_pre / tau_trace_ms	: event-driven, init = 0.0
					dtrace_post/dt = - trace_post/ tau_trace_ms	: event-driven, init = 0.0
					dw / dt = if DA > 0:
									 if trace_pre*trace_post - w > 0: 
										 lrate_pA_per_ms * (trace_pre*trace_post - w) 
									 else: 
										 0
								 else:
									if DA < 0:	
										 trace_pre*trace_post
									else: 
										 0
					dDA/dt = - DA / tau_DA_ms 								: init = 0.0
					""", # event-driven
	pre_spike =		"""
					g_target += w 
					trace_pre = 1.0
					""",
	post_spike = 	"""
					trace_post = 1.0
					"""
)


max_synapse_exc = netw_params[i_wmax_exc_exc] * 1/(sqrt(2*pi) * sigma_dist_cm)
min_synapse_exc = 0.0 ## 1.0e-6 * netw_params[i_wmax_exc_exc] * 1/(sqrt(2*pi) * sigma_dist_cm)
sigma_dist_sheet_units =  sigma_dist_cm * scale_dist / L_maze_cm ### / (L_maze_cm / n_grid) * 0.1841 * 1e-3 # test ## Distances between neurons are normalized to the [0,1] range by ANNarchy!!!
print("max_synapse_exc = ", max_synapse_exc)
print("sigma_dist_sheet_units = ", sigma_dist_sheet_units)

S_Exc_Exc = Projection(
	pre=Exc_neurons, post=Exc_neurons, target='exc', synapse=simple_syn
).connect_gaussian( amp = max_synapse_exc, sigma = sigma_dist_sheet_units, limit = min_synapse_exc, allow_self_connections = False, delays = 2.5) # 2.5

S_Exc_Inh = Projection(
    pre=Exc_neurons, post=Inh_neurons, target='exc', synapse=simple_syn
).connect_all_to_all( weights = netw_params[i_wmax_exc_inh], delays = 2.5) # 2.5

S_Inh_Inh = Projection(
    pre=Inh_neurons, post=Inh_neurons, target='inh', synapse=simple_syn
).connect_all_to_all( weights = netw_params[i_wmax_inh_inh], allow_self_connections = False, delays = 2.5) # 2.5

S_Inh_Exc = Projection(
    pre=Inh_neurons, 
    post=Exc_neurons, 
    target='inh',
    synapse=simple_syn
).connect_all_to_all( weights = netw_params[i_wmax_inh_exc], delays = 2.5) # 2.5

S_Poi_lo_cont_home = Projection(
	pre = poisson_inp_lo, post = context_pop_home, target = 'copy_mV', synapse = copy_syn
).connect_one_to_one( weights = 21.0 ) # orig. value 21.0 # simple_syn

S_Poi_hi_cont_home = Projection(
	pre = poisson_inp_hi, post = context_pop_home, target = 'copy_mV', synapse = copy_syn
).connect_one_to_one( weights = 0.0 ) # orig. value 21.0 # simple_syn


lil_w = lil_matrix( np.diag(1/rate_factor_test * placevalMat * 1e3) )
S_context_home_DG = Projection(
	pre = context_pop_home, post = DG_neurons, target = 'exc', synapse = plastic_syn_rewardmod # simple_syn
).connect_from_sparse( weights = lil_w ) 



fan_out = 5000.0 
fan_out = 10000.0
fan_out = 2500.0 
sigmafan_sheet_units = sqrt(0.5 * fan_out) / L_maze_cm # / n_grid)
max_synapse_dg_exc = 0.9 * 7.3 * (10000.0 / fan_out) 
min_synapse_dg_exc = 0.1 # 0.05 # 0.26855 # 

print("max_synapse_dg_exc = ", max_synapse_dg_exc)

S_DG_Exc = Projection(
	pre = DG_neurons, post = Exc_neurons, target = 'excDG', synapse = simple_syn
).connect_gaussian( amp = max_synapse_dg_exc, sigma = sigmafan_sheet_units, limit = min_synapse_dg_exc, delays = 0.1) # 2.0

start_bias = zeros(n_exc)
start_bias[list(range(n_exc))] = Exp_xyValue_normalized(list(range(n_exc)), n_grid, 0, start_index) # normalized - slower movement?!
print("max. start_bias = ", max(start_bias))

Exc_neurons.I_ext_pA = start_bias* 0.9e3 

stim_time_ms = 35 
init_time_ms = 50 
run_time_ms = 350 

global total_run_time_ms
total_run_time_ms = init_time_ms + run_time_ms

print("Initiating the network...") 
print("Poisson to DG synapses already active!")



# Don't forget to set the same weights below!!!

print("DG to CA3 synapses already active!")

# ------------------------------------------------------------------------------------

ion()

cont_popview = context_pop_home[0:50]

compile()

#'''#
print("Modifying synaptic weights...")
for dend in S_Exc_Exc.dendrites:
    dend.w *= (1.2*np.random.rand(dend.size)+0.4) 
print("Done")
#'''


# initiation period
Exc_neurons.u_mV = Exc_neurons.EL_mV # + rand(n_exc)*5
Inh_neurons.u_mV = Inh_neurons.EL_mV # + rand(n_exc)*5
DG_neurons.u_mV = DG_neurons.EL_mV


# Start recording
M_Exc = Monitor(Exc_neurons, ['spike', 'u_mV', 'g_exc', 'g_excDG', 'g_inh', 'ext_input','r']) # , 'I_ext_pA'
M_Inh = Monitor(Inh_neurons, ['spike', 'u_mV', 'g_exc', 'g_inh'])
M_DG = Monitor(DG_neurons, ['spike']) 
M_contpop = Monitor(cont_popview, ['spike']) #
center_mat = -Inf*ones([100, 100]) # Test 1.10.14: Not bad
center_len = len(center_mat[0,:])
n_edge = ceil(maze_edge_cm / float(L_maze_cm) * center_len)


deltaT = 5.0 # 8.0 # 5.0 # 2.5 # 5.0 # 2.5 # 1.0 # 5.0
shift_ms = 2.0 # 2.0 # 2.5 # 5.0 # 2.5 # 0.8 # deltaT # 2.5 # 0.2 # 

Exc_neurons.compute_firing_rate(window=deltaT)


simulate(stim_time_ms, measure_time=True) # 35 ms

print("Continuing init without I_ext") 

# end of initiation period
Exc_neurons.I_ext_pA = 0

simulate(init_time_ms - stim_time_ms, measure_time=True)

print("Switching to place-value noise...")

S_Poi_lo_cont_home.w = 0.0
S_Poi_hi_cont_home.w = 21.0


t_start = time()
t_sim = run_time_ms # 200
t_curr = 0
for step in range(10):
    simulate(t_sim * 0.1) # , measure_time=True)
    t_estim = int( (9-step) * (time()-t_start) / (step+1) )
    print("%i %% done, approx. %i seconds remaining" %( int((step+1)*10),  t_estim))
print("Duration: %i seconds" %( int(time() - t_start) ))


ion()

dpisize= 300 #        

fsize = 5 # 7 # 
col_blue2 = '#0072B2' #
col_green = '#2B9F78'
col5 = '#E69F00' # 'Orange'
col_purple = '#CC79A7'
col3 = '#D55E00' # 'Tomato'

t_end = init_time_ms + run_time_ms # 50 # 625


figure(figsize=(5,4), dpi=dpisize) 

subplot(711)
contpop_spikes = M_contpop.get(['spike'])
print("Mean Context population rate = ", np.mean( np.mean(M_contpop.population_rate(contpop_spikes, smooth=100.0)) )) # M_poi
te, ne = M_contpop.raster_plot(contpop_spikes)
tmin_cont, tmax_cont = te.min(), te.max()
plot(te, ne, 'k.', markersize=1.0)
xlabel('')
ylabel('context', fontsize=fsize)
ax=gca()
ax.set_xticks(arange(0, 440, 40))
ax.set_xticklabels([])
ax.xaxis.grid()
ax.set_yticks([0, 50])
ax.set_yticklabels([0, 50])
axis([0, 400, axis()[2], axis()[3]])
for label in ax.get_xticklabels() + ax.get_yticklabels(): 
    label.set_fontsize(fsize) #  6

ax_wrapper=[]
ax_wrapper.append(subplot(712))
dg_spikes = M_DG.get(['spike'])
te, ne = M_DG.raster_plot(dg_spikes)
plot(te, ne, 'k.', markersize=1.0)
xlabel('')
ylabel('DG', fontsize=fsize)
ax=gca()
ax.set_xticks(arange(0, 440, 40))
ax.set_xticklabels([])
ax.xaxis.grid()
ax.set_yticks([0, 6400])
ax.set_yticklabels([0, 6400])
for label in ax.get_xticklabels() + ax.get_yticklabels(): 
    label.set_fontsize(fsize) #  6
axis([0, 400, axis()[2], axis()[3]])

subplot(713)
exc_spikes = M_Exc.get(['spike'])
te, ne = M_Exc.raster_plot(exc_spikes)
axis([0, 400, axis()[2], axis()[3]])
'''#
file_spsave = open('spikedata_exc_normalized_08.txt', 'w')
pickle.dump(exc_spikes, file_spsave, 0)
pickle.dump(te, file_spsave, 0)
pickle.dump(ne, file_spsave, 0)
file_spsave.close()
'''
plot(te, ne, 'k.', markersize=0.1) # 0.25 # 1.0
xlabel('')
ylabel('CA3 exc.', fontsize=fsize)
ax=gca()
ax.set_xticks(arange(0, 440, 40))
ax.set_xticklabels([])
ax.xaxis.grid()
ax.set_yticks([0, 6400])
ax.set_yticklabels([0, 6400])
for label in ax.get_xticklabels() + ax.get_yticklabels(): 
    label.set_fontsize(fsize) #  6

subplot(714)
inh_spikes = M_Inh.get(['spike'])
ti, ni = M_Inh.raster_plot(inh_spikes)
plot(ti, ni, 'k.', markersize=1.0)
ylabel('CA3 inh.', fontsize=fsize) # 
ax=gca()
ax.set_xticks(arange(0, 440, 40))
ax.set_xticklabels([])
ax.xaxis.grid()
ax.set_yticks([0, n_inh])
ax.set_yticklabels([0, n_inh])
for label in ax.get_xticklabels() + ax.get_yticklabels(): 
    label.set_fontsize(fsize) # 6
axis([0, 400, axis()[2], axis()[3]])





x_centers, y_centers = xypos_maze(list(range(n_exc)), n_grid, L_maze_cm) ## Is this correct?
center_mat = np.inf * np.ones([100, 100])
scale = len(center_mat[0,:]) / float(L_maze_cm)

'''#
rates = M_Exc.smoothed_rate(exc_spikes, smooth=deltaT)
bump_center_x_cm = np.inf * np.ones(rates.shape[1])
bump_center_y_cm = np.inf * np.ones(rates.shape[1])
for t_ind in xrange(rates.shape[1]): 
    if np.sum(rates[:, t_ind]) > 0:
        bump_center_x_cm[t_ind] = np.average(x_centers, weights = rates[:, t_ind]) #  rates (unsmoothed) contains the reciprocal inter-spike-interval - alternative?
        bump_center_y_cm[t_ind] = np.average(y_centers, weights = rates[:, t_ind])
        center_mat[int(scale * bump_center_x_cm[t_ind]), int(scale * bump_center_y_cm[t_ind])] = dt() * t_ind
'''
#'''#
rates = M_Exc.get(['r'])
bump_center_x_cm = np.inf * np.ones(rates.shape[0])
bump_center_y_cm = np.inf * np.ones(rates.shape[0])
x_coords, y_coords = zeros(n_exc), zeros(n_exc)
for i_neuron in range(n_exc):
	x_coords[i_neuron], y_coords[i_neuron] = Exc_neurons.coordinates_from_rank(i_neuron)
for t_ind in range(rates.shape[0]): 
    if np.sum(rates[t_ind,:]) > 0:
        bump_center_x_cm[t_ind] = np.average(float(L_maze_cm) / n_grid * x_coords, weights = rates[t_ind,:])
        bump_center_y_cm[t_ind] = np.average(float(L_maze_cm) / n_grid * y_coords, weights = rates[t_ind,:])
        center_mat[int(scale * bump_center_x_cm[t_ind]), int(scale * bump_center_y_cm[t_ind])] = dt() * t_ind
        if t_ind >= 250 and t_ind < 260:
            print("rates[t_ind,:].mean(), rates[t_ind,:].max(), bump_center_x_cm[t_ind] = ", rates[t_ind,:].mean(), rates[t_ind,:].max(), bump_center_x_cm[t_ind])

#'''

print("x_init_rate, y_init_rate = %1.1f, %1.1f" %( bump_center_x_cm[50*5], bump_center_y_cm[50*5] ))
print("x_end_rate, y_end_rate = %1.1f, %1.1f" %( bump_center_x_cm[-1], bump_center_y_cm[-1] ))
diff_xy_cm = sqrt( (diff(bump_center_x_cm))**2 + (diff(bump_center_y_cm))**2 )
diff_x_cm = diff(bump_center_x_cm)
diff_y_cm = diff(bump_center_y_cm)



# Alternative - sliding window with adjustable overlap:
t_start = 0.0
t_end = 400.0 # 
 
n_frames = floor((t_end - t_start - deltaT) / shift_ms + 1) # with overlap
n_startframe = floor(t_start / shift_ms)
n_endframe = floor((t_end - deltaT)/ shift_ms)
#xall, yall = xypos(range(n_grid**2), n_grid)
xe = nan * ones(int(n_frames))
ye = nan * ones(int(n_frames))
spcount = nan * ones([int(n_frames), n_grid**2])

for i_frame in arange(int(n_startframe), int(n_endframe)+1):
    if mod(i_frame, 10)==0: print("i_frame = ", i_frame)
    nz_spiking = nonzero((te > i_frame * shift_ms) * (te <= i_frame * shift_ms + deltaT))
    ni = ne[nz_spiking] # With overlap
    counts = bincount(ni, minlength=n_grid**2)
    if max(bincount(ni, minlength=n_grid**2)) > 0:
        xe[i_frame] = np.average(x_centers, weights = counts)
        ye[i_frame] = np.average(y_centers, weights = counts)
        spcount[i_frame, :] = counts
loopdiff_x_cm = diff(xe)
loopdiff_y_cm = diff(ye)





#figure()
it_start = 50*5

#subplot(717)
ax_wrapper=[]
ax_wrapper.append(subplot(717))
plot(append(nan * ones(it_start), diff_xy_cm[it_start : ] / (dt()/shift_ms) ), color=col_blue2)#, lw=0.5) # units: m/sec
plot(append(nan * ones(it_start), diff_x_cm[it_start : ] / (dt()/shift_ms) ), '--', color=col3, lw=0.5) # col5) #, lw=0.5) # units: m/sec # Raw units: cm/dt = 0.01*m / (0.0002*sec) = m/(0.02sec)
firstind = int(50.0 / shift_ms)
plot(shift_ms/dt() * arange(0, len(loopdiff_x_cm)), append(nan*ones(firstind), loopdiff_x_cm[firstind:]), '-', color='b', lw=0.5) # / (0.02*shift_ms) # units: m/sec # Raw units: cm/shift_ms=m/(0.001*sec*shift_ms)=m/(0.1*sec*shift_ms/dt * dt) = m/(0.02*sec*shift_ms)
ylabel('Step size \n [m/sec]  ', fontsize=fsize)
ax=gca()
poprate_dg = M_DG.population_rate(dg_spikes, smooth=deltaT) # 10.0
poprate_exc = M_DG.population_rate(exc_spikes, smooth=deltaT) # 10.0
poprate_context = M_contpop.population_rate(contpop_spikes, smooth=deltaT) # 10.0
ax.set_xticks(5 * arange(0, 440, 40))
ax.set_xticklabels([])
ax.xaxis.grid()
axis([0, 400.0 / dt(), axis()[2], axis()[3]])

ax.set_yticks([axis()[2], 0, axis()[3]])
ax.yaxis.grid()
for label in ax.get_xticklabels() + ax.get_yticklabels(): 
    label.set_fontsize(fsize) #  6
xlabel('t [ms]', fontsize=fsize)
legend(('2D step size','x-step [ann]','windowed'), fontsize=fsize)

ax_wrapper.append(ax_wrapper[0].twinx())
mon_gexcCA3 = M_Exc.get(['g_exc'])
mon_gexcDG = M_Exc.get(['g_excDG'])
mon_ginh = M_Exc.get(['g_inh'])
print("mon_gexcCA3.shape = ", mon_gexcCA3.shape)
plot(mean(mon_gexcDG, 1) / mean(mon_gexcCA3, 1), color=col_green)
ax_twin=gca()
ax_twin.set_yticks([axis()[2], axis()[3]])
#legend(('2D step size','x-step','g_excCA3','g_excDG'), fontsize=fsize)
for label in ax_twin.get_xticklabels() + ax_twin.get_yticklabels(): 
    label.set_fontsize(fsize) #  6
    label.set_color(col_green)
ylabel('g_excDG / g_excCA3', fontsize=fsize, color=col_green)

ax_wrapper=[]
ax_wrapper.append(subplot(715))
DG_col = 'k' # '0.4'
plot(poprate_dg[ : ], color=DG_col)
ax_twin=gca()
ax_twin.xaxis.grid()
ax_twin.set_xticks(5 * arange(0, 440, 40))
labels = append(repeat("", len(arange(0, 400, 40))), '400')
labels[0] = 0
axis([0, 400.0 / dt(), axis()[2], axis()[3]])
ax_twin.set_yticks([axis()[2], axis()[3]])
ax_twin.set_xticklabels([])
for label in ax_twin.get_xticklabels() + ax_twin.get_yticklabels(): 
    label.set_fontsize(fsize) #  6
    label.set_color(DG_col)
ylabel('DG \n pop. rate \n [sp/sec]', fontsize=fsize, color=DG_col)
ax_wrapper.append(ax_wrapper[0].twinx())
plot(poprate_exc[ : ], color=col_green)
ylabel('CA3 exc.\n pop. rate \n [sp/sec]', fontsize=fsize, color=col_green)
ax=gca()
ax.set_yticks([axis()[2], axis()[3]])
for label in ax.get_xticklabels() + ax.get_yticklabels(): 
    label.set_fontsize(fsize) #  6
    label.set_color(col_green)



subplot(716)
plot(bump_center_x_cm - maze_edge_cm, '0.4', lw=1)#0.5)
plot(bump_center_y_cm - maze_edge_cm, color=col_purple, lw=1)#0.5)
plot(shift_ms/dt() * arange(0, len(xe)), xe - maze_edge_cm, 'k--', lw=0.5)
ylabel('Decoded \n location \n [cm]', fontsize=fsize)
legend(('x [ann]', 'y','xe'), fontsize=fsize)
ax=gca()
ax.set_xticks(5 * arange(0, 440, 40))
labels = append(repeat("", len(arange(0, 400, 40))), '400')
labels[0] = 0
ax.set_xticklabels([])
ax.xaxis.grid()
axis([0, 400.0 / dt(), 0, axis()[3]])
ax.set_yticks([axis()[2], axis()[3]])
for label in ax.get_xticklabels() + ax.get_yticklabels(): 
    label.set_fontsize(fsize) #  6


subplots_adjust(bottom=0.08, top=0.95, hspace=0.5)


#savefig('ann_decode_plot6_smoothedrate.png', dpi=300)
savefig('ann_decode_plot6_compute_firing_rate.png', dpi=300)
#savefig('ann_raster_syncurrents_nocontextosc_withdelay_stepsizes_deltaT'+str(deltaT)+'ms_shift'+str(shift_ms)+'.png', dpi=300)


print("Total time = ", time()-t_init)

print("tmin_cont, tmax_cont = ", tmin_cont, tmax_cont)



ioff()
show()















