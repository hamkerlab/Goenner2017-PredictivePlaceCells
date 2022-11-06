from pylab import *
from random import *
from time import time
import numpy as np
from brian import *
import matplotlib.cm as cm
import scipy.ndimage.filters as filt

#--------------------------------------------------------------------------------------------------------
def initConstants():
    # aEIF constants - standard excitatory neuron:
    #C=281*pF # used until 29.6.
    C=300*pF # changed 30.6. for consistency with "test bed"!
    gL=30*nS
    EL=-70.6*mV
   
    #tauTrace = 1*second # test for n_grid = 72: too short
    #tauTrace = 3*second # n_grid = 35
    #tauTrace = 6*second # Test for n_grid = 63

    #tauTrace = 10*second # test 9.10.14 - works for 60cm place field radius
    #tauTrace = 20*second # 30*second # test 14.11.14
    #tauTrace = 5*second # test 17.11.14
    tauTrace = 0.1*second
   
    return C, gL, EL, tauTrace

#--------------------------------------------------------------------------------------------------------
def initConstants_movement():
    DeltaT_step_sec = 0.1 
    #speed_cm_s = 30.0 # works well
    speed_cm_s = 15.0 # Test 29.7.15: Adjusted to match experimental data?
    turning_prob = 0.1                  # don't change direction in every step
    turn_factor = 0.1                   # fraction of 2 pi - corresponding to turns in the range turn_factor * [-180 deg, 180 deg]
    spatialBinSize_cm = 2 # 1 # Test 9.10.14
    spatialBinSize_firingMap_cm = 2 # 1 
    return DeltaT_step_sec, speed_cm_s, turning_prob, turn_factor, spatialBinSize_cm, spatialBinSize_firingMap_cm

#--------------------------------------------------------------------------------------------------------

def get_neuron_model():
    eqs_if_scaled_new ='''
	du/dt  = - gL/C * (u - EL) + 1/C * (I_exc - I_inh + I_ext) + 2*mV*xi/(ms**0.5) : volt # std = 2mV/ms
	dI_exc/dt = - I_exc / tau_exc 		: amp
	dI_inh/dt = - I_inh / tau_inh 		: amp
	I_ext 					: amp
	''' # dI_inh/dt = - I_inh / tau_exc

    eqs_if_scaled_new_DG ='''
	du/dt  = - gL/C * (u - EL) + (I_exc - I_inh)/C : volt    
	dI_exc/dt = - I_exc / tau_exc 		: amp
	dI_inh/dt = - I_inh / tau_exc 		: amp
        dspiketrace/dt = -spiketrace/tauTrace   : 1
        DA : 1
	'''

    eqs_if_trace ='''
        du/dt  = -(u-EL)/tau_membr                  : volt
        dspiketrace/dt = -spiketrace/tauTrace       : 1
        ''' 

    return eqs_if_scaled_new, eqs_if_scaled_new_DG, eqs_if_trace

#--------------------------------------------------------------------------------------------------------

def get_synapses(Exc_neurons, Inh_neurons, netw_params, n_grid, L_maze_cm, sigma_dist_cm):
    i_sigma_dist_cm, i_wmax_exc_exc, i_wmax_exc_inh,\
    i_wmax_inh_inh, i_wmax_inh_exc, i_pconn_recur, i_tau_membr, i_tau_exc, i_tau_inh, i_dt = range(10)

    sigma_CA3_scaling = 0.5 # 1.0

    S_Exc_Exc = Synapses(Exc_neurons, Exc_neurons, model='''w : 1''', pre ='''I_exc += w''')
    S_Exc_Exc[:,:] = '(i != j)'
#    S_Exc_Exc.w[:,:] = 'netw_params[i_wmax_exc_exc] * 1/(sqrt(2*pi) * netw_params[i_sigma_dist_cm]) * exp(-quad_dist_maze(i, j, n_grid, L_maze_cm) / (2*(netw_params[i_sigma_dist_cm])**2) )'
    #S_Exc_Exc.w[:,:] = 'netw_params[i_wmax_exc_exc] * 1/(sqrt(2*pi) * netw_params[i_sigma_dist_cm]) * exp(-quad_dist_maze(i, j, n_grid, L_maze_cm) / ( 2*(netw_params[i_sigma_dist_cm] * ((L_maze_cm/float(n_grid)) / (300.0/72.0))  )**2))'  # result: 2*sigma^2 = 80 for sigma_dist=30 => sigma=6.3 cm (sigma^2 = 40cm^2)
    S_Exc_Exc.w[:,:] = 'netw_params[i_wmax_exc_exc] * 1/(sqrt(2*pi) * netw_params[i_sigma_dist_cm] * sigma_CA3_scaling**2) * exp(-quad_dist_maze(i, j, n_grid, L_maze_cm) / ( 2*(netw_params[i_sigma_dist_cm] * sigma_CA3_scaling * ((350.0/float(n_grid)) / (300.0/72.0))  )**2))'  # result: 2*sigma^2 = 80 for sigma_dist=30 => sigma=6.3 cm (sigma^2 = 40cm^2) # changed 2.11.15: for n_grid=80, L_maze_cm could be > 350

    # Add normalization of CA3 recurrent weight sum for scaled sigma?
    # sum( S_Place_cells_recur.w )=  2.36723903755e-05 # for sigma_CA3_scaling = 0.5 !!!
    # sum( S_Place_cells_recur.w )=  2.13580244921e-05 # for sigma_CA3_scaling = 1.0 !!!
    if sigma_CA3_scaling <> 1.0:
        realsum_rec = S_Exc_Exc.w.data.sum()
        defaultsum = 2.13580244921e-5
        S_Exc_Exc.w[:,:] *= defaultsum / realsum_rec
    
    hom_delay_ms = 1
    S_Exc_Exc.delay = (0.5+hom_delay_ms)*ms

    S_Exc_Inh = Synapses(Exc_neurons, Inh_neurons, model='''w : 1''', pre ='''I_exc += w''')
    S_Exc_Inh[:,:] = True
    S_Exc_Inh.w[:,:] = netw_params[i_wmax_exc_inh]
    S_Exc_Inh.delay = (0.5+hom_delay_ms)*ms

    S_Inh_Exc = Synapses(Inh_neurons, Exc_neurons, model='''w : 1''', pre ='''I_inh += w''')
    S_Inh_Exc[:,:] = True
    S_Inh_Exc.w[:,:] = netw_params[i_wmax_inh_exc] / np.sqrt(sigma_CA3_scaling)
    S_Inh_Exc.delay = (0.5+hom_delay_ms)*ms 

    S_Inh_Inh = Synapses(Inh_neurons, Inh_neurons, model='''w : 1''', pre ='''I_inh += w''')
    S_Inh_Inh[:,:] = '(i != j)'
    S_Inh_Inh.w[:,:] = netw_params[i_wmax_inh_inh]
    S_Inh_Inh.delay = (0.5+hom_delay_ms)*ms

    return S_Exc_Exc, S_Exc_Inh, S_Inh_Exc, S_Inh_Inh

#--------------------------------------------------------------------------------------------------------

def get_synapses_rand(Exc_neurons, Inh_neurons, netw_params, n_grid, L_maze_cm, sigma_dist_cm):
    i_sigma_dist_cm, i_wmax_exc_exc, i_wmax_exc_inh,\
    i_wmax_inh_inh, i_wmax_inh_exc, i_pconn_recur, i_tau_membr, i_tau_exc, i_tau_inh, i_dt = range(10)

    S_Exc_Exc = Synapses(Exc_neurons, Exc_neurons, model='''w : 1''', pre ='''I_exc += w''')

    #S_Exc_Exc[:,:] = '(i != j)'
    #S_Exc_Exc.w[:,:] = 'netw_params[i_wmax_exc_exc] * 1/(sqrt(2*pi) * netw_params[i_sigma_dist_cm]) * exp(-quad_dist_maze(i, j, n_grid, L_maze_cm) / ( 2*(netw_params[i_sigma_dist_cm] * ((L_maze_cm/float(n_grid)) / (300.0/72.0))  )**2))'  # result: 2*sigma^2 = 80 for sigma_dist=30

    #S_Exc_Exc[:,:] = '(i != j) * rand() < 0.5'
    #S_Exc_Exc.w[:,:] = '2 * netw_params[i_wmax_exc_exc] * 1/(sqrt(2*pi) * netw_params[i_sigma_dist_cm]) * exp(-quad_dist_maze(i, j, n_grid, L_maze_cm) / ( 2*(netw_params[i_sigma_dist_cm] * ((L_maze_cm/float(n_grid)) / (300.0/72.0))  )**2))'
 
    S_Exc_Exc[:,:] = '(i != j)' # Now testing: Full connectivity, 50% noise!
    S_Exc_Exc.w[:,:] = '(1.0*rand()+0.5) * netw_params[i_wmax_exc_exc] * 1/(sqrt(2*pi) * netw_params[i_sigma_dist_cm]) * exp(-quad_dist_maze(i, j, n_grid, L_maze_cm) / ( 2*(netw_params[i_sigma_dist_cm] * ((L_maze_cm/float(n_grid)) / (300.0/72.0))  )**2))'

    
    hom_delay_ms = 1
    S_Exc_Exc.delay = (0.5+hom_delay_ms)*ms

    sparseness = 0.5 # 0.9 # 0.9 works (fanout=16000)

    rand_lowerb = sparseness - (1 - sparseness)
    rand_variation = 2*(1 - sparseness) # mean(rand_lowerb + rand() * rand_variation) = sparseness!
    mult_factor = (1 / sparseness)**2 # compensate for lower connectivity AND randomness in weights!

    S_Exc_Inh = Synapses(Exc_neurons, Inh_neurons, model='''w : 1''', pre ='''I_exc += w''')
    S_Exc_Inh[:,:] = 'rand() < sparseness' 
    S_Exc_Inh.w[:,:] = 'mult_factor * (rand_lowerb + rand() * rand_variation) * netw_params[i_wmax_exc_inh]'
    S_Exc_Inh.delay = (0.5+hom_delay_ms)*ms

    S_Inh_Exc = Synapses(Inh_neurons, Exc_neurons, model='''w : 1''', pre ='''I_inh += w''')
    S_Inh_Exc[:,:] = 'rand() < sparseness' 
    S_Inh_Exc.w[:,:] = 'mult_factor * (rand_lowerb + rand() * rand_variation) * netw_params[i_wmax_inh_exc]'
    S_Inh_Exc.delay = (0.5+hom_delay_ms)*ms 

    S_Inh_Inh = Synapses(Inh_neurons, Inh_neurons, model='''w : 1''', pre ='''I_inh += w''')
    S_Inh_Inh[:,:] = '(i != j) * (rand() < sparseness)'
    S_Inh_Inh.w[:,:] = 'mult_factor * (rand_lowerb + rand() * rand_variation) * netw_params[i_wmax_inh_inh]'
    S_Inh_Inh.delay = (0.5+hom_delay_ms)*ms

    return S_Exc_Exc, S_Exc_Inh, S_Inh_Exc, S_Inh_Inh


#--------------------------------------------------------------------------------------------------------

def get_synapses_stf(Exc_neurons, Inh_neurons, netw_params, n_grid, L_maze_cm, sigma_dist_cm): # WITH STF !!!
    i_sigma_dist_cm, i_wmax_exc_exc, i_wmax_exc_inh,\
    i_wmax_inh_inh, i_wmax_inh_exc, i_pconn_recur, i_tau_membr, i_tau_exc, i_tau_inh, i_dt = range(10)

    S_Exc_Exc = Synapses(Exc_neurons, Exc_neurons, model='''w : 1
                                                            mod_factor : 1''', pre ='''I_exc += w * mod_factor''')
    S_Exc_Exc[:,:] = '(i != j)'
#    S_Exc_Exc.w[:,:] = 'netw_params[i_wmax_exc_exc] * 1/(sqrt(2*pi) * netw_params[i_sigma_dist_cm]) * exp(-quad_dist_maze(i, j, n_grid, L_maze_cm) / (2*(netw_params[i_sigma_dist_cm])**2) )'
    S_Exc_Exc.w[:,:] = 'netw_params[i_wmax_exc_exc] * 1/(sqrt(2*pi) * netw_params[i_sigma_dist_cm]) * exp(-quad_dist_maze(i, j, n_grid, L_maze_cm) / ( 2*(netw_params[i_sigma_dist_cm] * ((L_maze_cm/float(n_grid)) / (300.0/72.0))  )**2))'
    S_Exc_Exc.mod_factor = 1
    
    hom_delay_ms = 1
    S_Exc_Exc.delay = (0.5+hom_delay_ms)*ms

    S_Exc_Inh = Synapses(Exc_neurons, Inh_neurons, model='''w : 1''', pre ='''I_exc += w''')
    S_Exc_Inh[:,:] = True
    S_Exc_Inh.w[:,:] = netw_params[i_wmax_exc_inh]
    S_Exc_Inh.delay = (0.5+hom_delay_ms)*ms

    S_Inh_Exc = Synapses(Inh_neurons, Exc_neurons, model='''w : 1''', pre ='''I_inh += w''')
    tau_facil = 1*second
    #f_rest = 0.333 # Test for I_max = 1.2x # 0.4 # good for i_max = 1.0x #
    f_rest = (n_grid==72) * 0.333 + (n_grid==63) * 0.4 # 
    S_Inh_Exc = Synapses(Inh_neurons, Exc_neurons, model='''w : 1
                                                        delta_f : 1
                                                        dfacil/dt = (f_rest - facil)/tau_facil : 1''', 
                                               pre = '''I_inh += w * facil
                                                        facil += delta_f * f_rest * (facil-f_rest) * (1-facil)''', clock = defaultclock) # TEST 29.10.14: Synaptic facilitation?!  # 
    S_Inh_Exc[:,:] = True
    S_Inh_Exc.w[:,:] = netw_params[i_wmax_inh_exc]
    S_Inh_Exc.delay = (0.5+hom_delay_ms)*ms 

    S_Inh_Inh = Synapses(Inh_neurons, Inh_neurons, model='''w : 1''', pre ='''I_inh += w''')
    S_Inh_Inh[:,:] = '(i != j)'
    S_Inh_Inh.w[:,:] = netw_params[i_wmax_inh_inh]
    S_Inh_Inh.delay = (0.5+hom_delay_ms)*ms

    return S_Exc_Exc, S_Exc_Inh, S_Inh_Exc, S_Inh_Inh

#--------------------------------------------------------------------------------------------------------

def get_synapses_wmod(Exc_neurons, Inh_neurons, netw_params, n_grid, L_maze_cm, sigma_dist_cm):
    i_sigma_dist_cm, i_wmax_exc_exc, i_wmax_exc_inh,\
    i_wmax_inh_inh, i_wmax_inh_exc, i_pconn_recur, i_tau_membr, i_tau_exc, i_tau_inh, i_dt = range(10)

    scale_dist = (350.0 / float(n_grid)) / (300.0 / 72.0)

    # weight modulation:
    print "before modulation"
    pvm = zeros([n_grid, n_grid])
    mod_factor = 0.4 # 0.2  ## 0.2 looks good for n_grid = 80!
    n_exc = int(n_grid**2)
    Z = weight_mod(mod_factor, range(n_exc), n_exc, n_grid)
    for iNeuron in xrange(n_exc):       
        x,y = xypos(iNeuron, n_grid) 
        pvm[x, y] = Z[iNeuron]
        #ion(), matshow(pvm), colorbar(), title('Example for weight modulation'), show()

    S_Exc_Exc = Synapses(Exc_neurons, Exc_neurons, model='''w : 1''', pre ='''I_exc += w''')
    S_Exc_Exc[:,:] = '(i != j)' # Test with spatially correlated noise in recurrent weights! Result: As low as 5% deviation drastcally shifts the end points!
    S_Exc_Exc.w[:,:] = 'weight_mod(mod_factor, j, n_exc, n_grid) * netw_params[i_wmax_exc_exc] * 1/(sqrt(2*pi) * sigma_dist_cm) * exp(-quad_dist_maze(i, j, n_grid, L_maze_cm) / (2*(sigma_dist_cm * scale_dist)**2) )' # L_maze_cm = 350, n_grid = 54
    #'''
    print "weights have been generated" 
    
    hom_delay_ms = 1
    S_Exc_Exc.delay = (0.5+hom_delay_ms)*ms

    S_Exc_Inh = Synapses(Exc_neurons, Inh_neurons, model='''w : 1''', pre ='''I_exc += w''')
    S_Exc_Inh[:,:] = True
    S_Exc_Inh.w[:,:] = netw_params[i_wmax_exc_inh]
    S_Exc_Inh.delay = (0.5+hom_delay_ms)*ms

    S_Inh_Exc = Synapses(Inh_neurons, Exc_neurons, model='''w : 1''', pre ='''I_inh += w''')
    S_Inh_Exc[:,:] = True
    S_Inh_Exc.w[:,:] = netw_params[i_wmax_inh_exc]
    S_Inh_Exc.delay = (0.5+hom_delay_ms)*ms 

    S_Inh_Inh = Synapses(Inh_neurons, Inh_neurons, model='''w : 1''', pre ='''I_inh += w''')
    S_Inh_Inh[:,:] = '(i != j)'
    S_Inh_Inh.w[:,:] = netw_params[i_wmax_inh_inh]
    S_Inh_Inh.delay = (0.5+hom_delay_ms)*ms

    return S_Exc_Exc, S_Exc_Inh, S_Inh_Exc, S_Inh_Inh

#--------------------------------------------------------------------------------------------------------

def xypos(mat_index, L):
    # Given that an array of L^2 neurons is arranged on a toroidal LxL grid, 
    # returns x and y grid coordinates for a given index
    # Values are in [0, L]
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
    # 
    # Inverse (arrangement) formula: mat_index =  nGrid*(x/DeltaL - 0.5) + (y/DeltaL - 0.5)
    
    mat_index = array(mat_index)
    nGrid = float(nGrid)
    DeltaL = L_maze / nGrid    
    x = (floor(mat_index / nGrid) + 0.5) * DeltaL  
    y = (mat_index - nGrid * floor(mat_index / nGrid) + 0.5) * DeltaL
    
    return x,y
#--------------------------------------------------------------------------------------------------------
def xy_index_maze(x, y, nGrid, L_maze):

    DeltaL = L_maze / float(nGrid)
    mat_index = floor(y/DeltaL) + nGrid * floor(x/DeltaL)    

    return mat_index
#--------------------------------------------------------------------------------------------------------
def quad_dist_grid(i, j, L):
    # Returns the SQUARED euclidean distance between i and j on a "normal" LxL grid (non-toroidal).
    xPre, yPre   = xypos(i, L) 
    xPost, yPost = xypos(j, L)    
    one_divby_floatL  = 1/float(L)    
    Deltax = (xPre-xPost) # bounded topology!
    Deltay = (yPre-yPost)    

    return one_divby_floatL**2 * (Deltax**2 + Deltay**2)    
#--------------------------------------------------------------------------------------------------------
def quad_dist_maze(i, j, nGrid, L_maze):
    xi, yi = xypos_maze(i, nGrid, L_maze)
    xj, yj = xypos_maze(j, nGrid, L_maze)

    return (xi-xj)**2 + (yi-yj)**2
#--------------------------------------------------------------------------------------------------------
def Exp_xyValue(mat_index, L, scale_noise, goal_index):
    # Given that an array of L^2 neurons is arranged on a toroidal LxL grid, 
    # returns "place-value" for a given MATRIX index
    # EXPONENTIALLY increasing in x and y directions    
    x,y = xypos(mat_index, L)
    xGoal, yGoal = xypos(goal_index, L)
    L = float(L)    
    val = exp(- sqrt( (x-xGoal)**2 + (y-yGoal)**2) / 60 ) + scale_noise * rand(size(mat_index)) # n_grid=70
    val /= (1 + scale_noise)
    
    return val
#-------------------------------------------------------------------------------------------------------
def Exp_xyValue2(mat_index, L, scale_noise, goal_index, sigma):
    # Given that an array of L^2 neurons is arranged on a toroidal LxL grid, 
    # returns "place-value" for a given MATRIX index
    # EXPONENTIALLY increasing in x and y directions    
    x,y = xypos(mat_index, L)
    xGoal, yGoal = xypos(goal_index, L)
    L = float(L)    
    val = exp(- sqrt( (x-xGoal)**2 + (y-yGoal)**2) / sigma ) + scale_noise * rand(size(mat_index)) # n_grid=70
    val /= (1 + scale_noise)
    
    return val
#-------------------------------------------------------------------------------------------------------
def plot_sequence(nGrid, nValueCells, M_sp_PlaceCells, placevalMat): #SpikeCount_PlaceCells,
    # plot a matrix showing which neurons spiked:
    L = nGrid
    spCountMat = zeros([L, L])
    spTimeMat = nan*ones([L,L]) 
    placeValueMat = zeros([L, L])
    for iNeuron in xrange(nValueCells):
        x,y = xypos(iNeuron, L)        
        placeValueMat[x, y] = placevalMat[iNeuron]
        times = M_sp_PlaceCells.spiketimes[iNeuron] / ms
        if len(times) > 0:
            spTimeMat[x, y]  = times[len(times) -1]            
    figure()        
    subplot(1,2,1)
    matshow(transpose(spTimeMat), cmap=cm.YlOrRd, fignum=False)
    colorbar()
    title('Spike times of place cells [ms]')
    xlabel('x position')
    ylabel('y position')    
    subplot(1,2,2)
    matshow(transpose(placeValueMat), fignum=False)
    colorbar()
    title('Place-value')
    xlabel('x position')
    ylabel('y position')
    return

#--------------------------------------------------------------------------------------------------------
def plot_firingMaps(nPlaceCells, mazeSize_cm, spatialBinSize_firingMap_cm, M_sp_PC, pathRecord_x_cm, pathRecord_y_cm, M_sp_DG, occupancyMap):
        # show place cell firing map: Randomly draw some place cells


        k_sp_bump   = nonzero((M_sp_PC.it[1] > 0.125) * (M_sp_PC.it[1] < 0.625))    # Indices of spikes between 0.125 and 0.625 sec (the first bump phase)
        k_sp_nobump = nonzero(M_sp_PC.it[1] > 0.625)                            # Indices of spikes after the bump phase

        bc_bump   = bincount(M_sp_PC.it[0][k_sp_bump],   minlength = nPlaceCells)   # Counts the spikes per neuron during the bump phase
        bc_nobump = bincount(M_sp_PC.it[0][k_sp_nobump], minlength = nPlaceCells)   # Counts the spikes per neuron after the bump phase
        bc_nobump_only = ((bc_bump == 0) * (bc_nobump > 0)) * bc_nobump              # Counts the spikes of neurons active ONLY AFTER the bump phase

        iN_nobump_only = nonzero(bc_nobump_only)[0]                          # Indices of neurons active ONLY AFTER the bump phase
        jsort_nobump_only = argsort(bc_nobump_only[iN_nobump_only])          # LIST indices sorting these neurons by no. of spikes, ascending order
        iN_sorted_nobump_only = iN_nobump_only[jsort_nobump_only]            # Indices of NEURONS active ONLY AFTER the bump phase, sorted by no. of spikes
        l_nobump_only = len(iN_sorted_nobump_only)

        ion()

        pcIndex = zeros(8)
	for i in xrange(len(pcIndex)):
            #pcIndex[i] = randrange(nPlaceCells)
	    #pcIndex[i] = 480 # goal index
            #for j in xrange(i-1):
            #    while pcIndex[i] == pcIndex[j]:
            #        pcIndex[i] = randrange(nPlaceCells)
            pcIndex[i] = iN_sorted_nobump_only[l_nobump_only - i - 1]
   
        firingMap_PC = nan * ones([mazeSize_cm / spatialBinSize_firingMap_cm, mazeSize_cm / spatialBinSize_firingMap_cm]) # use nan to create white background for plotting
        spike_mat = -inf*ones([mazeSize_cm / spatialBinSize_firingMap_cm, mazeSize_cm / spatialBinSize_firingMap_cm]) # use zeros to enable smoothing
        rate_mat = zeros([mazeSize_cm / spatialBinSize_firingMap_cm, mazeSize_cm / spatialBinSize_firingMap_cm]) # use zeros to enable smoothing

	for iPlaceCells in xrange(len(pcIndex)):
            spike_mat = zeros([mazeSize_cm / spatialBinSize_firingMap_cm, mazeSize_cm / spatialBinSize_firingMap_cm]) # use zeros to enable smoothing
            rate_mat = zeros([mazeSize_cm / spatialBinSize_firingMap_cm, mazeSize_cm / spatialBinSize_firingMap_cm]) # use zeros to enable smoothing
            spiketimes = M_sp_PC.spiketimes[pcIndex[iPlaceCells]]

            xmin = 200
            xmax = 0
            ymin = 200
            ymax = 0

            for iTimes in xrange(len(spiketimes)):        
                #x = pathRecord_x_cm[spiketimes[iTimes]]
                #y = pathRecord_y_cm[spiketimes[iTimes]]
                x = pathRecord_x_cm[int(rint(spiketimes[iTimes] / 0.1)) ]
                y = pathRecord_y_cm[int(rint(spiketimes[iTimes] / 0.1)) ]

                if x < xmin:
                    xmin = x
                if x > xmax:
                    xmax = x
                if y < ymin:
                    ymin = y
                if y > ymax:
                    ymax = y

                firingMap_PC[rint(x / spatialBinSize_firingMap_cm), rint(y / spatialBinSize_firingMap_cm)] = iPlaceCells # defines the color by index
		spike_mat[rint(x / spatialBinSize_firingMap_cm), rint(y / spatialBinSize_firingMap_cm)] = max(1, spike_mat[rint(x / spatialBinSize_firingMap_cm), rint(y / spatialBinSize_firingMap_cm)] + 1) # += 1 # = iPlaceCells
            subplot(4, len(pcIndex), 1 + iPlaceCells)
            rate_mat = (spike_mat*(occupancyMap <> 0)) / ((occupancyMap <> 0)*occupancyMap + (occupancyMap==0)*pi) # spike_mat / occupancyMap
            convolved_rate = filt.gaussian_filter(rate_mat, 9) # convolution with a Gaussian kernel    
            if convolved_rate.max() > 0:
                convolved_rate *= rate_mat.max() / convolved_rate.max()
	    matshow(convolved_rate, fignum=False)

            subplot(4, len(pcIndex), len(pcIndex) + 1 + iPlaceCells)
	    matshow(spike_mat, fignum=False)

            #title('PC '+str(int(pcIndex[iPlaceCells])))
            if xmin <> xmax and ymin <> ymax:
                title(str(int(xmax-xmin))+'x'+str(int(ymax-ymin)), fontsize=8)
            colorbar()


        k_sp_bump   = nonzero((M_sp_DG.it[1] > 0.125) * (M_sp_DG.it[1] < 0.625))    # Indices of spikes between 0.125 and 0.625 sec (the first bump phase)
        k_sp_nobump = nonzero(M_sp_DG.it[1] > 0.625)                            # Indices of spikes after the bump phase

        bc_bump   = bincount(M_sp_DG.it[0][k_sp_bump],   minlength = nPlaceCells)   # Counts the spikes per neuron during the bump phase
        bc_nobump = bincount(M_sp_DG.it[0][k_sp_nobump], minlength = nPlaceCells)   # Counts the spikes per neuron after the bump phase
        bc_nobump_only = ((bc_bump == 0) * (bc_nobump > 0)) * bc_nobump              # Counts the spikes of neurons active ONLY AFTER the bump phase

        iN_nobump_only = nonzero(bc_nobump_only)[0]                          # Indices of neurons active ONLY AFTER the bump phase
        jsort_nobump_only = argsort(bc_nobump_only[iN_nobump_only])          # LIST indices sorting these neurons by no. of spikes, ascending order
        iN_sorted_nobump_only = iN_nobump_only[jsort_nobump_only]            # Indices of NEURONS active ONLY AFTER the bump phase, sorted by no. of spikes
        l_nobump_only = len(iN_sorted_nobump_only)

        
        # randomly draw some value cells: # DG CELLS!
        #vcIndex = zeros(10)
        vcIndex = zeros(len(pcIndex))
        for i in xrange(len(vcIndex)):
            #vcIndex[i] = randrange(nPlaceCells)
            #for j in xrange(i-1):
            #    while vcIndex[i] == vcIndex[j]:
            #        vcIndex[i] = randrange(nPlaceCells)
	#vcIndex[3] = 480 # goal index
             vcIndex[i] = iN_sorted_nobump_only[l_nobump_only - i - 1]
        
        firingMap_VC = nan * ones([mazeSize_cm / spatialBinSize_firingMap_cm, mazeSize_cm / spatialBinSize_firingMap_cm]) # use nan to create white background for plotting

        for iValueCells in xrange(len(vcIndex)):
            spiketimes = M_sp_DG.spiketimes[vcIndex[iValueCells]]
            spike_mat_vc = zeros([mazeSize_cm / spatialBinSize_firingMap_cm, mazeSize_cm / spatialBinSize_firingMap_cm]) # use zeros to enable smoothing
            rate_mat_vc = zeros([mazeSize_cm / spatialBinSize_firingMap_cm, mazeSize_cm / spatialBinSize_firingMap_cm]) # use zeros to enable smoothing

            xmin = 350
            xmax = 0
            ymin = 350
            ymax = 0
                
            for iTimes in xrange(len(spiketimes)):        
                #x = pathRecord_x_cm[spiketimes[iTimes]]
                #y = pathRecord_y_cm[spiketimes[iTimes]]
                x = pathRecord_x_cm[int(rint(spiketimes[iTimes] / 0.1)) ]
                y = pathRecord_y_cm[int(rint(spiketimes[iTimes] / 0.1)) ]


                if x < xmin:
                    xmin = x
                if x > xmax:
                    xmax = x 
                if y < ymin:
                    ymin = y
                if y > ymax:
                    ymax = y
                area = (xmax - xmin) * (ymax - ymin)
             
                firingMap_VC[rint(x / spatialBinSize_firingMap_cm), rint(y / spatialBinSize_firingMap_cm)] = iValueCells # defines the color by index
		spike_mat_vc[rint(x / spatialBinSize_firingMap_cm), rint(y / spatialBinSize_firingMap_cm)] = 1 # max(1, spike_mat_vc[rint(x / spatialBinSize_firingMap_cm), rint(y / spatialBinSize_firingMap_cm)] + 1) # += 1 # = iPlaceCells
            subplot(4, len(vcIndex), 2*len(vcIndex) + 1 + iValueCells)
            rate_mat_vc = (spike_mat_vc*(occupancyMap <> 0)) / ((occupancyMap <> 0)*occupancyMap + (occupancyMap==0)*pi) # spike_mat / occupancyMap
            convolved_rate_vc = filt.gaussian_filter(rate_mat_vc, 9) # convolution with a Gaussian kernel
            if convolved_rate_vc.max() > 0:
                convolved_rate_vc *= rate_mat_vc.max() / convolved_rate_vc.max()
	    matshow(convolved_rate_vc, fignum=False) 

            subplot(4, len(vcIndex), 3*len(vcIndex) + 1 + iValueCells)
	    matshow(spike_mat_vc, fignum=False) 
            ax=gca()
            for label in ax.get_xticklabels() + ax.get_yticklabels():
                label.set_fontsize(8)
            ax.set_xticks([])
            ax.set_xticklabels([])
            ax.set_yticks([0, mazeSize_cm / spatialBinSize_firingMap_cm])
            ax.set_yticklabels([0, mazeSize_cm])


 	    #colorbar()
            cbar = colorbar() #fraction = 0.05)
            cbar.set_ticks([cbar.vmin, cbar.vmax])
            cbar.set_ticklabels([cbar.vmin, cbar.vmax])
            axes(cbar.ax)
            yticks(fontsize=8)
            axes(ax)

            #title('DG '+str(int(vcIndex[iValueCells])))
            if xmin <> xmax and ymin <> ymax:
                #title( '('+str(int(xmin))+','+str(int(ymin))+')-('+str(int(xmax))+','+str(int(ymax))+')', fontsize=8)
                title( str(int(xmax-xmin))+'x'+str(int(ymax-ymin)), fontsize=8)
         
        # savefig('placefield_plot_'+str(randint(100)))
        savefig('placefield_plot_'+str(int(time()))  )



        ioff()        
        #show()



	return spike_mat # firingMap_PC

#--------------------------------------------------------------------------------------------------------
        
def plot_LearnedPlaceValue(L_grid, nPlaceCells, S_VC_PC):
    # Show learned place-value:
    
    plotW_VC_PC = zeros([L_grid, L_grid])    
    for iNeuron in xrange(nPlaceCells):
        x,y = xypos(iNeuron, L_grid)
        plotW_VC_PC[x, y] = S_VC_PC.w[iNeuron, iNeuron]
    
    #subplot(1,4,4)    
    matshow(plotW_VC_PC, fignum=False) 
    #colorbar() 
    title('Synaptic weights - value cell projection to place cells')
    
    return

#--------------------------------------------------------------------------------------------------------

def weight_mod(factor, mat_index, nmax, L):
 
    x,y = xypos(mat_index, L)
    L = float(L)
    N=20
    ci = randint(0, nmax, N)
    Z_mod = zeros(len(mat_index))
    for k in xrange(N):  
        xc, yc = xypos(ci[k], L)
        Z_mod += exp(- ((x-xc)**2 + (y-yc)**2) / 10.0**2 ) 
    Z_mod -= Z_mod.mean(); Z_mod /= Z_mod.max() # zero mean, max. 1
 
    wmod = ones(len(mat_index)) + factor * Z_mod # mean = 1, max = 1 + factor
  
    return wmod 
        
