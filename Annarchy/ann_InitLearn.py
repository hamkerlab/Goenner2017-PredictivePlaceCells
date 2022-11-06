from pylab import *
from random import *
from time import time
import numpy as np
#from brian import *
import matplotlib.cm as cm
import scipy.ndimage.filters as filt

#--------------------------------------------------------------------------------------------------------
def initConstants():
    # aEIF constants - standard excitatory neuron:
    C=300*pF # changed 30.6. for consistency with "test bed"!
    gL=30*nS
    EL=-70.6*mV
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

#--------------------------------------------------------------------------------------------------------

def weight_mod(factor, mat_index, nmax, L):
 
    x,y = xypos(mat_index, L)
    L = float(L)
    N=20
    ci = randint(0, nmax, N)
    Z_mod = zeros(len(mat_index))
    for k in range(N):  
        xc, yc = xypos(ci[k], L)
        Z_mod += exp(- ((x-xc)**2 + (y-yc)**2) / 10.0**2 ) 
    Z_mod -= Z_mod.mean(); Z_mod /= Z_mod.max() # zero mean, max. 1
 
    wmod = ones(len(mat_index)) + factor * Z_mod # mean = 1, max = 1 + factor
  
    return wmod 
        
