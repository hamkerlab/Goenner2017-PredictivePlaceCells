# TO DO:
# Separate evaluations / figures between "home" and "away" trials for
# - reward latencies
# - distance between sequence endpoints and reward locations
# - navigation times (per strategy)
# - Bump movement plots

# CAUTION:
# Networks not reaching the performance criterion (aborted after 30 min.) must be excluded from analysis !!!

# ACHTUNG 19.10.18:
# Vermutung:
# In den Netzwerken mit 15 Trials geht beim reshape() der occupancyMap und center_mat_plot / center_mat_array
# etwas durcheinander!!!
# Alternativ: Probleme bei ungerader Anzahl Trials?!
# Oder: Probleme bei folgender Zeile:
# 		center_mat_array_Home[i_netw][:] = center_mat_array[i_netw][nz_even_Home]



from brian import *
from brian.tools.datamanager import *
import pickle
import mDB_azizi_newInitLearn as net_init
import plotting_func as pf
from pylab import *

from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from scipy.optimize import curve_fit
from matplotlib import colors
import new_colormaps as nc
from scipy.interpolate import griddata

# ---------------------------------------------------------------------------------
# Defines a Gaussian function, used for evaluating Gaussian fits of learned weights
def func(x, a, mx, my, sigmaqd, b):                               
    return a * exp( -((x[0]-mx)**2 + (x[1]-my)**2)/sigmaqd ) + b
# ---------------------------------------------------------------------------------




standard_plots = 0 # 1
save_plotdata = 1  # 1
learning_prec_plot= 0
rotated_plot = 0 # 1
occupancy_plot = 1 # 0
i_net_occ = 9 # 1 # 0 # No bump movement for network 1 ?!

speed_plot = 0
figs_for_movie = 0
weights_for_movie = 0

savefigs = True # False # True #  

dpisize = 300 # 50 #           
dpisize_save = 300     
n_grid = 80 
maze_edge_cm = 110 # for n_grid = 80
L_maze_cm = 200 + 2 * maze_edge_cm

nTrials = 3 # 1 # 3
N = 10

#PFCalpha = 0.1 # 0.2	
#identifier = 'DAfactor_0-5_N'+str(N)+'_'+str(nTrials)+'trials'

#identifier = 'recoded_v1'+'_N'+str(N)+'_'+str(nTrials)+'trials'
#identifier = 'recoded_withSim_v2_N'+str(N)+'_'+str(nTrials)+'trials'
identifier = 'recoded_withSim_v3_N'+str(N)+'_'+str(nTrials)+'trials'

#nTrials = 3
#N = 20
#identifier = 'pubModel_all2all_PFCalpha0-0_N'+str(N)+'_'+str(nTrials)+'trials'




dataman = DataManager(identifier)

latencies, endpoints, weights, seqcount, seqstart, seqend, rand_nav_time, \
        goal_nav_time, focal_search_time, occupancyMap, weight_array_home, weight_array_away, center_mat_plot, center_mat_array, goal_index_array = zip(*dataman.values())

grid_factor = round(n_grid / 6.0)
xg, yg = net_init.xypos_maze(goal_index_array[0][0], n_grid, L_maze_cm)


# Excluding failed (too long) networks from the analysis:
valid_data = nonzero( (sum(latencies, 1) < 3600) * (sum(latencies, 1) > 0) )
#for i_netw in xrange(dataman.itemcount()):
#	print "Network, seqstart[i_netw].shape = ", i_netw, seqstart[i_netw].shape
invalid_data =  nonzero( (sum(latencies, 1) >= 3600) + (sum(latencies, 1) <= 0) )
print "valid_data =", valid_data
#print "invalid_data =", invalid_data

n_valid_nets = size(valid_data)
print "Excluded %i network(s) from analysis" %(dataman.itemcount() - n_valid_nets)

latencies = reshape(latencies, (dataman.itemcount(), nTrials))
endpoints = reshape(endpoints, (dataman.itemcount(), nTrials))
seqcount = reshape(seqcount, (dataman.itemcount(), nTrials))
seqstart = reshape(seqstart, (dataman.itemcount(), nTrials, len(seqstart[0][0])))
seqend = reshape(seqend, (dataman.itemcount(), nTrials, len(seqstart[0][0])))
rand_nav_time = reshape(rand_nav_time, (dataman.itemcount(), nTrials))
goal_nav_time = reshape(goal_nav_time, (dataman.itemcount(), nTrials))
focal_search_time = reshape(focal_search_time, (dataman.itemcount(), nTrials))
occ_len = len(occupancyMap[0][0][0,:])
if len(occupancyMap[0][:]) == 20:
        occupancyMap = reshape(occupancyMap, (dataman.itemcount(), 20, occ_len, occ_len))
elif len(occupancyMap[0][:]) == 40:
        occupancyMap = reshape(occupancyMap, (dataman.itemcount(), 40, occ_len, occ_len))        
else:
        occupancyMap = reshape(occupancyMap, (dataman.itemcount(), nTrials, occ_len, occ_len))        


weight_array_home = reshape(weight_array_home, (dataman.itemcount(), nTrials, n_grid**2))
weight_array_away = reshape(weight_array_away, (dataman.itemcount(), nTrials, n_grid**2))
center_len = len(center_mat_plot[0][0,:])
center_mat_plot = reshape(center_mat_plot, (dataman.itemcount(), center_len, center_len))
#for i_netw in xrange(dataman.itemcount()):
#	print "i_netw: center_mat_array[i_netw][0].max() = ", i_netw, center_mat_array[i_netw][0].max()
center_mat_array = reshape(center_mat_array, (dataman.itemcount(), nTrials, center_len, center_len))
goal_index_array = reshape(goal_index_array, (dataman.itemcount(), nTrials))


latencies = reshape(latencies[valid_data], (n_valid_nets, nTrials))
endpoints = reshape(endpoints[valid_data], (n_valid_nets, nTrials))
seqcount = reshape(seqcount[valid_data], (n_valid_nets, nTrials))
seqstart = reshape(seqstart[valid_data], (n_valid_nets, nTrials, len(seqstart[0][0])))
seqend = reshape(seqend[valid_data], (n_valid_nets, nTrials, len(seqend[0][0])))
rand_nav_time = reshape(rand_nav_time[valid_data], (n_valid_nets, nTrials))
goal_nav_time = reshape(goal_nav_time[valid_data], (n_valid_nets, nTrials))
focal_search_time = reshape(focal_search_time[valid_data], (n_valid_nets, nTrials))
if len(occupancyMap[0][:]) == 20:
        occupancyMap = reshape(occupancyMap[valid_data], (n_valid_nets, 20, occ_len, occ_len))
elif len(occupancyMap[0][:]) == 40:
        occupancyMap = reshape(occupancyMap[valid_data], (n_valid_nets, 40, occ_len, occ_len))  
else:
        occupancyMap = reshape(occupancyMap[valid_data], (n_valid_nets, nTrials, occ_len, occ_len))  

 
weight_array_home = reshape(weight_array_home[valid_data], (n_valid_nets, nTrials, n_grid**2))
weight_array_away = reshape(weight_array_away[valid_data], (n_valid_nets, nTrials, n_grid**2))
center_mat_plot = reshape(center_mat_plot[valid_data], (n_valid_nets, center_len, center_len))
center_mat_array = reshape(center_mat_array[valid_data], (n_valid_nets, nTrials, center_len, center_len))
goal_index_array = reshape(goal_index_array[valid_data], (n_valid_nets, nTrials))


nz_even_Home = nonzero(mod(xrange(nTrials),2)-1)
nz_odd_Away = nonzero(mod(xrange(nTrials),2))
latencies_Home = zeros([n_valid_nets, int( ceil(nTrials/2.0) )])
latencies_Away = zeros([n_valid_nets, int( floor(nTrials/2.0) )])
goal_nav_time_Home = zeros([n_valid_nets, int( ceil(nTrials/2.0) )])
goal_nav_time_Away = zeros([n_valid_nets, int( floor(nTrials/2.0) )])
focal_search_time_Home = zeros([n_valid_nets, int( ceil(nTrials/2.0) )])
focal_search_time_Away = zeros([n_valid_nets, int( floor(nTrials/2.0) )])
rand_nav_time_Home = zeros([n_valid_nets, int( ceil(nTrials/2.0) )])
rand_nav_time_Away = zeros([n_valid_nets, int( floor(nTrials/2.0) )])
endpoints_Home = zeros([n_valid_nets, int( ceil(nTrials/2.0) )])
endpoints_Away = zeros([n_valid_nets, int( floor(nTrials/2.0) )])
goal_index_array_Home = zeros([n_valid_nets, int( ceil(nTrials/2.0) )])
goal_index_array_Away = zeros([n_valid_nets, int( floor(nTrials/2.0) )])

center_mat_plot_Home = -Inf*ones([n_valid_nets, center_len, center_len])
center_mat_plot_Away = -Inf*ones([n_valid_nets, center_len, center_len])
center_mat_array_Home = -Inf*ones([n_valid_nets, int( ceil(nTrials/2.0) ), center_len, center_len])
seqstart_Home = zeros([n_valid_nets, int( ceil(nTrials/2.0) ), len(seqstart[0][0])])
seqstart_Away = zeros([n_valid_nets, int( floor(nTrials/2.0) ), len(seqstart[0][0])])
seqend_Home = zeros([n_valid_nets, int( ceil(nTrials/2.0) ), len(seqend[0][0])])
seqend_Away = zeros([n_valid_nets, int( floor(nTrials/2.0) ), len(seqend[0][0])])


for i_netw in xrange(n_valid_nets):
    latencies_Home[i_netw][:] = latencies[i_netw][nz_even_Home]
    latencies_Away[i_netw][:] = latencies[i_netw][nz_odd_Away]
    goal_nav_time_Home[i_netw][:] = goal_nav_time[i_netw][nz_even_Home]
    goal_nav_time_Away[i_netw][:] = goal_nav_time[i_netw][nz_odd_Away]
    rand_nav_time_Home[i_netw][:] = rand_nav_time[i_netw][nz_even_Home]
    rand_nav_time_Away[i_netw][:] = rand_nav_time[i_netw][nz_odd_Away]
    focal_search_time_Home[i_netw][:] = focal_search_time[i_netw][nz_even_Home]
    focal_search_time_Away[i_netw][:] = focal_search_time[i_netw][nz_odd_Away]
    endpoints_Home[i_netw][:] = endpoints[i_netw][nz_even_Home]
    endpoints_Away[i_netw][:] = endpoints[i_netw][nz_odd_Away]
    goal_index_array_Home[i_netw][:] = goal_index_array[i_netw][nz_even_Home]
    goal_index_array_Away[i_netw][:] = goal_index_array[i_netw][nz_odd_Away]
    center_mat_array_Home[i_netw][:] = center_mat_array[i_netw][nz_even_Home]
    seqstart_Home[i_netw][:] = seqstart[i_netw][nz_even_Home]
    seqend_Home[i_netw][:] = seqend[i_netw][nz_even_Home]
    if len(nz_odd_Away[0]) > 0:
        seqstart_Away[i_netw][:] = seqstart[i_netw][nz_odd_Away]
        seqend_Away[i_netw][:] = seqend[i_netw][nz_odd_Away]

 

#standard_plots = True # False #  

if standard_plots:

        ion()



        # plot reward latencies across time:
        pf.plot_latencies_timecourse(latencies_Home, latencies_Away, n_valid_nets, nTrials, identifier, dpisize, dpisize_save, savefigs, save_plotdata)

        #pf.plot_latency_hist(latencies_Home, latencies_Away, nTrials, identifier, dpisize, dpisize_save, savefigs)

        # Bar comparison of mean latencies (Away vs. Home):
        pf.plot_latencies_overall(latencies_Home, latencies_Away, nTrials, identifier, dpisize, dpisize_save, savefigs)

        # plot start-to-end distance of sequences:

        #dist_startend_Home, dist_startend_Away = 
	pf.plot_startenddist(seqstart, seqend, n_grid, L_maze_cm, n_valid_nets, nTrials, identifier, dpisize, dpisize_save, savefigs, save_plotdata)

        # plot remaining distance to goal:
        pf.plots_dist_to_goal(endpoints_Home, endpoints_Away, goal_index_array_Home, goal_index_array_Away, L_maze_cm, maze_edge_cm, n_grid, n_valid_nets, nTrials, identifier, dpisize, dpisize_save, savefigs, save_plotdata)



        # plot "Home" LEC-DG weights: 
        pf.plot_Home_weights(weight_array_home, weights, L_maze_cm, maze_edge_cm, n_grid, n_valid_nets, nTrials, identifier, dpisize, dpisize_save, savefigs)


        # plot "Away" LEC-DG weights: 
        pf.plot_Away_weights(weight_array_away, weights, L_maze_cm, maze_edge_cm, n_grid, n_valid_nets, nTrials, identifier, dpisize, dpisize_save, savefigs)

        #'''#
        # plot navigation times for the different strategies:
        pf.plot_navigation_strategies(rand_nav_time_Home, goal_nav_time_Home, focal_search_time_Home, rand_nav_time_Away, goal_nav_time_Away, focal_search_time_Away, n_valid_nets, nTrials, identifier, dpisize, dpisize_save, savefigs)

        #'''




        #'''#
        # Plot start-to-end distances of sequences:
        # DIFFERENCES found between seqstart and first location of bump initialisation!!!
        i_netw, j_trial, k_seq = nonzero(seqstart)        
        st_nz = zeros(len(i_netw))
        se_nz = zeros(len(i_netw))
        for ind in xrange(len(i_netw)):
            st_nz[ind] = seqstart[i_netw[ind]][j_trial[ind]][k_seq[ind]]
            se_nz[ind] = seqend[i_netw[ind]][j_trial[ind]][k_seq[ind]]
        xs, ys = net_init.xypos_maze(st_nz, n_grid, L_maze_cm)
        xe, ye = net_init.xypos_maze(se_nz, n_grid, L_maze_cm)
        dist_startend = sqrt((xs-xe)**2 + (ys-ye)**2)
        figure(figsize=(3,3), dpi=dpisize)
        hist(dist_startend, 40, linewidth = 0.5)
        ax = gca()
        for label in ax.get_xticklabels() + ax.get_yticklabels():
            label.set_fontsize(8)
        xlabel('d [cm]', fontsize=8)
        ylabel('No. of sequences', fontsize=8)
        title('Distribution of start-to-end distances of sequences (all trials)', fontsize=8)
        if savefigs:
                savefig('movement_data/'+'seq_distances_all_'+identifier+'_'+str(nTrials), dpi=dpisize_save) 
    	seqstart_nz = zeros([n_valid_nets, nTrials])
        for i_netw in xrange(n_valid_nets):
            j_trial, k_seq = nonzero(seqstart[i_netw])
            # Problem: If the time_out defined in the simulation script is exceeded, seqstart will contain zero-values
            # seqstart_nz[i_netw][:] = seqstart[i_netw][j_trial, k_seq]
            seqstart_nz[i_netw][:] = append(seqstart[i_netw][j_trial, k_seq], nan*ones(len(seqstart_nz[i_netw][:]) - len(seqstart[i_netw][j_trial, k_seq])) )

        seqstart_Home_nz = zeros([n_valid_nets, int( ceil(nTrials/2.0) )])
        seqend_Home_nz = zeros([n_valid_nets, int( ceil(nTrials/2.0) )])
        for i_netw in xrange(n_valid_nets):
            j_trial, k_seq = nonzero(seqstart_Home[i_netw])
            #seqstart_Home_nz[i_netw][:] = seqstart_Home[i_netw][j_trial, k_seq]
            #seqend_Home_nz[i_netw][:] = seqend_Home[i_netw][j_trial, k_seq]
            seqstart_Home_nz[i_netw][:] = append(seqstart_Home[i_netw][j_trial, k_seq], nan*ones(len(seqstart_Home_nz[i_netw][:]) - len(seqstart_Home[i_netw][j_trial, k_seq])) )
            seqend_Home_nz[i_netw][:] = append(seqend_Home[i_netw][j_trial, k_seq], nan*ones(len(seqend_Home_nz[i_netw][:]) - len(seqend_Home[i_netw][j_trial, k_seq])) )
        xs_home, ys_home = net_init.xypos_maze(seqstart_Home_nz, n_grid, L_maze_cm)
        xe_home, ye_home = net_init.xypos_maze(seqend_Home_nz, n_grid, L_maze_cm)
        delta_x_home = xe_home - xs_home
        delta_y_home = ye_home - ys_home

        seqstart_Away_nz = zeros([n_valid_nets, int(floor(nTrials / 2.0))])
        seqend_Away_nz = zeros([n_valid_nets, int(floor(nTrials / 2.0))])
        for i_netw in xrange(n_valid_nets):
            j_trial, k_seq = nonzero(seqstart_Away[i_netw])
            #seqstart_Away_nz[i_netw][:] = seqstart_Away[i_netw][j_trial, k_seq]
            #seqend_Away_nz[i_netw][:] = seqend_Away[i_netw][j_trial, k_seq]
            seqstart_Away_nz[i_netw][:] = append(seqstart_Away[i_netw][j_trial, k_seq], nan*ones(len(seqstart_Away_nz[i_netw][:]) - len(seqstart_Away[i_netw][j_trial, k_seq])) )
            seqend_Away_nz[i_netw][:] = append(seqend_Away[i_netw][j_trial, k_seq], nan*ones(len(seqend_Away_nz[i_netw][:]) - len(seqend_Away[i_netw][j_trial, k_seq])) )

        xs_away, ys_away = net_init.xypos_maze(seqstart_Away_nz, n_grid, L_maze_cm)
        xe_away, ye_away = net_init.xypos_maze(seqend_Away_nz, n_grid, L_maze_cm)


    
        delta_x_away = xe_away - xs_away
        delta_y_away = ye_away - ys_away

        alpha_home = arccos( (delta_x_home * 1 + delta_y_home * 0) / ((delta_x_home**2 + delta_y_home**2 == 0) + sqrt( (delta_x_home**2 + delta_y_home**2) * (1**2 + 0**2) ) ) ) 
        alpha_away = arccos( (delta_x_away * 1 + delta_y_away * 0) / ((delta_x_away**2 + delta_y_away**2 == 0) + sqrt( (delta_x_away**2 + delta_y_away**2) * (1**2 + 0**2) ) ) )

        ahs = std(alpha_home, 1)
        aas = std(alpha_away, 1)


        #show()  

        '''#
        # Plot of DIFFERENCES found between seqstart and first location of bump initialisation!!!
        start_dist_diff = zeros(len(dist_startend))
        x_start = zeros([n_valid_nets, nTrials])
        y_start = zeros([n_valid_nets, nTrials])
        for i_netw in xrange(n_valid_nets):
            for i_trial in xrange(nTrials):
                x_start[i_netw, i_trial], y_start[i_netw, i_trial] = net_init.xypos_maze(seqstart[i_netw][i_trial][0], n_grid, L_maze_cm)
                nz_i, nz_j = nonzero(center_mat_array[i_netw][i_trial] > -inf)
                imin = argmin(center_mat_array[i_netw][i_trial][nz_i, nz_j])
                x_start_bump, y_start_bump = net_init.xypos_maze(net_init.xy_index_maze(nz_i[imin], nz_j[imin], center_len, center_len), center_len, L_maze_cm)
                #start_dist_diff[i_netw * nTrials + i_trial] = sqrt( (xs[i_netw * nTrials + i_trial]-x_start_bump)**2 + (ys[i_netw * nTrials + i_trial]-y_start_bump)**2 )
                start_dist_diff[i_netw * nTrials + i_trial] = sqrt( (x_start[i_netw, i_trial] - x_start_bump)**2 + (y_start[i_netw, i_trial] - y_start_bump)**2 ) # 
                
                xtemp, ytemp = net_init.xypos_maze(seqstart[i_netw][i_trial][0], n_grid, L_maze_cm) # for checking
                #start_dist_diff[i_netw * nTrials + i_trial] = sqrt( (x_start[i_netw, i_trial] - xtemp)**2 + (y_start[i_netw, i_trial] - ytemp)**2 ) # check - OK

        ioff()
        '''	


if learning_prec_plot:
        # plot a histogram of "learning accuracy", measured by differences (in cm) between Home locations and centers of Gaussian fits to "Home" weights over the course of learning:        
        xind, yind = net_init.xypos(range(n_grid**2), n_grid)
        n_home_trials = int( ceil(nTrials/2.0) )
        xma = zeros([n_valid_nets, n_home_trials])
        yma = zeros([n_valid_nets, n_home_trials])
        xha = zeros([n_valid_nets, n_home_trials])
        yha = zeros([n_valid_nets, n_home_trials])

        trial_list = range(nTrials-2, nTrials, 2)
        #for i_trial in xrange(0, nTrials, 2):
        for i_trial in trial_list:
                #pa = zeros([ceil(len(trial_list)), n_valid_nets])
                pa = zeros([5, n_valid_nets])
                for i_netw in xrange(n_valid_nets):
                    print "curve_fit for netw ", i_netw
                    xgf, ygf = net_init.xypos_maze(goal_index_array[i_netw][i_trial], n_grid, n_grid)			
                    #pa[:,i_netw], pcov = curve_fit(func, [xind, yind], weight_array_home[i_netw][i_trial], p0=(1, 30, 30, 200, 0))
                    pa[:,i_netw], pcov = curve_fit(func, [xind, yind], weight_array_home[i_netw][i_trial], p0=(1, xgf + 2*randn(), ygf + 2*randn(), 200, 0))
                    xma[i_netw, int(ceil(i_trial/2.0))] = pa[1, i_netw]
                    yma[i_netw, int(ceil(i_trial/2.0))] = pa[2, i_netw]
                    xha[i_netw, int(ceil(i_trial/2.0))], yha[i_netw, int(ceil(i_trial/2.0))] = net_init.xypos(goal_index_array_Home[i_netw][0], n_grid) # redundant data inserted for easier display

        dist_weight_fit = L_maze_cm / float(n_grid) * sqrt((xma-xha)**2 + (yma-yha)**2)
        figure(figsize=(3,3), dpi=dpisize)
        boxplot(dist_weight_fit[:,n_home_trials-1])
        title('Distances between Home locations and centers of Gaussians fitted to Home weights', fontsize=8)
        xlabel('Trial', fontsize=8)
        ylabel('d [cm]', fontsize=8)
        ax = gca()
        for label in ax.get_xticklabels() + ax.get_yticklabels():
            label.set_fontsize(8)
        if savefigs:
                savefig('movement_data/'+'learning_prec_'+identifier+'_'+str(nTrials), dpi=dpisize_save) 

        # plot weights across distance from Home:
        #f.plot_weight_vs_dist(weight_array_home, goal_index_array_Home, n_grid, L_maze_cm, maze_edge_cm, n_valid_nets, nTrials, identifier, dpisize, dpisize_save, savefigs)
        pf.plot_weight_vs_dist(weight_array_home, weight_array_away, goal_index_array_Home, n_grid, L_maze_cm, maze_edge_cm, n_valid_nets, nTrials, identifier, dpisize, dpisize_save, savefigs)


#speed_plot = False # True # 
if speed_plot:
    #i_netw = 6 # 1
    #i_trial = 6 # 2 # 0
    max_seq_len_sec = 0.4 # 0.65 # 0.625
    binsize_sec = 0.0002 # 0.004  # 0.002 # 0.05
    # CAUTION: Function running_bump_center is called every 0.2ms, but values are stored into center_mat only every 20th timestep!

    array_speed = nan * ones([n_valid_nets, nTrials,  int( ceil(max_seq_len_sec / binsize_sec))])
    dist_cm = nan * ones([n_valid_nets, nTrials, int( ceil(max_seq_len_sec / binsize_sec))])
    total_dist_cm = nan * ones([n_valid_nets, nTrials, int( ceil(max_seq_len_sec / binsize_sec))])
    delta_total_dist_cm = nan * ones([n_valid_nets, nTrials, int( ceil(max_seq_len_sec / binsize_sec))])

    delta_t_sec = -inf * ones([n_valid_nets, nTrials,  int( ceil(max_seq_len_sec / binsize_sec))])
    times_new = -inf * ones([n_valid_nets, nTrials,  int( ceil(max_seq_len_sec / binsize_sec))])
    t_new_sec = -inf * ones([n_valid_nets, nTrials,  int( ceil(max_seq_len_sec / binsize_sec))])
    elapsed_time_sec = -inf * ones([n_valid_nets, nTrials,  int( ceil(max_seq_len_sec / binsize_sec))])

    nan_centermat_bool = nan * ones([n_valid_nets, nTrials, center_len, center_len])
    nan_centermat_jump = nan * ones([n_valid_nets, nTrials, center_len, center_len])
    nan_centermat_rest = nan * ones([n_valid_nets, nTrials, center_len, center_len])
    nan_centermat_nojump = nan * ones([n_valid_nets, nTrials, center_len, center_len])
    total_seq_dist_cm = nan * ones([n_valid_nets, nTrials])
    total_seq_dist_cm_jump = nan * ones([n_valid_nets, nTrials])
    total_seq_dist_cm_nojump = nan * ones([n_valid_nets, nTrials])
    array_speed_jump = nan * ones([n_valid_nets, nTrials,  int( ceil(max_seq_len_sec / binsize_sec))])
    array_speed_nojump = nan * ones([n_valid_nets, nTrials,  int( ceil(max_seq_len_sec / binsize_sec))])
    prop_travelled_100ms = nan * ones([n_valid_nets, nTrials])

    dist_new = nan * ones([n_valid_nets, nTrials,  int( ceil(max_seq_len_sec / binsize_sec))])
    max_dist_new = nan * ones([n_valid_nets, nTrials])

    for i_netw in xrange(n_valid_nets):
	    for i_trial in xrange(nTrials):
		    i,j = nonzero(center_mat_array[i_netw][i_trial] > -inf)
		    tmp = zeros(len(i))
		    for k in xrange(len(i)):
		        tmp[k] = center_mat_array[i_netw][i_trial][i[k]][j[k]]
		        nan_centermat_bool[i_netw][i_trial][i[k]][j[k]] = 1
		    i_sorted=argsort(tmp)
		    t_init_sec = center_mat_array[i_netw][i_trial][ i[i_sorted[0]] ][ j[i_sorted[0]] ]


		    for k in xrange(len(i) - 1):
			    t_new_sec[i_netw, i_trial, k] = center_mat_array[i_netw][i_trial][ i[i_sorted[k+1]] ][ j[i_sorted[k+1]] ]
			    t_old_sec = center_mat_array[i_netw][i_trial][ i[i_sorted[k  ]] ][ j[i_sorted[k  ]] ]
			    t_index = int( (t_new_sec[i_netw, i_trial, k] - t_init_sec) / binsize_sec)
			    times_new[i_netw, i_trial, t_index] = t_new_sec[i_netw, i_trial, k]
			    elapsed_time_sec[i_netw, i_trial, t_index] = t_new_sec[i_netw, i_trial, k] - t_init_sec
			    delta_t_sec[i_netw, i_trial, t_index] = t_new_sec[i_netw, i_trial, k] - t_old_sec
			    dist_cm[i_netw, i_trial, t_index] = L_maze_cm / float(center_len) * sqrt( (i[i_sorted[k+1]] - i[i_sorted[k]])**2 + (j[i_sorted[k+1]]-j[i_sorted[k]])**2 )
			    total_dist_cm[i_netw, i_trial, t_index] = L_maze_cm / float(center_len) * sqrt( (i[i_sorted[k+1]] - i[i_sorted[0]])**2 + (j[i_sorted[k+1]]-j[i_sorted[0]])**2 )
			    delta_total_dist_cm[i_netw, i_trial, t_index] = L_maze_cm / float(center_len) * (sqrt( (i[i_sorted[k+1]] - i[i_sorted[0]])**2 + (j[i_sorted[k+1]]-j[i_sorted[0]])**2 ) - sqrt( (i[i_sorted[k]] - i[i_sorted[0]])**2 + (j[i_sorted[k]]-j[i_sorted[0]])**2 ))
			    array_speed[i_netw, i_trial, t_index] = dist_cm[i_netw, i_trial, t_index] / delta_t_sec[i_netw, i_trial, t_index]
			    if t_new_sec[i_netw, i_trial, k] > 0.05:
			        dist_new[i_netw, i_trial, k] = sqrt( diff(i[i_sorted])[k]**2 + diff(j[i_sorted])[k]**2 ) # in units of cells

		    max_dist_new[i_netw, i_trial] = nanmax(dist_new[i_netw, i_trial, :])


    #distmean = mean(mean(dist_cm, 0), 0)
    distmean = nanmean(nanmean(dist_cm, 0), 0)
    cumdistsum = zeros(len(distmean))
    for k in xrange(len(distmean)):
        cumdistsum[k] = sum(distmean[0:k])

    ion()
    
    figure()
    subplot(211)
    plot(nanmax(total_dist_cm, 2), nanmax(dist_cm, 2), '.')
    title('Max. shift across total dist')
    subplot(212)
    maxdist = nanmax(dist_cm, 2)    
    hist(reshape(maxdist, prod(maxdist.shape)), 20)
    title('Distribution of maxdist') 
    ax=gca()
    ax.set_yscale('log')
    show()

  


 
    i_netw = 0# i_net_occ
    jumps = 0
    nojumps = 0
    rest = 0
    for i_netw in xrange(n_valid_nets): # n_valid_nets
        for i_trial in xrange(nTrials): # 
            nz_totdist = nonzero(isnan(total_dist_cm[i_netw, i_trial, :]) == 0)[0]
        
            # Potential criteria to identify "jump" sequences:
            # Total start-to-end-distance greater than 50 cm (?) and strictly positive values of delta_total_dist_cm for t >= 200 ms    
            # dist_startend_Home
            # Instead of loop over time steps (nonzeros) with elapsed_time_sec > 0.2, check if min(delta_total_dist_cm) > 0

            nz_index_200ms = nonzero(elapsed_time_sec[i_netw, i_trial, nz_totdist] > 0.2)[0][0]
            nz_index_100ms = nonzero(elapsed_time_sec[i_netw, i_trial, nz_totdist] > 0.1)[0][0]

            # test for "jump" sequences:
            i,j = nonzero(center_mat_array[i_netw][i_trial] > -inf)
            tmp = zeros(len(i))
            for k in xrange(len(i)):
                tmp[k] = center_mat_array[i_netw][i_trial][i[k]][j[k]]
            imin=argmin(tmp)
            imax=argmax(tmp)


            total_seq_dist_cm[i_netw, i_trial] = L_maze_cm / float(center_len) * ( sqrt( (i[imax] - i[imin])**2 + (j[imax]-j[imin])**2 ) )
            #print "total_seq_dist_cm[i_netw, i_trial] = ", total_seq_dist_cm[i_netw, i_trial]
            min_delta_200plus = min(delta_total_dist_cm[i_netw, i_trial, nz_totdist[nz_index_200ms : ]])
            if total_seq_dist_cm[i_netw, i_trial] > 25: # 50
                prop_travelled_100ms[i_netw, i_trial] = total_dist_cm[i_netw, i_trial, nz_totdist[nz_index_100ms]] / nanmax(total_dist_cm[i_netw, i_trial, :])
                #if min_delta_200plus > 0:
                if prop_travelled_100ms[i_netw, i_trial] > 0.9:
                    #nan_centermat_jump[i_netw][i_trial][:][:] = center_mat_array[i_netw][i_trial][:][:]
                    for k in xrange(len(i)):
		                nan_centermat_jump[i_netw][i_trial][i[k]][j[k]] = 1
		                total_seq_dist_cm_jump[i_netw, i_trial] = total_seq_dist_cm[i_netw, i_trial]
		                array_speed_jump[i_netw, i_trial, :] = array_speed[i_netw, i_trial, :]
                    jumps += 1
                else:
    #                nan_centermat_nojump[i_netw][i_trial][:][:] = center_mat_array[i_netw][i_trial][:][:]
                    for k in xrange(len(i)):
		                nan_centermat_nojump[i_netw][i_trial][i[k]][j[k]] = 1
		                total_seq_dist_cm_nojump[i_netw, i_trial] = total_seq_dist_cm[i_netw, i_trial]
		                array_speed_nojump[i_netw, i_trial, :] = array_speed[i_netw, i_trial, :]
                    nojumps += 1
            else: # insufficent bump movement
                for k in xrange(len(i)):
                    nan_centermat_rest[i_netw][i_trial][i[k]][j[k]] = 1
                rest += 1

    print "jumps = ", jumps
    print "nojumps = ", nojumps
    print "rest = ", rest
    #print "max(total_seq_dist_cm) = ", nanmax(nanmax(total_seq_dist_cm))
    print "mean(total_seq_dist_cm_jump) = ", nanmean(nanmean(total_seq_dist_cm_jump))
    print "mean(total_seq_dist_cm_nojump) = ", nanmean(nanmean(total_seq_dist_cm_nojump))

    #print "min(total_seq_dist_cm_jump) = ", nanmin(nanmin(total_seq_dist_cm_jump))

    #figure(figsize=(3,3), dpi=dpisize)



    #'''#
    figure(figsize=(3,3), dpi=dpisize)
    subplot(221)
    i,j = nonzero( isnan(total_seq_dist_cm_jump) == 0 )
    rs1 = 0.01 * reshape(total_seq_dist_cm_jump[i,j], len(total_seq_dist_cm_jump[i,j]))
    hist(rs1, 30)
    title('dist, jump')
    subplot(222)
    i,j,k = nonzero( isnan(array_speed_jump) == 0 )
    rs2 = 0.01 * reshape(array_speed_jump[i,j,k], len(array_speed_jump[i,j,k]))
    hist(rs2, 30)
    title('speed, jump')
    subplot(223)
    i,j = nonzero( isnan(total_seq_dist_cm_nojump) == 0 )
    rs3 = 0.01 * reshape(total_seq_dist_cm_nojump[i,j], len(total_seq_dist_cm_nojump[i,j]))
    hist(rs3, 30)
    title('dist, no jump')
    subplot(224)
    i,j,k = nonzero( isnan(array_speed_nojump) == 0 )
    rs4 = 0.01 * reshape(array_speed_nojump[i,j,k], len(array_speed_nojump[i,j,k]))
    #hist(rs4, 30)
    thr = 200
    hist([rs4 * (rs4 <= thr) + (2*thr)*(rs4>thr)], 60)
    ax=gca()
    ax.set_xticks([0, thr, 2*thr])
    ax.set_xticklabels([0, thr, '>'+str(thr)], fontsize=8)
    title('speed, no jump')
    #'''
    ioff()
    show()


#rotated_plot = False # True # False # 

if rotated_plot:
        ion()
        rotated_home = zeros([2*center_len, 2*center_len]) # corresponds to an area of [- Lmaze_cm, Lmaze_cm]^2 
        rotated_away = zeros([2*center_len, 2*center_len]) # center_len = 100 (3.5cm - bins)
        alpha_all = zeros([n_valid_nets, nTrials])
        seqdist_all = zeros([n_valid_nets, nTrials])
        scaling_all = zeros([n_valid_nets, nTrials])
        k_end = zeros([n_valid_nets, nTrials])
        l_end = zeros([n_valid_nets, nTrials])

        for i_netw in xrange(n_valid_nets): 
            alpha = zeros(nTrials)

            for i_trial in xrange(nTrials): 

                if mod(i_trial, 2) == 0: # Home trial
                    i,j = nonzero(center_mat_array[i_netw][i_trial] > -inf)
                    sortinds = argsort(center_mat_array[i_netw][i_trial][i,j])
                    k = zeros(len(i))
                    l = zeros(len(i))
                    xs, ys = net_init.xypos_maze(seqstart[i_netw][i_trial][0], n_grid, L_maze_cm)
                    xh, yh = net_init.xypos_maze(goal_index_array[i_netw][0], n_grid, L_maze_cm)
                    xe, ye = net_init.xypos_maze(seqend[i_netw][i_trial][0], n_grid, L_maze_cm)
                    seq_dist = sqrt((xe-xs)**2 + (ye-ys)**2)
                    home_dist_start = sqrt((xh-xs)**2 + (yh-ys)**2)
                    if home_dist_start > 0:
                        alpha[i_trial] = arccos( (0 * (xh-xs) + 1*(yh-ys)) / (1*home_dist_start) )
                        scaling = (100 / home_dist_start)     
                    else:
                        alpha[i_trial] = 0
                        scaling = 0
                    if xh - xs > 0:
                        alpha[i_trial] = 2*pi - alpha[i_trial] 

                    alpha[i_trial] *= -1

                    for ind in xrange(len(i)):
                        k[ind] = cos(alpha[i_trial]) * (i[ind] - i[sortinds[0]]) - sin(alpha[i_trial]) * (j[ind] - j[sortinds[0]])
                        l[ind] = sin(alpha[i_trial]) * (i[ind] - i[sortinds[0]]) + cos(alpha[i_trial]) * (j[ind] - j[sortinds[0]])
                        k[ind] = round(k[ind] * scaling + center_len ) # 
                        l[ind] = round(l[ind] * scaling + center_len ) # 
                        if (ind==0 or sum([r==s for r,s in zip([k[:ind],l[:ind]], [k[ind], l[ind]])], 0).max() < 2): 
                                # Exclude "double" counts by comparing current [k[ind],l[ind]] to the whole [k,l] array

                                if max(k[ind], l[ind]) > 2*center_len - 1:
                                        print "Error: Values outside maze area, for i_netw= %i, i_trial= %i, ind= %i" %(i_netw, i_trial, ind)
                                else:
                                        rotated_home[int(k[ind]), int(l[ind])] = max(1, rotated_home[int(k[ind]), int(l[ind])] + 1) # for "-inf" init values

                else: # Random trial
                    i,j = nonzero(center_mat_array[i_netw][i_trial] > -inf)
                    sortinds = argsort(center_mat_array[i_netw][i_trial][i,j])
                    k = zeros(len(i))
                    l = zeros(len(i))
                    xs, ys = net_init.xypos_maze(seqstart[i_netw][i_trial][0], n_grid, L_maze_cm)
                    xe, ye = net_init.xypos_maze(seqend[i_netw][i_trial][0], n_grid, L_maze_cm)
                    seq_dist = sqrt((xe-xs)**2 + (ye-ys)**2)
                    if i_trial > 2:
                            xh, yh = net_init.xypos_maze(goal_index_array[i_netw][i_trial-2], n_grid, L_maze_cm)
                            prev_rand_dist_start = sqrt((xh-xs)**2 + (yh-ys)**2)
                            if prev_rand_dist_start > 0:
                                alpha[i_trial] =arccos( (0 * (xh-xs) + 1*(yh-ys)) / (1*prev_rand_dist_start) )
                                scaling = (100.0 / prev_rand_dist_start)     
                            else:
                                alpha[i_trial] = 0
                                scaling = 0    
                            if xh - xs > 0:
                                alpha[i_trial] = 2*pi - alpha[i_trial] 

                            alpha[i_trial] *= -1

                            imin = 0 ## Possibly remove the "start" points for ease of display
                            for ind in xrange(imin, len(i)):
                                k[ind] = cos(alpha[i_trial]) * (i[ind] - i[sortinds[0]]) - sin(alpha[i_trial]) * (j[ind] - j[sortinds[0]])
                                l[ind] = sin(alpha[i_trial]) * (i[ind] - i[sortinds[0]]) + cos(alpha[i_trial]) * (j[ind] - j[sortinds[0]])
                                k[ind] = min(round(k[ind] * scaling + center_len), 2*center_len -1)
                                l[ind] = min(round(l[ind] * scaling + center_len), 2*center_len -1)

                                if (ind==imin or sum([r==s for r,s in zip([k[:ind],l[:ind]], [k[ind], l[ind]])], 0).max() < 2): 
                                        # Exclude "double" counts by comparing current [k[ind],l[ind]] to the whole [k,l] array
                                        rotated_away[int(k[ind]), int(l[ind])] = max(1, rotated_away[int(k[ind]), int(l[ind])] + 1)
                alpha_all[i_netw, i_trial] = alpha[i_trial]
                seqdist_all[i_netw, i_trial] = seq_dist
                scaling_all[i_netw, i_trial] = scaling
                k_end[i_netw, i_trial] = k[sortinds[-1]]
                l_end[i_netw, i_trial] = l[sortinds[-1]]

        figure(figsize=(6,3), dpi=dpisize)
        subplot(121)

	rotated_home -= 0.1*(rotated_home <> 0.1)

        matshow(transpose((1+rotated_home[int(0.5*center_len) : int(1.5*center_len), int(0.5*center_len) : int(1.5*center_len)])), origin ='lower', fignum=False, norm=colors.LogNorm(), cmap=nc.inferno)
        ax = gca()
        center_resol = L_maze_cm / float(center_len)
        ax.set_xticks([0.5*center_len]) # [0, , center_len]
        ax.set_xticklabels(['$\longleftarrow$ Perpendicular to Home $\longrightarrow$'])
        ax.set_yticks([0.25*center_len, 0.75*center_len])
        ax.set_yticklabels(['Away from \n $\longleftarrow$  $\quad$ Home', 'Towards $\quad$ \n Home $\longrightarrow$'], rotation='vertical')
        ax.xaxis.set_ticks_position('bottom') #'''
        ax.xaxis.set_tick_params(length=0)
        ax.yaxis.set_tick_params(length=0)
        plot([0.5*center_len, 0.5*center_len], [0, center_len], '--', color='k', linewidth=0.5)
        plot([0, center_len], [0.5*center_len, 0.5*center_len], '--', color='k', linewidth=0.5)
        plot([0, center_len], [0.75*center_len, 0.75*center_len], '--', color='k', linewidth=0.5)

        for label in ax.get_xticklabels() + ax.get_yticklabels(): 
            label.set_fontsize(8)
        title('Home events rotated \n to Home well location', fontsize=8)



        subplot(122)
        matshow(transpose((1+rotated_away[int(0.5*center_len) : int(1.5*center_len), int(0.5*center_len) : int(1.5*center_len)])), origin ='lower', fignum=False, norm=colors.LogNorm(), cmap=nc.inferno)
        ax = gca()
        ax.set_xticks([0.5*center_len]) # [0, , center_len]
        ax.set_xticklabels(['$\longleftarrow$ Perpendicular to prev. Random $\longrightarrow$']) # [-0.5*center_len*center_resol, 0, 0.5*center_len*center_resol]
        ax.set_yticks([0.2*center_len, 0.8*center_len])
        ax.set_yticklabels(['$\quad$ Away from \n $\longleftarrow$ prev. Random' , 'Towards $\qquad \quad$ \n prev. Random $\longrightarrow$'], rotation='vertical')
        ax.xaxis.set_ticks_position('bottom') #'''
        ax.xaxis.set_tick_params(length=0)
        ax.yaxis.set_tick_params(length=0)
        plot([0.5*center_len, 0.5*center_len], [0, center_len], '--', color='k', linewidth=0.5)
        plot([0, center_len], [0.5*center_len, 0.5*center_len], '--', color='k', linewidth=0.5)
        plot([0, center_len], [0.75*center_len, 0.75*center_len], '--', color='k', linewidth=0.5)

        for label in ax.get_xticklabels() + ax.get_yticklabels(): 
            label.set_fontsize(8)
        title('Away events rotated \n  to previous random location', fontsize=8)
        axins = inset_axes(ax, width="5%", height="100%",loc=3, bbox_to_anchor=(1.05, 0., 1, 1), bbox_transform=ax.transAxes, borderpad=0)
        cbar = colorbar(cax=axins) #fraction = 0.05)
        cbar.set_ticks([1, 2, 11, 101, 1001])
        cbar.set_ticklabels([0, 1, 10, 100, 1000])

        cbar.ax.set_ylabel('Number of sequences', fontsize=8, rotation='vertical', position=(0, 0.5), va='bottom')
        axes(cbar.ax)
        yticks(fontsize=8)

        if savefigs:
            savefig('movement_data/'+'rotated_'+identifier+'_'+str(nTrials), dpi=dpisize_save)

        ioff()


if occupancy_plot:
    ion()
    i_netw = i_net_occ

    #'''#
    # Histogram "Remote representation of goal location":
    bincount = zeros(36)
    for j_trial in xrange(0, nTrials, 2): # Home trials
        i,j = nonzero(center_mat_array[i_netw][j_trial] > -inf)
        index = net_init.xy_index_maze(i-19, j-19, 6, 61) # shifting indices of center_mat_array by the "edges" of the maze
        for k in xrange(len(i)):
            bincount[int(index[k])] += 1

    xg,yg = net_init.xypos(goal_index_array[i_netw][0], n_grid)
    home_index = int(net_init.xy_index_maze(xg, yg, 6, n_grid))
    nonhome_bins = range(0, home_index) + range(home_index+1, 36)
    figure()
    #subplot(121)
    bar(nonhome_bins, bincount[nonhome_bins], color='k')
    bar(home_index, bincount[home_index], color='r')
    title('Total representation of Away events, binned into 6x6 subsections')
    #'''





    # Creating occupancy plots per trial:
    occ_mat_pertrial = -Inf * ones([nTrials, occ_len, occ_len])

    for i_trial in xrange(nTrials):
        i,j = nonzero(occupancyMap[i_netw][i_trial] < inf) # new
        for k in xrange(len(i)):
            occ_mat_pertrial[i_trial][i[k]][j[k]] = 1 # standard version
            #occ_mat_pertrial[i_trial][i[k]][j[k]] = occupancyMap[i_netw][i_trial][i[k]][j[k]] # test 9.3.15

    # Adding reward location to the occupancy map:
    for i_trial in xrange(nTrials):
        xg, yg = net_init.xypos_maze(goal_index_array[i_netw][i_trial], n_grid, occ_len)
        for i in xrange(occ_len):
            for j in xrange(occ_len):
                #if (i - xg)**2 + (j - yg)**2 < 20: # 25:
                if ( i > xg - 6.0 and i < xg + 6.0 and abs(j - yg) > 4.0 and abs(j - yg) < 6.0 ) \
                        or ( abs(i - xg) < 6.0 and abs(i - xg) > 4.0 and j > yg - 6.0 and j < yg + 6.0 ) :
                    #occ_mat_pertrial[i_trial][i,j] = max(occ_mat_pertrial[i_trial][i,j], nTrials + 1)
                    occ_mat_pertrial[i_trial][i,j] = 0.75 # 1
        scale = float(L_maze_cm) / len(center_mat_array[0][0][0,:])

        xg, yg = net_init.xypos_maze(goal_index_array[i_netw][i_trial], n_grid, center_len)
        for i in xrange(center_len):
            for j in xrange(center_len):
                if ( i > xg - 3.0 and i < xg + 3.0 and abs(j - yg) > 2.0 and abs(j - yg) < 3.0 ) \
                        or ( abs(i - xg) < 3.0 and abs(i - xg) > 2.0 and j > yg - 3.0 and j < yg + 3.0 ) :

                    # Test: Add reward location also to sequence plots
                    center_mat_array[i_netw][i_trial][i,j] = max(center_mat_array[i_netw][i_trial][i,j], center_mat_array[i_netw][i_trial].max())


    figure(figsize=(4,4), dpi=dpisize)
    for i_trial in xrange(min(nTrials, 12)):
        subplot(3, 4, i_trial + 1)
        n_edge = int( ceil(maze_edge_cm / float(L_maze_cm) * occ_len) )
        matshow(transpose(occ_mat_pertrial[i_trial][n_edge : occ_len - n_edge, n_edge : occ_len - n_edge]), origin='lower', fignum = False, cmap=cm.Spectral)
        ax = gca()
        ax.set_xticks([])
        ax.set_xticklabels([])
        ax.set_yticks([])
        ax.set_yticklabels([])
        ax.xaxis.set_ticks_position('bottom')
        if mod(i_trial, 2) == 0:
            #title('Trial '+str(int(i_trial/2.0)+1)+', Home')
            title('Trial '+str(i_trial+1) + ', Home', fontsize=8)
        else:
            #title('Trial '+str(int((i_trial-1)/2.0)+1)+', Away')
            title('Trial '+str(i_trial+1) + ', Random', fontsize=8)

        if i_trial==8:
            xlabel('x position [m]', fontsize=8)
            ylabel('y position [m]', fontsize=8)
            ax.set_xticks([0, occ_len - 2*n_edge])
            ax.set_xticklabels([0, int(0.01*(L_maze_cm - 2*maze_edge_cm))])
            ax.set_yticks([0, occ_len - 2*n_edge])
            ax.set_yticklabels([0, int(0.01*(L_maze_cm - 2*maze_edge_cm))])
            for label in ax.get_xticklabels() + ax.get_yticklabels(): 
                label.set_fontsize(8) 

    suptitle('Per-trial occupancy maps, Network '+str(i_netw), fontsize=8)    
    if savefigs:
        savefig('movement_data/'+'occupancy_pertrial_'+identifier+'_'+str(nTrials), dpi=dpisize_save)

    '''#figure(figsize=(3,3), dpi=dpisize)
    # Creating plot of bump movement (all trials):    
#    figure()

    for i in xrange(100):
        for j in xrange(100):
            if mod(center_mat_plot[i_netw][i,j], 2) == 0:
                center_mat_plot_Home[i_netw][i,j] = center_mat_plot[i_netw][i,j]
            elif mod(center_mat_plot[i_netw][i,j], 2) == 1:
                center_mat_plot_Away[i_netw][i,j] = center_mat_plot[i_netw][i,j]

    subplot(1,2,1)
    #matshow(transpose(center_mat_plot_Home[i_netw]), origin='lower', fignum=False)
    n_edge = ceil(maze_edge_cm / float(L_maze_cm) * center_len)
    matshow(transpose(center_mat_plot_Home[i_netw][n_edge : center_len - n_edge, n_edge : center_len - n_edge]), origin='lower', fignum=False, cmap=cm.YlOrRd)

    ax = gca()
    #cbar = colorbar(shrink = 0.4)
    #cbar.set_ticks([cbar.vmin, cbar.vmax])
    axins = inset_axes(ax, width="5%", height="100%",loc=3, bbox_to_anchor=(1.05, 0., 1, 1), bbox_transform=ax.transAxes, borderpad=0)
    cbar = colorbar(cax=axins) #fraction = 0.05)
    cbar.set_ticks(xrange(0, nTrials, 2))
    cbar.set_ticklabels(xrange(1, nTrials+1, 2))
    axes(cbar.ax)
    yticks(fontsize=8)
    title('Trial', fontsize=8)
    axes(ax)

    ax = gca()
    ax.set_xticks([0, center_len - 2*n_edge])
    ax.set_xticklabels([0, int(0.01*(L_maze_cm - 2*maze_edge_cm))])
    ax.set_yticks([0, center_len - 2*n_edge])
    ax.set_yticklabels([0, int(0.01*(L_maze_cm - 2*maze_edge_cm))])
    ax.xaxis.set_ticks_position('bottom')
    for label in ax.get_xticklabels() + ax.get_yticklabels(): 
        label.set_fontsize(8)
    xlabel('x position [m]', fontsize=8)
    ylabel('y position [m]', fontsize=8)
    title('Home trials, network '+str(i_netw), fontsize=8) # Bump movement for 

    subplot(1,2,2)
    #matshow(transpose(center_mat_plot_Away[i_netw]), origin='lower', fignum=False)
    n_edge = ceil(maze_edge_cm / float(L_maze_cm) * center_len)
    matshow(transpose(center_mat_plot_Away[i_netw][n_edge : center_len - n_edge, n_edge : center_len - n_edge]), origin='lower', fignum=False, cmap=cm.YlOrRd)
    ax = gca()
    #cbar = colorbar(shrink = 0.4)
    #cbar.set_ticks([cbar.vmin, cbar.vmax])
    axins = inset_axes(ax, width="5%", height="100%",loc=3, bbox_to_anchor=(1.05, 0., 1, 1), bbox_transform=ax.transAxes, borderpad=0)
    cbar = colorbar(cax=axins) #fraction = 0.05)
    cbar.set_ticks(xrange(1, nTrials+1, 2))
    cbar.set_ticklabels(xrange(2, nTrials+1, 2))
    axes(cbar.ax)
    yticks(fontsize=8)
    title('Trial', fontsize=8)
    axes(ax)

    ax = gca()
    #ax.set_xticks([0, center_len - 2*n_edge])
    #ax.set_xticklabels([0, L_maze_cm - 2*maze_edge_cm])
    #ax.set_yticks([0, center_len - 2*n_edge])
    #ax.set_yticklabels([0, L_maze_cm - 2*maze_edge_cm])
    #ax.xaxis.set_ticks_position('bottom')
    ax.set_xticks([])
    ax.set_xticklabels([])
    ax.set_yticks([])
    ax.set_yticklabels([])

    #xlabel('x position [m]')
    #ylabel('y position [m]')
    title('Random trials, network '+str(i_netw), fontsize=8) # Bump movement for Away
    if savefigs:
        savefig('movement_data/'+'bump_movement_alltrials_'+identifier+'_'+str(nTrials), dpi=dpisize_save)
    '''


    # Creating plot of bump movement across trials:
    figure(figsize=(4,4), dpi=dpisize)
    for i_trial in xrange(min(nTrials, 12)):
        subplot(3, 4, i_trial + 1)
	n_edge = int(ceil(maze_edge_cm / float(L_maze_cm) * center_len))
	matshow(transpose(center_mat_array[i_netw][i_trial][n_edge : center_len - n_edge, n_edge : center_len - n_edge]), origin='lower', fignum=False, cmap=nc.inferno_r)

        ax = gca()
        if i_trial == min(nTrials, 12)-1:        
                    axins = inset_axes(ax, width="5%", height="100%",loc=3, bbox_to_anchor=(1.05, 0., 1, 1), bbox_transform=ax.transAxes, borderpad=0)
                    cbar = colorbar(cax=axins) #fraction = 0.05)
                    cbar.set_ticks([cbar.vmin, cbar.vmax])
                    cbar.set_ticklabels([0, cbar.vmax - cbar.vmin])
                    axes(cbar.ax)
                    yticks(fontsize=8)
                    title('t [s]', fontsize=8)
                    axes(ax)
	ax.set_xticks([])
	ax.set_xticklabels([])
	ax.set_yticks([])
	ax.set_yticklabels([])
        ax.xaxis.set_ticks_position('bottom')
        if mod(i_trial, 2) == 0:
            #title('Trial '+str(int(i_trial/2.0)+1)+', Home')
            title('Trial '+str(i_trial+1)+', Home', fontsize=8)
        else:
            #title('Trial '+str(int((i_trial-1)/2.0)+1)+', Away')
            title('Trial '+str(i_trial+1)+', Random', fontsize=8)
        if i_trial==8:
            ax.set_xticks([0, center_len - 2*n_edge - 1])
            ax.set_xticklabels([0, int(0.01*(L_maze_cm - 2*maze_edge_cm))])
            ax.set_yticks([0, center_len - 2*n_edge - 1])
            ax.set_yticklabels([0, int(0.01*(L_maze_cm - 2*maze_edge_cm))])
            xlabel('x position [m]', fontsize=8)
            ylabel('y position [m]', fontsize=8)
            for label in ax.get_xticklabels() + ax.get_yticklabels(): 
                label.set_fontsize(8)
    subplots_adjust(wspace=.01) # hspace=.05, 
    suptitle('Bump movement across trials, Network '+str(i_netw), fontsize=8)   
    #show()
    if savefigs:
        savefig('movement_data/'+'bump_movement_pertrial_'+identifier+'_'+str(nTrials), dpi=dpisize_save)


    # Creating a combined plot of bump movement, trajectory and weights: 
    combined_plot_Both = 0 # 1
    combined_plot_Home = 1 # 0
    combined_Both_singlepics = 0

    if combined_Both_singlepics:
            # plot:
            # - initial weights for Home and Away trials,
            # - sequence trajectories for the first two Home and Away trials,
            # - movement trajectories for the first two Home and Away trials,
            # - modified LEC-DG weights for the first two Home and Away trials

            n_edge_w = int(ceil(maze_edge_cm / float(L_maze_cm) * n_grid))
            disp_trials = 4

            # plot initial weight matrices:
            figure(figsize=(4,4), dpi=dpisize)
            matshow(0.15*rand(n_grid - 2*n_edge_w, n_grid - 2*n_edge_w), origin='lower', fignum=False, vmax = 1, cmap=nc.viridis)
	    #axis('off')
            ax=gca()
            ax.set_xticks([])
            ax.set_xticklabels([])
            ax.set_yticks([])
            ax.set_yticklabels([])

            if savefigs:
                savefig('movement_data/'+'single_w_home_init_'+identifier+'_'+str(nTrials), dpi=dpisize_save, bbox_inches='tight', transparent="True")

            figure(figsize=(4,4), dpi=dpisize)
            matshow(0.15*rand(n_grid - 2*n_edge_w, n_grid - 2*n_edge_w), origin='lower', fignum=False, cmap=cm.binary, vmax = 1) # modified 28.1.16 for current init. weight
	    #axis('off')
            ax=gca()
            ax.set_xticks([])
            ax.set_xticklabels([])
            ax.set_yticks([])
            ax.set_yticklabels([])

            if savefigs:
                savefig('movement_data/'+'single_w_away_init_'+identifier+'_'+str(nTrials), dpi=dpisize_save, bbox_inches='tight', transparent="True")

            # Weights, trajectories and bump movement across trials:
            for i_trial in xrange(min(nTrials, disp_trials)):
                # Weights
                placeValueMat = zeros([n_grid, n_grid])
                if mod(i_trial, 2) == 0:
                        # Home trials:
                        for iNeuron in xrange(len(weight_array_home[i_netw][i_trial])):
                            x,y = net_init.xypos(iNeuron, n_grid)
                            placeValueMat[x, y] = weight_array_home[i_netw][i_trial][iNeuron] / 1.0e-9
                        figure(figsize=(4,4), dpi=dpisize)
                        matshow(transpose(placeValueMat[n_edge_w : n_grid - n_edge_w, n_edge_w : n_grid - n_edge_w]), origin='lower', fignum=False, vmax = 1, cmap=nc.viridis)
	                #axis('off')
                        ax=gca()
                        ax.set_xticks([])
                        ax.set_xticklabels([])
                        ax.set_yticks([])
                        ax.set_yticklabels([])
                        if savefigs:
                            savefig('movement_data/'+'single_w_home_trial_'+str(i_trial)+identifier+'_'+str(nTrials), dpi=dpisize_save, bbox_inches='tight', transparent="True")

                else:
                        # Away trials:
                        for iNeuron in xrange(len(weight_array_away[i_netw][i_trial])):
                            x,y = net_init.xypos(iNeuron, n_grid)
                            placeValueMat[x, y] = weight_array_away[i_netw][i_trial][iNeuron] / 1.0e-9
                        figure(figsize=(4,4), dpi=dpisize)
                        matshow(transpose(placeValueMat[n_edge_w : n_grid - n_edge_w, n_edge_w : n_grid - n_edge_w]), origin='lower', fignum=False, cmap=cm.binary, vmax=1) # binary or terrain colormap might be acceptable
	                #axis('off')
                        ax=gca()
                        ax.set_xticks([])
                        ax.set_xticklabels([])
                        ax.set_yticks([])
                        ax.set_yticklabels([])

                        if savefigs:
                            savefig('movement_data/'+'single_w_away_trial_'+str(i_trial)+identifier+'_'+str(nTrials), dpi=dpisize_save, bbox_inches='tight', transparent="True")

                # Trajectory
                figure(figsize=(4,4), dpi=dpisize)
                n_edge_path = int(ceil(maze_edge_cm / float(L_maze_cm) * occ_len))
                matshow(transpose(occ_mat_pertrial[i_trial][n_edge_path : occ_len - n_edge_path, n_edge_path : occ_len - n_edge_path]), origin='lower', fignum = False, cmap=cm.Spectral)
                ax=gca()
                ax.set_xticks([])
                ax.set_xticklabels([])
                ax.set_yticks([])
                ax.set_yticklabels([])

                if savefigs:
                    if mod(i_trial, 2) == 0:
                    # Home trials:
                        savefig('movement_data/'+'single_path_home_trial_'+str(i_trial)+identifier+'_'+str(nTrials), dpi=dpisize_save, bbox_inches='tight', transparent="True")
                    else:
                    # Away trials:
                        savefig('movement_data/'+'single_path_away_trial_'+str(i_trial)+identifier+'_'+str(nTrials), dpi=dpisize_save, bbox_inches='tight', transparent="True")

                # Bump movement            
        	n_edge_seq = int(ceil(maze_edge_cm / float(L_maze_cm) * center_len))
                figure(figsize=(4,4), dpi=dpisize)
                matshow(transpose(center_mat_array[i_netw][i_trial][n_edge_seq : center_len - n_edge_seq, n_edge_seq : center_len - n_edge_seq]), origin='lower', fignum=False, cmap=nc.inferno_r) # 
                ax=gca()
                ax.set_xticks([])
                ax.set_xticklabels([])
                ax.set_yticks([])
                ax.set_yticklabels([])

                if savefigs:
                    if mod(i_trial, 2) == 0:
                    # Home trials:
                        savefig('movement_data/'+'single_seq_home_trial_'+str(i_trial)+identifier+'_'+str(nTrials), dpi=dpisize_save, bbox_inches='tight', transparent="True")
                    else:
                    # Away trials:
                        savefig('movement_data/'+'single_seq_away_trial_'+str(i_trial)+identifier+'_'+str(nTrials), dpi=dpisize_save, bbox_inches='tight', transparent="True")


    if combined_plot_Both:
            figure(figsize=(4,4), dpi=dpisize)
            disp_trials = 10 ## 7 # 4 # 10 # 6 # 5 #
            wmax = 0
            wmax

            for i_trial in xrange( min(nTrials, disp_trials)-1, -1, -1): # Reversed order!
                # Bump movement
                subplot(3, disp_trials, i_trial + 1)
                n_edge = int(ceil(maze_edge_cm / float(L_maze_cm) * center_len))
                matshow(transpose(center_mat_array[i_netw][i_trial][n_edge : center_len - n_edge, n_edge : center_len - n_edge]), origin='lower', fignum=False, cmap=nc.inferno_r)
                ax = gca()
                ax.set_xticks([0, center_len - 2*n_edge])
                ax.set_xticklabels([0, int(0.01*(L_maze_cm - 2*maze_edge_cm))])
                ax.set_yticks([0, center_len - 2*n_edge])
                ax.set_yticklabels([0, int(0.01*(L_maze_cm - 2*maze_edge_cm))])
                ax.xaxis.set_ticks_position('bottom')
                for label in ax.get_xticklabels() + ax.get_yticklabels(): 
                    label.set_fontsize(8)

                if i_trial==0:
                    xlabel('x position [m]', fontsize=8)
                    ylabel('y position [m]', fontsize=8)
                else:
                    ax.set_xticks([])
                    ax.set_xticklabels([])
                    ax.set_yticks([])
                    ax.set_yticklabels([])

                if i_trial == disp_trials - 1:
                    axins = inset_axes(ax, width="5%", height="100%",loc=3, bbox_to_anchor=(1.05, 0., 1, 1), bbox_transform=ax.transAxes, borderpad=0)
                    cbar = colorbar(cax=axins) #fraction = 0.05)
                    cbar.set_ticks([cbar.vmin, cbar.vmax])
                    cbar.set_ticklabels([0, cbar.vmax - cbar.vmin])
                    axes(cbar.ax)
                    yticks(fontsize=8)
                    title('t [s]', fontsize=8)
                    axes(ax)


                if mod(i_trial, 2) == 0:
                    title('Trial '+str(int(i_trial/2.0)+1)+', Home', fontsize=8)
                else:
                    title('Trial '+str(int((i_trial-1)/2.0)+1)+', Away', fontsize=8)
                if i_trial==1:
                    xlabel('x position [cm]', fontsize=8)
                    ylabel('y position [cm]', fontsize=8)

                # Trajectory
                subplot(3, disp_trials, i_trial + 1 + disp_trials)
                n_edge = int(ceil(maze_edge_cm / float(L_maze_cm) * occ_len))

                #n_edge = ceil(maze_edge_cm / float(L_maze_cm) * occ_len)
                matshow(transpose(occ_mat_pertrial[i_trial][n_edge : occ_len - n_edge, n_edge : occ_len - n_edge]), origin='lower', fignum = False, cmap=cm.Spectral)
                ax=gca()
                ax.set_xticks([0, occ_len - 2*n_edge])
                ax.set_xticklabels([0, int(0.01*(L_maze_cm - 2*maze_edge_cm))])
                ax.set_yticks([0, occ_len - 2*n_edge])
                ax.set_yticklabels([0, int(0.01*(L_maze_cm - 2*maze_edge_cm))])
                ax.xaxis.set_ticks_position('bottom')
                for label in ax.get_xticklabels() + ax.get_yticklabels(): 
                    label.set_fontsize(8)

                if i_trial==0:
                    xlabel('x position [m]', fontsize=8)
                    ylabel('y position [m]', fontsize=8)
                else:
                    ax.set_xticks([])
                    ax.set_xticklabels([])
                    ax.set_yticks([])
                    ax.set_yticklabels([])

                # Weights
                placeValueMat = zeros([n_grid, n_grid])
                n_edge = int(ceil(maze_edge_cm / float(L_maze_cm) * n_grid))

                if mod(i_trial, 2) == 0:
                    for iNeuron in xrange(len(weight_array_home[i_netw][i_trial])):
                        x,y = net_init.xypos(iNeuron, n_grid)
                        placeValueMat[int(x), int(y)] = weight_array_home[i_netw][i_trial][iNeuron] / 1.0e-9

                    if placeValueMat.max() > wmax:
                        wmax = placeValueMat.max()

                    subplot(3, disp_trials, i_trial + 1 + 2*disp_trials)
                    matshow(transpose(placeValueMat[n_edge : n_grid - n_edge, n_edge : n_grid - n_edge]), origin='lower', fignum=False, vmax=wmax, cmap=nc.viridis)
                else:
                    for iNeuron in xrange(len(weight_array_away[i_netw][i_trial])):
                        x,y = net_init.xypos(iNeuron, n_grid)
                        placeValueMat[int(x), int(y)] = weight_array_away[i_netw][i_trial][iNeuron] / 1.0e-9

                    if placeValueMat.max() > wmax:
                        wmax = placeValueMat.max()

                    subplot(3, disp_trials, i_trial + 1 + 2*disp_trials)
                    matshow(transpose(placeValueMat[n_edge : n_grid - n_edge, n_edge : n_grid - n_edge]), origin='lower', fignum=False, cmap=cm.binary, vmax=wmax) 

                ax = gca()
                ax.set_xticks([0, n_grid - 2*n_edge - 1])
                ax.set_xticklabels([0, int(0.01*(L_maze_cm - 2*maze_edge_cm))])
                ax.set_yticks([0, n_grid - 2*n_edge - 1])
                ax.set_yticklabels([0, int(0.01*(L_maze_cm - 2*maze_edge_cm))])
                for label in ax.get_xticklabels() + ax.get_yticklabels(): 
                    label.set_fontsize(8)

                if i_trial==0:
                    xlabel('x position [m]', fontsize=8)
                    ylabel('y position [m]', fontsize=8)

                if i_trial == min(nTrials, disp_trials) - 1 or i_trial == min(nTrials, disp_trials) - 2:
                    axins = inset_axes(ax, width="5%", height="100%",loc=3, bbox_to_anchor=(1.05, 0., 1, 1), bbox_transform=ax.transAxes, borderpad=0)
                    cbar = colorbar(cax=axins)
                    cbar.set_ticks([cbar.vmin, round(cbar.vmax, 3)])
                    cbar.set_ticklabels([round(cbar.vmin, 3), round(cbar.vmax, 3)])
                    axes(cbar.ax)
                    yticks(fontsize=8)
                    title('w [nA]', fontsize=8)
                    axes(ax)

                if mod(i_trial, 2) == 1: # Away trial
		            filename_w = 'data/wdata_cont_away_DG_' + str(nTrials) + '_' + identifier + '_netw_'+str(i_netw)+'_trial_'+str(2*i_trial)
		            file = open(filename_w, 'w'); pickle.dump(weight_array_away[i_netw][i_trial],file,0); file.close()
		            print "Saving weights to file ", filename_w



            suptitle('Bump movement, trajectory and weights across trials, Network '+str(i_netw), fontsize=8)   
            if savefigs:
                savefig('movement_data/'+'combined_plot_Both_'+identifier+'_'+str(nTrials), dpi=dpisize_save)


    if combined_plot_Home:
            figure(figsize=(4,4), dpi=dpisize)
            disp_trials = 4 # 2 #  6 # 5 #
            wmax = 0

            for i_trial in xrange( min(int(int( ceil(nTrials/2.0) )), disp_trials)-1, -1, -1): # Reversed order!
                # Bump movement

                subplot(3, disp_trials, i_trial + 1)

                title('Trial '+str(2*int(i_trial)+1), fontsize=8)

                n_edge = int(ceil(maze_edge_cm / float(L_maze_cm) * center_len))
                matshow(transpose(center_mat_array_Home[i_netw][i_trial][n_edge : center_len - n_edge, n_edge : center_len - n_edge]), origin='lower', fignum=False, cmap=cm.YlOrRd)
                ax = gca()
                ax.set_xticks([0, center_len - 2*n_edge])
                ax.set_xticklabels([0, int(0.01*(L_maze_cm - 2*maze_edge_cm))])
                ax.set_yticks([0, center_len - 2*n_edge])
                ax.set_yticklabels([0, int(0.01*(L_maze_cm - 2*maze_edge_cm))])

                ax.xaxis.set_ticks_position('bottom')
                for label in ax.get_xticklabels() + ax.get_yticklabels(): 
                    label.set_fontsize(8)

                if i_trial==0:
                    xlabel('x position [m]', fontsize=8)
                    ylabel('y position [m]', fontsize=8)
                else:
                    ax.set_xticks([])
                    ax.set_xticklabels([])
                    ax.set_yticks([])
                    ax.set_yticklabels([])

                if i_trial == min( int(int( ceil(nTrials/2.0) )), disp_trials) - 1:
                    axins = inset_axes(ax, width="5%", height="100%",loc=3, bbox_to_anchor=(1.05, 0., 1, 1), bbox_transform=ax.transAxes, borderpad=0)
                    cbar = colorbar(cax=axins) #fraction = 0.05)
                    cbar.set_ticks([cbar.vmin, cbar.vmax])
                    cbar.set_ticklabels([0, cbar.vmax - cbar.vmin])
                    axes(cbar.ax)
                    yticks(fontsize=8)
                    title('t [s]', fontsize=8)
                    axes(ax)


                # Trajectory

                subplot(3, disp_trials, i_trial + 1 + disp_trials)

                n_edge = int(ceil(maze_edge_cm / float(L_maze_cm) * occ_len))
                matshow(transpose(occ_mat_pertrial[2 * i_trial][n_edge : occ_len - n_edge, n_edge : occ_len - n_edge]), origin='lower', fignum = False)
                ax = gca()
                ax.set_xticks([0, occ_len - 2*n_edge])
                ax.set_xticklabels([0, int(0.01*(L_maze_cm - 2*maze_edge_cm))])
                ax.set_yticks([0, occ_len - 2*n_edge])
                ax.set_yticklabels([0, int(0.01*(L_maze_cm - 2*maze_edge_cm))])
                ax.xaxis.set_ticks_position('bottom')
                for label in ax.get_xticklabels() + ax.get_yticklabels(): 
                    label.set_fontsize(8)

                if i_trial==0:
                    xlabel('x position [m]', fontsize=8)
                    ylabel('y position [m]', fontsize=8)
                else:
                    ax.set_xticks([])
                    ax.set_xticklabels([])
                    ax.set_yticks([])
                    ax.set_yticklabels([])

                # Weights:
                # Initial weights
                '''#
                placeValueMat = 1.0e-9 * rand(n_grid, n_grid) # zeros([n_grid, n_grid])
                if placeValueMat.max() > wmax:
                    wmax = placeValueMat.max()
                #subplot(3, disp_trials, i_trial + 1 + 2*disp_trials)
                #subplot(disp_trials, 4, i_trial*4 + 1 + 0)
                subplot(1, 7, 1)
                n_edge = ceil(maze_edge_cm / float(L_maze_cm) * n_grid)
                matshow(transpose(placeValueMat[n_edge : n_grid - n_edge, n_edge : n_grid - n_edge]), origin='lower', fignum=False, vmax=wmax)
                ax = gca()
                ax.set_xticks([0, n_grid - 2*n_edge - 1])
                ax.set_xticklabels([0, int(0.01*(L_maze_cm - 2*maze_edge_cm))])
                ax.set_yticks([0, n_grid - 2*n_edge - 1])
                ax.set_yticklabels([0, int(0.01*(L_maze_cm - 2*maze_edge_cm))])
                ax.xaxis.set_ticks_position('bottom')
                for label in ax.get_xticklabels() + ax.get_yticklabels(): 
                    label.set_fontsize(8)
                if i_trial==0:
                    xlabel('x position [m]', fontsize=8)
                    ylabel('y position [m]', fontsize=8)
                else:
                    ax.set_xticks([])
                    ax.set_xticklabels([])
                    ax.set_yticks([])
                    ax.set_yticklabels([])                
                if i_trial == min( int(int( ceil(nTrials/2.0) )), disp_trials) - 1:
                    axins = inset_axes(ax, width="5%", height="100%",loc=3, bbox_to_anchor=(1.05, 0., 1, 1), bbox_transform=ax.transAxes, borderpad=0)
                    cbar = colorbar(cax=axins) #fraction = 0.05)
                    cbar.set_ticks([cbar.vmin, round(cbar.vmax, 3)])
                    cbar.set_ticklabels([round(cbar.vmin, 3), round(cbar.vmax, 3)])
                    axes(cbar.ax)
                    yticks(fontsize=8)
                    title('w [nA]', fontsize=8)
                    axes(ax)
                '''                


                # Weights at the end of the trial
                placeValueMat = zeros([n_grid, n_grid])
                for iNeuron in xrange(len(weight_array_home[i_netw][i_trial])):
                        x,y = net_init.xypos(iNeuron, n_grid)
                        placeValueMat[int(x), int(y)] = weight_array_home[i_netw][2*i_trial][iNeuron] / 1.0e-9
                if placeValueMat.max() > wmax:
                    wmax = placeValueMat.max()
                subplot(3, disp_trials, i_trial + 1 + 2*disp_trials)
                #subplot(disp_trials, 4, i_trial*4 + 1 + 3)
                #if i_trial == 0:
                #    subplot(1, 7, 4)
                #else:
                #    subplot(1, 7, 7)
                n_edge = int(ceil(maze_edge_cm / float(L_maze_cm) * n_grid))
                matshow(transpose(placeValueMat[n_edge : n_grid - n_edge, n_edge : n_grid - n_edge]), origin='lower', fignum=False, vmax=wmax)
                ax = gca()
                ax.set_xticks([0, n_grid - 2*n_edge - 1])
                ax.set_xticklabels([0, int(0.01*(L_maze_cm - 2*maze_edge_cm))])
                ax.set_yticks([0, n_grid - 2*n_edge - 1])
                ax.set_yticklabels([0, int(0.01*(L_maze_cm - 2*maze_edge_cm))])
                ax.xaxis.set_ticks_position('bottom')
                for label in ax.get_xticklabels() + ax.get_yticklabels(): 
                    label.set_fontsize(8)
                if i_trial==0:
                    xlabel('x position [m]', fontsize=8)
                    ylabel('y position [m]', fontsize=8)
                else:
                    ax.set_xticks([])
                    ax.set_xticklabels([])
                    ax.set_yticks([])
                    ax.set_yticklabels([])                
                if i_trial == min( int(int( ceil(nTrials/2.0) )), disp_trials) - 1:
                    axins = inset_axes(ax, width="5%", height="100%",loc=3, bbox_to_anchor=(1.05, 0., 1, 1), bbox_transform=ax.transAxes, borderpad=0)
                    cbar = colorbar(cax=axins) #fraction = 0.05)
                    cbar.set_ticks([cbar.vmin, round(cbar.vmax, 3)])
                    cbar.set_ticklabels([round(cbar.vmin, 3), round(cbar.vmax, 3)])
                    axes(cbar.ax)
                    yticks(fontsize=8)
                    title('w [nA]', fontsize=8)
                    axes(ax)


                filename_w = 'data/wdata_cont_home_DG_' + str(nTrials) + '_' + identifier + '_netw_'+str(i_netw)+'_trial_'+str(2*i_trial)
                file = open(filename_w, 'w'); pickle.dump(weight_array_home[i_netw][2*i_trial],file,0); file.close()
                print "Saving weights to file ", filename_w


            #subplots_adjust(hspace=.05, wspace=.01)
            #tight_layout()            

            suptitle('Bump movement, trajectory and weights across Home trials, Network '+str(i_netw), fontsize=8)   
            if savefigs:
                #savefig('movement_data/'+'combined_plot_Home_'+identifier+'_'+str(nTrials), dpi=dpisize_save)
                savefig('movement_data/'+'combined_plot_Home_'+identifier+'_'+str(nTrials)+'_netw'+str(i_netw), dpi=dpisize_save)




    ioff()
    #show()



if figs_for_movie:

    figure(figsize=(12,6))

    #n_edge = ceil(maze_edge_cm / float(L_maze_cm) * n_grid)
    n_edge = ceil(maze_edge_cm / float(L_maze_cm) * center_len)

    n_edge_w = ceil(maze_edge_cm / float(L_maze_cm) * n_grid)

    ioff()
    i_trial = 3 # 1 # 4 # 3 # 2 # 1 # 0
    i_netw = 9 # 

    print "Sequence: i_netw, i_trial =", i_netw, i_trial

    wmax_home = 0
    for k_trial in xrange(4): # 5
            placeValueMat = zeros([n_grid, n_grid])
            for iNeuron in xrange(len(weight_array_home[i_netw][k_trial])):
                x,y = net_init.xypos(iNeuron, n_grid)
                placeValueMat[x, y] = weight_array_home[i_netw][2*k_trial][iNeuron]
            if placeValueMat.max() > wmax_home:
                wmax_home = placeValueMat.max()
    if i_trial > 0:
            placeValueMat = zeros([n_grid, n_grid])
            for iNeuron in xrange(len(weight_array_home[i_netw][i_trial])):
                x,y = net_init.xypos(iNeuron, n_grid)
                placeValueMat[x, y] = weight_array_home[i_netw][2*(i_trial-1)][iNeuron]
    else:
            placeValueMat = 1.0e-10 * rand(n_grid, n_grid)

    # Place-cell sequence:
    i,j = nonzero(center_mat_array_Home[i_netw][i_trial] > -inf)
    tmp = zeros(len(i))
    for k in xrange(len(i)):
        tmp[k] = center_mat_array_Home[i_netw][i_trial][i[k]][j[k]]
    i_sorted=argsort(tmp)
    #hugemat = -inf*ones([len(i), 200, 200])
    hugemat = -inf*ones([len(i), center_len, center_len])
    for k in xrange(len(i)):                                                   
        for l in xrange(k):
            #hugemat[k, i[l], j[l]] = center_mat_array_Home[0][i_trial][i[l]][j[l]]
            hugemat[k, i[i_sorted[l]], j[i_sorted[l]]] = center_mat_array_Home[i_netw][i_trial][i[i_sorted[l]]][j[i_sorted[l]]]
    seq_counter = 0

    suptitle('Trial '+str(int(i_trial)+1))

    for k in xrange(len(i)):
         if mod(k, 10)==0:
            print "Bin %i of %i"%(k, len(i))

         seq_counter += 1
         subplot(1,3,1)
         #matshow(transpose(hugemat[k,:,:]), origin='lower', cmap=cm.YlOrRd) #, fignum=False)
         matshow(transpose(hugemat[k, n_edge : center_len - n_edge, n_edge : center_len - n_edge]), origin='lower', fignum=False) #, vmax = wmax_home)
         ax = gca()
         '''#ax.set_xticks([0, 199])
         ax.set_xticklabels([0, 2])
         ax.set_yticks([0, 199])
         ax.set_yticklabels([0, 2])'''
         ax.set_xticks([0, center_len - 2*n_edge])
         ax.set_xticklabels([0, int(0.01*(L_maze_cm - 2*maze_edge_cm))])
         ax.set_yticks([0, center_len - 2*n_edge])
         ax.set_yticklabels([0, int(0.01*(L_maze_cm - 2*maze_edge_cm))])

         ax.xaxis.set_ticks_position('bottom')
         xlabel('x [m]')
         ylabel('y [m]')
         #title('Place-cell sequence, Trial '+str(i_trial_temp+1))
         title('Place-cell sequence')

         subplot(1,3,3)
         matshow(transpose(placeValueMat[n_edge_w : n_grid - n_edge_w, n_edge_w : n_grid - n_edge_w]), origin='lower', vmax = wmax_home, fignum=False)
         title('LEC-DG weights')
         ax = gca()
         ax.set_xticks([0, n_grid - 2*n_edge_w - 1])
         ax.xaxis.set_ticks_position('bottom')
         ax.set_xticklabels([0, int(0.01*(L_maze_cm - 2*maze_edge_cm))])
         ax.set_yticks([0, n_grid - 2*n_edge_w - 1])
         ax.set_yticklabels([0, int(0.01*(L_maze_cm - 2*maze_edge_cm))])
         xlabel('x [m]')
         ylabel('y [m]')


         #savefig('img_movie/sequence_'+str(k))
         savefig('img_movie/combined_'+str(k))
         if k < len(i)-1:
             close()

    # Movement trajectory:

    #print "Movement: i_netw, 2*i_trial =", i_netw, 2*i_trial
    #figure()
    i_trial_temp = i_trial
    i,j = nonzero(occupancyMap[i_netw][2*i_trial] < inf)
    tmp = zeros(len(i))
    for k in xrange(len(i)):
        tmp[k] = occupancyMap[i_netw][2*i_trial][i[k]][j[k]]
    i_sorted=argsort(tmp)
    hugemat = -inf*ones([len(i), 200, 200])
    for k in xrange(len(i)):                                                   
        for l in xrange(k):
            #hugemat[k, i[l], j[l]] = occupancyMap[0][i_trial][i[l]][j[l]]
            hugemat[k, i[i_sorted[l]], j[i_sorted[l]]] = 1 # occupancyMap[i_netw][i_trial][i[i_sorted[l]]][j[i_sorted[l]]]

    path_counter = 0
    for k in xrange(len(i)):
         if mod(k, 10)==0:   
            print "Bin %i of %i"%(k, len(i))

         path_counter += 1
         subplot(1,3,2)
         n_edge = ceil(maze_edge_cm / float(L_maze_cm) * occ_len)

         #matshow(transpose(hugemat[k,:,:]), origin='lower') #, fignum=False) # , cmap=cm.YlOrRd
         matshow(transpose(hugemat[k, n_edge : occ_len - n_edge, n_edge : occ_len - n_edge]), origin='lower', fignum=False) #, vmax = wmax_home)
         ax = gca()
         '''#ax.set_xticks([0, 199])
         ax.set_xticklabels([0, 2])
         ax.set_yticks([0, 199])
         ax.set_yticklabels([0, 2])'''
         ax.set_xticks([0, occ_len - 2*n_edge])
         ax.set_xticklabels([0, int(0.01*(L_maze_cm - 2*maze_edge_cm))])
         ax.set_yticks([0, occ_len - 2*n_edge])
         ax.set_yticklabels([0, int(0.01*(L_maze_cm - 2*maze_edge_cm))])

         ax.xaxis.set_ticks_position('bottom')
         xlabel('x [m]')
         ylabel('y [m]')
         #title('Movement trajectory, Trial '+str(i_trial_temp+1))
         title('Movement trajectory')
         #savefig('img_movie/trajectory_'+str(k))
         savefig('img_movie/combined_'+str(k + seq_counter))
         #if k < len(i)-1:
         #    close()


if weights_for_movie:
    # Weights
    n_trial = 3 # 1 # 3 # 2 # 1 # 0
    i_netw = 9 # 2

    n_edge_w = ceil(maze_edge_cm / float(L_maze_cm) * n_grid)

    wmax_home = 0
    for i_trial in xrange(4): # 5
            placeValueMat = zeros([n_grid, n_grid])
            for iNeuron in xrange(len(weight_array_home[i_netw][i_trial])):
                x,y = net_init.xypos(iNeuron, n_grid)
                placeValueMat[x, y] = weight_array_home[i_netw][2*i_trial][iNeuron]
            if placeValueMat.max() > wmax_home:
                wmax_home = placeValueMat.max()


    print "seq_counter, path_counter = ", seq_counter, path_counter

    for i_trial in xrange(n_trial, n_trial+1):
            placeValueMat = zeros([n_grid, n_grid])
            for iNeuron in xrange(len(weight_array_home[i_netw][i_trial])):
                x,y = net_init.xypos(iNeuron, n_grid)
                placeValueMat[x, y] = weight_array_home[i_netw][2*i_trial][iNeuron]
            #matshow(transpose(placeValueMat), origin='lower') # , fignum=False)
            subplot(1,3,3)
            matshow(transpose(placeValueMat[n_edge_w : n_grid - n_edge_w, n_edge_w : n_grid - n_edge_w]), origin='lower', vmax = wmax_home, fignum=False)
            title('LEC-DG weights') # Learned 
            ax = gca()
            ax.set_xticks([0, n_grid - 2*n_edge_w - 1])
            ax.xaxis.set_ticks_position('bottom')
            ax.set_xticklabels([0, int(0.01*(L_maze_cm - 2*maze_edge_cm))])
            ax.set_yticks([0, n_grid - 2*n_edge_w - 1])
            ax.set_yticklabels([0, int(0.01*(L_maze_cm - 2*maze_edge_cm))])
            #savefig('movement_data/'+'weights_Home_'+identifier+'_trial_'+str(i_trial))            
            for ind in xrange(50): # 25
                savefig('img_movie/combined_'+str(seq_counter + path_counter + ind))

    show() # remove this!!!

# avconv -f image2 -i test_%d.png -r 76 -s 800x600 foo.avi # works!!!

# ffmpeg -f image2 -r 24 -i test_%d.png -vcodec mpeg4 -y movie.mp4 # works!!!



ioff()
#show()


