import mDB_azizi_newInitLearn as net_init
from pylab import *
import pickle
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from scipy.optimize import curve_fit

import numpy as np
#from ann_InitLearn import xypos, xypos_maze, xy_index_maze
from matplotlib import colors
import new_colormaps as nc


# ---------------------------------------------------------------------------------
# Defines a Gaussian function, used for evaluating Gaussian fits of learned weights
def func(x, a, mx, my, sigmaqd, b):                               
    return a * exp( -((x[0]-mx)**2 + (x[1]-my)**2)/sigmaqd ) + b
# ---------------------------------------------------------------------------------


# ------------------------------------------------------------------------------------------------------------
def plot_latencies_timecourse(latencies_Home, latencies_Away, n_valid_nets, nTrials, identifier, dpisize, dpisize_save, savefigs, save_plotdata = False):

        figure(figsize=(4,4), dpi=dpisize) 

        y_err_bar_Home = [std(latencies_Home, 0) / sqrt(n_valid_nets) , std(latencies_Home, 0)/ sqrt(n_valid_nets)]
        errorbar(xrange(1, nTrials+1,   2), mean(latencies_Home, 0), yerr=y_err_bar_Home, color= 'r', marker='o', markersize=3, linewidth=2)
        y_err_bar_Away = [std(latencies_Away, 0) / sqrt(n_valid_nets), std(latencies_Away, 0) / sqrt(n_valid_nets)]
        errorbar(xrange(2, nTrials+1, 2), mean(latencies_Away, 0), yerr=y_err_bar_Away, color= 'k', marker='o', markersize=3, linewidth=2)

        ax=gca()
        ax.set_xticks([1, nTrials])
        ax.set_xticklabels([1, nTrials])
        yl = ax.get_ylim()
        ax.set_yticks([0, yl[1]])
        ax.set_yticklabels([0, str(int(yl[1]))])
        for label in ax.get_xticklabels() + ax.get_yticklabels(): 
            label.set_fontsize(15)
        xlabel('Trial', fontsize=15)
        ylabel('Latency [s]', fontsize=15) # , mean $\pm$ s.e.m.
        title('Mean of '+str(n_valid_nets)+' networks', fontsize=15) # Average latencies across
        #legend(('Home', 'Random'), numpoints=1, loc='upper right', frameon=False, fontsize=10)
        legend(('Home', 'Random'), numpoints=1, loc='center right', frameon=True, fontsize=15)
        tight_layout()

        if savefigs:
                savefig('movement_data/'+'latencies_'+identifier+'_'+str(nTrials), dpi=dpisize_save)

		if save_plotdata:
			filename_w = 'data/latencies_' + str(nTrials) + '_' + identifier + '.txt'
			file = open(filename_w, 'w')
			pickle.dump(mean(latencies_Home, 0), file, 0)
			pickle.dump(std(latencies_Home, 0) / sqrt(n_valid_nets), file, 0)
			pickle.dump(mean(latencies_Away, 0), file, 0)
			pickle.dump(std(latencies_Away, 0) / sqrt(n_valid_nets), file, 0)
			pickle.dump(n_valid_nets, file, 0)
			file.close()


        return

# ------------------------------------------------------------------------------------------------------------
def plot_latencies_overall(latencies_Home, latencies_Away, nTrials, identifier, dpisize, dpisize_save, savefigs):
        figure(figsize=(4,4), dpi=dpisize)

        n_home_trials = int( ceil(nTrials/2.0) )
        n_away_trials = int( floor(nTrials/2.0) )
        ax=gca()
        rects1 = ax.bar(0.5, mean(latencies_Away), 1, color='k')
        rects2 = ax.bar(1.5, mean(mean(latencies_Home, 0)[0 : n_away_trials]), 1, color='r')
        title('Overall mean', fontsize=15)
        ax.set_xticks([0, 1, 2, 3])
        ax.set_xticklabels(['', 'Random', 'Home', ''])
        yl = ax.get_ylim()
        ax.set_yticks([0, yl[1]])
        ax.set_yticklabels([0, str(int(yl[1]))])
        for label in ax.get_xticklabels() + ax.get_yticklabels(): 
            label.set_fontsize(15)
        xlabel('Trial phase', fontsize=15)
        ylabel('Latency [s]', fontsize=15) # , mean $\pm$ s.e.m.
        tight_layout()
        show()

        if savefigs:
                savefig('movement_data/'+'latencies-overall_'+identifier+'_'+str(nTrials), dpi=dpisize_save)

        return

# ------------------------------------------------------------------------------------------------------------
def plot_latency_hist(latencies_Home, latencies_Away, nTrials, identifier, dpisize, dpisize_save, savefigs):

        # plot a histogram of latencies:
        figure(figsize=(4,4), dpi=dpisize)
        rsh = reshape(latencies_Home, len(nonzero(latencies_Home)[0]))
        rsa = reshape(latencies_Away, len(nonzero(latencies_Away)[0]))
        thr = 300
        n_bins = 30
        binsize_sec = thr / float(n_bins)
        #hist([rsh * (rsh < thr) + (thr + 100)*(rsh > thr), rsa * (rsa < thr) + (thr+100)*(rsa > thr)], n_bins, color=['r','k'], linewidth=0.5)
        hist([rsh, rsa ], n_bins, color=['r','k'], linewidth=0.5)
        ax=gca()
        #ax.set_xticks([0, thr, thr + 100])
        #ax.set_xticklabels([0, thr, '>'+str(thr)], fontsize=8)

        for label in ax.get_xticklabels() + ax.get_yticklabels():
            label.set_fontsize(15)
        xlabel('Latency [s]', fontsize=15)
        ylabel('Number of trials', fontsize=15)
        #title('Latencies across 36 networks', fontsize=8)
        legend(('Home', 'Random'), numpoints=1, loc='upper left', frameon=True, fontsize=15)
        tight_layout()

        if savefigs:
                savefig('movement_data/'+'hist_latencies_'+identifier+'_'+str(nTrials), dpi=dpisize_save)


# ------------------------------------------------------------------------------------------------------------
def plot_Home_weights(weight_array_home, weights, L_maze_cm, maze_edge_cm, n_grid, n_valid_nets, nTrials, identifier, dpisize, dpisize_save, savefigs):

        placeValueMat = zeros([n_grid, n_grid])
        figure(figsize=(4, 4), dpi=dpisize)
        wmax_home = round(weight_array_home.max() / 1.0e-9, 2)
        for n in xrange(min(36, n_valid_nets)): # Caution - "new" values will be inserted at lower indices!
                placeValueMat = zeros([n_grid, n_grid])
                for iNeuron in xrange(len(weight_array_home[n][nTrials-1])):
                    x,y = net_init.xypos(iNeuron, n_grid)
                    placeValueMat[int(x), int(y)] = weight_array_home[n][nTrials-1][iNeuron] / 1.0e-9
                n_edge = int( ceil(maze_edge_cm / float(L_maze_cm) * n_grid) )
                spl = subplot(6,6,n+1)
                #spl.text(.505, .802, 'Network '+str(n), fontsize=8, horizontalalignment='center', transform=spl.transAxes, color='w')
                #spl.text(.5, .8, 'Network '+str(n), fontsize=8, horizontalalignment='center', transform=spl.transAxes, color='k')
                #spl.text(.505, .102, round(placeValueMat.max(), 3), fontsize=8, horizontalalignment='center', transform=spl.transAxes, color='w')
                #spl.text(.5, .1, round(placeValueMat.max(), 3), fontsize=8, horizontalalignment='center', transform=spl.transAxes, color='k')
                title('Network '+str(n), fontsize=8, verticalalignment='top')
                matshow(transpose(placeValueMat[n_edge : n_grid - n_edge, n_edge : n_grid - n_edge]), origin='lower', fignum=False, vmax = wmax_home)
                ax = gca()
                ax.set_xticks([])
                ax.set_xticklabels([])
                ax.set_yticks([])
                ax.set_yticklabels([])
                if n==30:
                    ax.set_xticks([0, n_grid - 2*n_edge - 1])
                    ax.xaxis.set_ticks_position('bottom')
                    ax.set_xticklabels([0, L_maze_cm - 2*maze_edge_cm], fontsize=8)
                    ax.set_yticks([0, n_grid - 2*n_edge - 1])
                    ax.set_yticklabels([0, L_maze_cm - 2*maze_edge_cm], fontsize=8)
                    xlabel('x [m]', fontsize=8, verticalalignment= 'bottom')
                    ylabel('y [m]', fontsize=8) #, verticalalignment= 'top', horizontalalignment='left')
                if n==min(n_valid_nets-1, 35):
                    axins = inset_axes(ax, width="5%", height="100%",loc=3, bbox_to_anchor=(1.05, 0., 1, 1), bbox_transform=ax.transAxes, borderpad=0)
                    cbar = colorbar(cax=axins) #fraction = 0.05)
                    cbar.set_ticks([cbar.vmin, cbar.vmax])
                    cbar.set_ticklabels([0, str(wmax_home) + ' nA'])
                    axes(cbar.ax)
                    yticks(fontsize=8)
                    #title('w [nA]', fontsize=8)
                    axes(ax)
                if n==0 or n==1:
                    filename_w = 'data/wdata_cont_home_DG_' + str(nTrials) + '_' + identifier + '_netw_'+str(n)
                    file = open(filename_w, 'w'); pickle.dump(weights[n],file,0); file.close()
        #subplots_adjust(hspace=.05, wspace=.01)
        #subplots_adjust(hspace=.25, wspace=.01)

        suptitle('Weights from LEC Home cells onto DG', fontsize=8)
        if savefigs:
                savefig('movement_data/'+'placevalue_'+identifier+'_'+str(nTrials), dpi=dpisize_save)

        return

# ------------------------------------------------------------------------------------------------------------
def plot_Away_weights(weight_array_away, weights, L_maze_cm, maze_edge_cm, n_grid, n_valid_nets, nTrials, identifier, dpisize, dpisize_save, savefigs):

        plot_away_weights = True # False # 
        if plot_away_weights:
                placeValueMat = zeros([n_grid, n_grid])
                figure(figsize=(4, 4), dpi=dpisize)
                wmax_away = round(weight_array_away.max() / 1.0e-9, 2)
                for n in xrange(min(36, n_valid_nets)): # Caution - "new" values will be inserted at lower indices!
                        placeValueMat = zeros([n_grid, n_grid])
                        for iNeuron in xrange(len(weight_array_away[n][nTrials-1])):
                            x,y = net_init.xypos(iNeuron, n_grid)
                            placeValueMat[int(x), int(y)] = weight_array_away[n][nTrials-1][iNeuron] / 1.0e-9
                        n_edge = int( ceil(maze_edge_cm / float(L_maze_cm) * n_grid) )
                        spl = subplot(6,6,n+1)
                        #spl.text(.505, .802, 'Network '+str(n), fontsize=8, horizontalalignment='center', transform=spl.transAxes, color='w')
                        #spl.text(.5, .8, 'Network '+str(n), fontsize=8, horizontalalignment='center', transform=spl.transAxes, color='k')
                        #spl.text(.505, .102, round(placeValueMat.max(), 3), fontsize=8, horizontalalignment='center', transform=spl.transAxes, color='w')
                        #spl.text(.5, .1, round(placeValueMat.max(), 3), fontsize=8, horizontalalignment='center', transform=spl.transAxes, color='k')
                        title('Network '+str(n), fontsize=8, verticalalignment='top')
                        matshow(transpose(placeValueMat[n_edge : n_grid - n_edge, n_edge : n_grid - n_edge]), origin='lower', fignum=False, vmax = wmax_away) # cmap=cm.Greys,
                        ax = gca()
                        ax.set_xticks([])
                        ax.set_xticklabels([])
                        ax.set_yticks([])
                        ax.set_yticklabels([])
                        if n==30:
                            ax.set_xticks([0, n_grid - 2*n_edge - 1])
                            ax.xaxis.set_ticks_position('bottom')
                            ax.set_xticklabels([0, L_maze_cm - 2*maze_edge_cm], fontsize=8)
                            ax.set_yticks([0, n_grid - 2*n_edge - 1])
                            ax.set_yticklabels([0, L_maze_cm - 2*maze_edge_cm], fontsize=8)
                            xlabel('x [m]', fontsize=8, verticalalignment= 'bottom')
                            ylabel('y [m]', fontsize=8) #, verticalalignment= 'top', horizontalalignment='left')

                        if n==min(36, n_valid_nets)-1:
                            #ax.set_xticks([0, n_grid-1])
                            #ax.set_xticklabels([0, 3.5])
                            #ax.set_yticks([0, n_grid-1])
                            #ax.set_yticklabels([0, 3.5])
                            #ax.set_xticks([0, n_grid - 2*n_edge - 1])
                            #ax.set_xticklabels([0, L_maze_cm - 2*maze_edge_cm], fontsize=8)
                            #ax.xaxis.set_ticks_position('bottom')
                            #ax.set_yticks([0, n_grid - 2*n_edge - 1])
                            #ax.set_yticklabels([0, L_maze_cm - 2*maze_edge_cm], fontsize=8)
                            #xlabel('x [m]', fontsize=8)
                            #ylabel('y [m]', fontsize=8)
                            axins = inset_axes(ax, width="5%", height="100%",loc=3, bbox_to_anchor=(1.05, 0., 1, 1), bbox_transform=ax.transAxes, borderpad=0)
                            cbar = colorbar(cax=axins) #fraction = 0.05)
                            cbar.set_ticks([cbar.vmin, cbar.vmax])
                            #cbar.set_ticklabels([0, 'Max'])
                            cbar.set_ticklabels([0, str(wmax_away) + ' nA'])
                            axes(cbar.ax)
                            yticks(fontsize=8)
                            #title('w [nA]', fontsize=8)
                            axes(ax)
                        #filename_w = 'data/wdata_cont_home_DG_' + str(nTrials) + '_' + identifier + '_netw_'+str(n)
                        #file = open(filename_w, 'w'); pickle.dump(weights[n],file,0); file.close()
                #subplots_adjust(hspace=.05, wspace=.01)
                suptitle('Weights from LEC Away cells onto DG', fontsize=8)
                if savefigs:
                        savefig('movement_data/'+'placevalue_away_'+identifier+'_'+str(nTrials), dpi=dpisize_save)


        return

# ------------------------------------------------------------------------------------------------------------
def plots_dist_to_goal(endpoints_Home, endpoints_Away, goal_index_array_Home, goal_index_array_Away, L_maze_cm, maze_edge_cm, n_grid, n_valid_nets, nTrials, identifier, dpisize, dpisize_save, savefigs, save_plotdata=True):
        figure(figsize=(4,4), dpi=dpisize)       
        x,y = net_init.xypos_maze(endpoints_Home, n_grid, L_maze_cm)
        xr, yr = net_init.xypos_maze(goal_index_array_Home, n_grid, L_maze_cm)
        dist_home = sqrt((x-xr)**2 + (y-yr)**2)

        boxplot(dist_home)
        #plot(xrange(1, int(floor(nTrials/2.0) +1)), mean(dist_home, 0), 'r*')
        #plot(xrange(1, len(endpoints_Home[0])+1), mean(dist_home, 0), 'r*') # Extra plot of mean values
        title('Distances between sequence endpoint and reward location (Home trials)', fontsize=15) # , N='+str(n_valid_nets)+' networks', fontsize=8)#  (last)  #, unscaled
        title('Pooled across '+str(n_valid_nets)+' networks', fontsize=15) # , fontsize=8)#  (last)  #, unscaled
        xlabel('Trial', fontsize=15)
        ylabel('Remaining distance to Home [cm]', fontsize=15)
        ax = gca()
        ax.set_xticks([1, int(ceil(nTrials/2.0))])
        ax.set_xticklabels([1, int(ceil(nTrials/2.0))])
        for label in ax.get_xticklabels() + ax.get_yticklabels(): 
            label.set_fontsize(15)
        tight_layout()
        if savefigs:
                savefig('movement_data/'+'endpoint_distances_'+identifier+'_'+str(nTrials), dpi=dpisize_save)


        # Plot a histogram of distance to goal:
        figure(figsize=(3,3), dpi=dpisize)
        x,y = net_init.xypos_maze(endpoints_Away, n_grid, L_maze_cm)
        xr, yr = net_init.xypos_maze(goal_index_array_Away, n_grid, L_maze_cm)
        dist_away = sqrt((x-xr)**2 + (y-yr)**2)
	#print "dist_home, dist_away = ", dist_home, dist_away
        hist([reshape(dist_home, prod(dist_home.shape)), reshape(dist_away, prod(dist_away.shape))], 25, color=['r','k'])
        xlabel('d [cm]', fontsize=8)
        ylabel('Number of trials', fontsize=8)
        title('Histogram of d(endpoint, reward)', fontsize=8)
        ax=gca()
        for label in ax.get_xticklabels() + ax.get_yticklabels():
            label.set_fontsize(8)
        legend(('Home', 'Random'), numpoints=1, loc='upper left', frameon=False, fontsize=8)
        if savefigs:
                savefig('movement_data/'+'hist_disttogoal_'+identifier+'_'+str(nTrials), dpi=dpisize_save)

		if save_plotdata:
			filename_w = 'data/disttogoal_' + str(nTrials) + '_' + identifier + '.txt'
			file = open(filename_w, 'w')
			pickle.dump(mean(dist_home, 0), file, 0)
			pickle.dump(std(dist_home, 0) / sqrt(n_valid_nets), file, 0)
			pickle.dump(mean(dist_away, 0), file, 0)
			pickle.dump(std(dist_away, 0) / sqrt(n_valid_nets), file, 0)
			pickle.dump(n_valid_nets, file, 0)
			file.close()


        return

# ------------------------------------------------------------------------------------------------------------
def plot_navigation_strategies(rand_nav_time_Home, goal_nav_time_Home, focal_search_time_Home, rand_nav_time_Away, goal_nav_time_Away, focal_search_time_Away, n_valid_nets, nTrials, identifier, dpisize, dpisize_save, savefigs):

        total_nav_time_Home = goal_nav_time_Home + rand_nav_time_Home + focal_search_time_Home
        total_nav_time_Away = goal_nav_time_Away + rand_nav_time_Away + focal_search_time_Away

        fig, ax = subplots(figsize=(3,3), dpi=dpisize) # creates a new figure

        latsum_Home = mean(rand_nav_time_Home + goal_nav_time_Home + focal_search_time_Home, 0)
        latsum_Away = mean(rand_nav_time_Away + goal_nav_time_Away + focal_search_time_Away, 0)

        width = 0.35
        fsize = 6 # 8
        ind = arange(int(ceil(nTrials/2.0)))
        y_err_bar_Home_mean = [1/sqrt(n_valid_nets) * std(total_nav_time_Home, 0), 1/sqrt(n_valid_nets) * std(total_nav_time_Home, 0)]
        rects1 = ax.bar(ind, mean(rand_nav_time_Home, 0), width, color='c') # b
        rects2 = ax.bar(ind, mean(goal_nav_time_Home, 0), width, color='k', bottom=mean(rand_nav_time_Home, 0)) # r
        rects3 = ax.bar(ind, mean(focal_search_time_Home, 0), width, color='m', bottom=mean(rand_nav_time_Home, 0)+mean(goal_nav_time_Home, 0), yerr=y_err_bar_Home_mean) # y
        ind = arange(int(floor(nTrials/2.0)))
        y_err_bar_Away_mean = [1/sqrt(n_valid_nets) * std(total_nav_time_Away, 0), 1/sqrt(n_valid_nets) * std(total_nav_time_Away, 0)]
        rects4 = ax.bar(ind+width, mean(rand_nav_time_Away, 0), width, color='b')
        rects5 = ax.bar(ind+width, mean(goal_nav_time_Away, 0), width, color='r', bottom=mean(rand_nav_time_Away, 0)) # c
        rects6 = ax.bar(ind+width, mean(focal_search_time_Away, 0), width, color='y', bottom=mean(rand_nav_time_Away, 0)+mean(goal_nav_time_Away, 0), yerr=y_err_bar_Away_mean) # k
        legend(('Random navigation [Home]', 'Goal navigation [Home]', 'Focal search [Home]', 'Random navigation [Away]', 'Goal navigation [Away]', 'Focal search [Away]'), fontsize=fsize) # m
        xlabel('Trial', fontsize=fsize)
        ylabel('t [s]', fontsize=fsize)          
        title('Navigation times (Mean $\pm$ s.e.m.) across trials ('+str(n_valid_nets)+' networks)', fontsize=fsize)
        ax=gca()
        for label in ax.get_xticklabels() + ax.get_yticklabels():
            label.set_fontsize(fsize)
        if savefigs:
                savefig('movement_data/'+'nav_strategies_'+identifier+'_'+str(nTrials), dpi=dpisize_save)

        show()  

        return

# ------------------------------------------------------------------------------------------------------------
def plot_startenddist(seqstart, seqend, n_grid, L_maze_cm, n_valid_nets, nTrials, identifier, dpisize, dpisize_save, savefigs, save_plotdata = False):

        figure(figsize=(4,4), dpi=dpisize)         


        # Plot start-to-end distances of sequences across trials:
        seqst_nz_Home = zeros([n_valid_nets, int( ceil(nTrials/2.0) )])
        seqend_nz_Home = zeros([n_valid_nets, int( ceil(nTrials/2.0) )])
        seqst_nz_Away = zeros([n_valid_nets, int(floor(nTrials / 2.0))])
        seqend_nz_Away = zeros([n_valid_nets, int(floor(nTrials / 2.0))])
        #seqst_nz_Away = zeros([n_valid_nets, int( ceil(nTrials/2.0) )])
        #seqend_nz_Away = zeros([n_valid_nets, int( ceil(nTrials/2.0) )])
        for inet in xrange(n_valid_nets):                                 
            for jt in xrange(nTrials):
                if mod(jt, 2)==0:
                    seqst_nz_Home[inet, int(ceil(jt/2.0))] = seqstart[inet][jt][0]
                    seqend_nz_Home[inet, int(ceil(jt/2.0))] = seqend[inet][jt][0]
                else:
                    seqst_nz_Away[inet, int(floor(jt/2.0))] = seqstart[inet][jt][0]
                    seqend_nz_Away[inet, int(floor(jt/2.0))] = seqend[inet][jt][0]
                    #seqst_nz_Away[inet, ceil(jt/2.0)] = seqstart[inet][jt][0]
                    #seqend_nz_Away[inet, ceil(jt/2.0)] = seqend[inet][jt][0]
        xs_home, ys_home = net_init.xypos_maze(seqst_nz_Home, n_grid, L_maze_cm)
        xe_home, ye_home = net_init.xypos_maze(seqend_nz_Home, n_grid, L_maze_cm)
        xs_away, ys_away = net_init.xypos_maze(seqst_nz_Away, n_grid, L_maze_cm)
        xe_away, ye_away = net_init.xypos_maze(seqend_nz_Away, n_grid, L_maze_cm)
        dist_startend_Home = sqrt((xs_home - xe_home)**2 + (ys_home - ye_home)**2)
        dist_startend_Away = sqrt((xs_away - xe_away)**2 + (ys_away - ye_away)**2)
        y_err_bar_Home = [std(dist_startend_Home, 0) / sqrt(n_valid_nets) , std(dist_startend_Home, 0)/ sqrt(n_valid_nets)]
        errorbar(xrange(1, nTrials+1,   2), mean(dist_startend_Home, 0), yerr=y_err_bar_Home, color= 'r', marker='o', markersize=3, linewidth=1.5)
        y_err_bar_Away = [std(dist_startend_Away, 0) / sqrt(n_valid_nets), std(dist_startend_Away, 0) / sqrt(n_valid_nets)]
        #errorbar(xrange(2, nTrials+1, 2), mean(dist_startend_Away, 0), yerr=y_err_bar_Away, color= 'k', marker='o', markersize=3, linewidth=1.5)
        errorbar(xrange(1, nTrials,   2), mean(dist_startend_Away, 0), yerr=y_err_bar_Away, color= 'k', marker='o', markersize=3, linewidth=1.5)
        ax=gca()
        ax.set_xticks([1, nTrials])
        ax.set_xticklabels([1, nTrials])
        yl = ax.get_ylim()
        ax.set_yticks([0, yl[1]])
        ax.set_yticklabels([0, str(int(yl[1]))])
        for label in ax.get_xticklabels() + ax.get_yticklabels(): 
            label.set_fontsize(15)
        xlabel('Trial', fontsize=15)
        ylabel('Start-to-end distance [cm]', fontsize=15) # , mean $\pm$ s.e.m.
        #title('Average start-to-end distance of sequences across '+str(n_valid_nets)+' networks', fontsize=15)
        title('Mean of '+str(n_valid_nets)+' networks', fontsize=15)
        legend(('Home', 'Random'), numpoints=1, loc='lower right', frameon=True, fontsize=15)
        tight_layout()

        if savefigs:
                savefig('movement_data/'+'startenddist_'+identifier+'_'+str(nTrials), dpi=dpisize_save)

		if save_plotdata:
			filename_w = 'data/startenddist_' + str(nTrials) + '_' + identifier + '.txt'
			file = open(filename_w, 'w')
			pickle.dump(mean(dist_startend_Home, 0), file, 0)
			pickle.dump(std(dist_startend_Home, 0) / sqrt(n_valid_nets), file, 0)
			pickle.dump(mean(dist_startend_Away, 0), file, 0)
			pickle.dump(std(dist_startend_Away, 0) / sqrt(n_valid_nets), file, 0)
			pickle.dump(n_valid_nets, file, 0)
			file.close()


        #return

# ------------------------------------------------------------------------------------------------------------
#def plot_weight_vs_dist(weight_array_home, goal_index_array_Home, n_grid, L_maze_cm, maze_edge_cm, n_valid_nets, nTrials, identifier, dpisize, dpisize_save, savefigs):
def plot_weight_vs_dist(weight_array_home, weight_array_away, goal_index_array_Home, n_grid, L_maze_cm, maze_edge_cm, n_valid_nets, nTrials, identifier, dpisize, dpisize_save, savefigs):
        # weights plotted across distance from Home:

        figure(figsize=(4,4), dpi=dpisize)

        dist = -inf*ones([n_valid_nets, n_grid**2])
        x, y = net_init.xypos_maze(range(n_grid**2), n_grid, L_maze_cm)
        for inet in xrange(n_valid_nets):                             
            xh, yh = net_init.xypos_maze(goal_index_array_Home[inet][0], n_grid, L_maze_cm)
            for iw in xrange(n_grid**2):
                if x[iw] > maze_edge_cm and x[iw] < L_maze_cm - maze_edge_cm and y[iw] > maze_edge_cm and y[iw] < L_maze_cm - maze_edge_cm:
                     dist[inet, iw] = sqrt((x[iw]-xh)**2 + (y[iw]-yh)**2)       # dist[:,:] contains the distances to the "current" Home for all cells, for all networks

        inz = nonzero(dist[0,:] > -inf)[0]              # Neuron indices of neurons "inside the maze"
        #isorted = argsort(dist[0,:][inz])
        #wsorted=-inf*ones([n_valid_nets, len(isorted)])
        maxdist = 0
        imax = 0
        for inet in xrange(n_valid_nets):
            #isorted = argsort(dist[inet,:][inz])
            #wsorted[inet] = weight_array_home[inet][floor(nTrials/2.0)][inz[isorted]]
            #wsorted[inet] = weight_array_home[inet][i_trial][inz[isorted]]
            if dist[inet,:][inz].max() > maxdist:
                maxdist = dist[inet,:][inz].max()
                imax = inet                             # Get the network index with "maximum" distance to Home (= imax)
        dists_sorted = sort(dist[imax, inz])            # Distances (for network imax) sorted
        lastval = dists_sorted[0]
        n_vals = 1
        for k in xrange(len(dists_sorted)-1):           # Now, summarize all "equal" distances into the same bin
            if dists_sorted[k+1] <> lastval:
                n_vals += 1
                lastval = dists_sorted[k+1]
        distvals=zeros(n_vals)
        lastval = dists_sorted[0]
        n_vals = 1
        for k in xrange(len(dists_sorted)-1):                                  
            if dists_sorted[k+1] <> lastval:
                n_vals += 1
                distvals[n_vals-1] = dists_sorted[k+1]  # distvals now contains all possible distances exactly once
                lastval = dists_sorted[k+1]


        # Initial weights:
        
        wmean = nan*ones([n_valid_nets, len(distvals)])
        wmean_away = nan*ones([n_valid_nets, len(distvals)])
        initweights = 1.0e-10 * rand(len(weight_array_home[:,0,0]), len(weight_array_home[0,:,0]), len(weight_array_home[0,0,:]))
        initweights_away = 1.0e-10 * rand(len(weight_array_away[:,0,0]), len(weight_array_away[0,:,0]), len(weight_array_away[0,0,:]))

        for inet in xrange(n_valid_nets):
            for idv in xrange(len(distvals)):
                ind = nonzero(dist[inet, inz] == distvals[idv])[0]

                if len(ind)>0: wmean[inet, idv] = initweights[inet][0][inz][ind].mean()
                if len(ind)>0: wmean_away[inet, idv] = initweights_away[inet][0][inz][ind].mean()

        #plot(dist[n_valid_nets-1, inz][isorted], mean(wsorted, 0))
        #subplot(1,2,1)
        plot(distvals, nanmean(wmean, 0) / 1.0e-9, '.', markersize=3)
        
        # Weights after learning:
        #for i_trial in xrange(0, int(floor(nTrials/2.0)), 2):
        #for i_trial in xrange(0, int(floor(nTrials/2.0))+1, 5):

        #for i_trial in xrange(0, int(floor(nTrials/2.0))+1, 10): # last version
        for i_trial in xrange(0, int(floor(nTrials/2.0))+1, 1):
                wmean = nan*ones([n_valid_nets, len(distvals)])
                wmean_away = nan*ones([n_valid_nets, len(distvals)])
                for inet in xrange(n_valid_nets):
                    for idv in xrange(len(distvals)):
                        ind = nonzero(dist[inet, inz] == distvals[idv])[0]
                        #if len(ind)>0: wmean[inet, idv] = weight_array_home[inet][floor(nTrials/2.0)][inz][ind].mean()
                        if len(ind)>0: wmean[inet, idv] = weight_array_home[inet][i_trial][inz][ind].mean()             
                        if len(ind)>0: wmean_away[inet, idv] = weight_array_away[inet][i_trial][inz][ind].mean()

                        # For each network, wmean[:,idv] now contains the mean of Home weights onto cells with distance from Home equal to: dist[inet, inz] == distvals[idv]

                #plot(dist[n_valid_nets-1, inz][isorted], mean(wsorted, 0))
                plot(distvals, nanmean(wmean, 0) / 1.0e-9, '.', markersize=3)   # Mean across networks

        ax=gca()
        ax.set_xticks([0, 300])
        ax.set_xticklabels([0, 300])
        ax.set_yticks([0, 1.0])
        ax.set_yticklabels([0, 1.0])

        xlabel('Distance from Home [cm]', fontsize=15)
        ylabel('Mean weight [nA]', fontsize=15)
        #title('Mean Home weight plotted across distance from Home', fontsize=15)
        title('Mean across networks', fontsize=15)
        #legend(('Initial weights', 'After Trial 1', 'After Trial 2', 'After Trial 3', 'After Trial 4'), fontsize=8)
        #legend(('Initial', 'Trial 1', 'Trial 5', 'Trial 10', 'Trial 15','Trial 20'), fontsize=15)

        #legend(('Initial', 'Trial 1', 'Trial 10', 'Trial 20'), fontsize=15)
        legend(('Initial', 'Trial 1', 'Trial 2', 'Trial 3'), fontsize=15)

        ax = gca()
        for label in ax.get_xticklabels() + ax.get_yticklabels():
            label.set_fontsize(15)
        tight_layout()        

    
        if savefigs:
                savefig('movement_data/'+'meanweight_'+identifier+'_'+str(nTrials), dpi=dpisize_save) 


        return

# ------------------------------------------------------------------------------------------------------------

def combined_plot_Both(center_mat_array, summed_prob, occupancyMap, weight_array_home, weight_array_away, goal_index_array, i_netw, nTrials, maze_edge_cm, L_maze_cm, center_len, occ_len, dpisize, n_spatial_bins):
    figure(figsize=(4,4), dpi=dpisize)
    disp_trials = 10 ## 7 # 4 # 10 # 6 # 5 #
    wmax = 0
    wmax#_home = round(weight_array_home.max() / 1.0e-9, 2)
    #i_netw = i_net_occ
    n_grid = 80

    for i_trial in xrange( min(nTrials, disp_trials)-1, -1, -1): # Reversed order!
        # Bump movement - data manager
        subplot(2, disp_trials, i_trial + 1)
        n_edge = int( ceil(maze_edge_cm / float(L_maze_cm) * center_len) )
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
            xlabel('x [m]', fontsize=8)
            ylabel('y [m]', fontsize=8)
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
        #if i_trial==1:
        #    xlabel('x position [cm]', fontsize=8)
        #    ylabel('y position [cm]', fontsize=8)
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

        # Bump movement - Bayes
        subplot(2, disp_trials, disp_trials + i_trial + 1)
        matshow(summed_prob[i_netw][i_trial], origin='lower', fignum=False, cmap=nc.inferno_r)
        xlim([maze_edge_cm/420.0 * n_spatial_bins , (420-maze_edge_cm) / 420.0 * n_spatial_bins])
        ylim([maze_edge_cm/420.0 * n_spatial_bins , (420-maze_edge_cm) / 420.0 * n_spatial_bins])
        ax=gca()
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xticklabels([])
        ax.set_yticklabels([])

    suptitle('Bump movement, trajectory and weights across trials, Network '+str(i_netw), fontsize=8)   
    #if savefigs:
    #    savefig('movement_data/'+'combined_plot_Both_'+identifier+'_'+str(nTrials), dpi=dpisize_save)


# ------------------------------------------------------------------------------------------------------------

def rotated_plot(center_mat_array, seqstart, goal_index_array, seqend, center_len, n_valid_nets, nTrials, L_maze_cm, n_spatial_bins, dpisize):
        rotated_home = zeros([2*center_len, 2*center_len]) # corresponds to an area of [- Lmaze_cm, Lmaze_cm]^2 
        rotated_away = zeros([2*center_len, 2*center_len]) # center_len = 100 (3.5cm - bins)
        alpha_all = zeros([n_valid_nets, nTrials])
        seqdist_all = zeros([n_valid_nets, nTrials])
        scaling_all = zeros([n_valid_nets, nTrials])
        k_end = zeros([n_valid_nets, nTrials])
        l_end = zeros([n_valid_nets, nTrials])

        n_grid = 80 # center_len

        for i_netw in xrange(n_valid_nets): 
            alpha = zeros(nTrials)
            for i_trial in xrange(nTrials): 

                print "i_netw, i_trial = ", i_netw, i_trial

                #matshow(center_mat_array[i_netw][i_trial]); colorbar()

                if mod(i_trial, 2) == 0: # Home trial
                    i,j = nonzero(center_mat_array[i_netw][i_trial] > -inf) # old
                    #print "i,j = ", i,j

                    sortinds = argsort(center_mat_array[i_netw][i_trial][i,j]) # ascending order; works only with "old" data manager input!!
                    #i_start, j_start = net_init.xypos_maze(seqstart[i_netw][i_trial], center_len, L_maze_cm) # Bayes
                    #print "Old: i[sortinds[0]] = ", i[sortinds[0]]#, i_start
                    #print "Old: j[sortinds[0]] = ", j[sortinds[0]]#, j_start

                    k = zeros(len(i))
                    l = zeros(len(i))
                    xs, ys = net_init.xypos_maze(seqstart[i_netw][i_trial][0], n_grid, L_maze_cm) # For results from data manager
                    xh, yh = net_init.xypos_maze(goal_index_array[i_netw][0], n_grid, L_maze_cm)
                    xe, ye = net_init.xypos_maze(seqend[i_netw][i_trial][0], n_grid, L_maze_cm)
                    print "seqstart[i_netw][i_trial][0] = ", seqstart[i_netw][i_trial][0]
                    print "xs, ys = ", xs, ys
                    print "xh, yh = ", xh, yh
                    print "xe, ye = ", xe, ye

                    seq_dist = np.sqrt((xe-xs)**2 + (ye-ys)**2)
                    home_dist_start = np.sqrt((xh-xs)**2 + (yh-ys)**2)
                    #print "len(home_dist_start) = ", len(home_dist_start)
                    #print "home_dist_start.shape = ", home_dist_start.shape
                    print "home_dist_start = ", home_dist_start

                    if home_dist_start > 0:
                        alpha[i_trial] = np.arccos( (0 * (xh-xs) + 1*(yh-ys)) / (1*home_dist_start) )
                        scaling = (100 / home_dist_start)     

                    else:
                        alpha[i_trial] = 0
                        scaling = 0
                    if xh - xs > 0:
                        alpha[i_trial] = 2*pi - alpha[i_trial] 

                    alpha[i_trial] *= -1

                    print "alpha[i_trial], scaling = ", alpha[i_trial], scaling

                    for ind in xrange(len(i)):
                        k[ind] = cos(alpha[i_trial]) * (i[ind] - i[sortinds[0]]) - sin(alpha[i_trial]) * (j[ind] - j[sortinds[0]]) # old
                        l[ind] = sin(alpha[i_trial]) * (i[ind] - i[sortinds[0]]) + cos(alpha[i_trial]) * (j[ind] - j[sortinds[0]])
                        k[ind] = round(k[ind] * scaling + center_len ) # 
                        l[ind] = round(l[ind] * scaling + center_len ) # 

                        #print "i[ind] - i[sortinds[0]] = ", i[ind] - i[sortinds[0]]

                        if (ind==0 or sum([r==s for r,s in zip([k[:ind],l[:ind]], [k[ind], l[ind]])], 0).max() < 2): 
                                # Exclude "double" counts by comparing current [k[ind],l[ind]] to the whole [k,l] array

                                if max(k[ind], l[ind]) > 2*center_len - 1:
                                        print "Error: Values outside maze area, for i_netw= %i, i_trial= %i, ind= %i" %(i_netw, i_trial, ind)
                                else:
                                        rotated_home[k[ind], l[ind]] = max(1, rotated_home[k[ind], l[ind]] + 1) # for "-inf" init values ### old
                        #else:
                        #    print "Excluding double count"

                else: # Random trial
                    i,j = nonzero(center_mat_array[i_netw][i_trial] > -inf) # old
                    #print "i,j = ", i,j

                    sortinds = argsort(center_mat_array[i_netw][i_trial][i,j]) # ascending order; works only with "old" data manager input!!
                    #i_start, j_start = net_init.xypos_maze(seqstart[i_netw][i_trial], center_len, L_maze_cm) # Bayes

                    #print "i[sortinds[0]], i_start = ", i[sortinds[0]], i_start
                    #print "j[sortinds[0]], j_start = ", j[sortinds[0]], j_start

                    k = zeros(len(i))
                    l = zeros(len(i))
                    xs, ys = net_init.xypos_maze(seqstart[i_netw][i_trial][0], n_grid, L_maze_cm) # data manager
                    xe, ye = net_init.xypos_maze(seqend[i_netw][i_trial][0], n_grid, L_maze_cm)                    
                    seq_dist = np.sqrt((xe-xs)**2 + (ye-ys)**2)

                    if i_trial > 2: ### old: > 2:
                            xh, yh = net_init.xypos_maze(goal_index_array[i_netw][i_trial-2], n_grid, L_maze_cm) # data manager
                            prev_rand_dist_start = np.sqrt((xh-xs)**2 + (yh-ys)**2)
                            print "prev_rand_dist_start = ", prev_rand_dist_start

                            if prev_rand_dist_start > 0:
                                alpha[i_trial] = np.arccos( (0 * (xh-xs) + 1*(yh-ys)) / (1*prev_rand_dist_start) )
                                scaling = (100.0 / prev_rand_dist_start)     
                            else:
                                alpha[i_trial] = 0
                                scaling = 0    
                            if xh - xs > 0:
                                alpha[i_trial] = 2*pi - alpha[i_trial] 

                            alpha[i_trial] *= -1

                            imin = 0 ## Possibly remove the "start" points for ease of display
                            for ind in xrange(imin, len(i)):
                                k[ind] = cos(alpha[i_trial]) * (i[ind] - i[sortinds[0]]) - sin(alpha[i_trial]) * (j[ind] - j[sortinds[0]]) # old
                                l[ind] = sin(alpha[i_trial]) * (i[ind] - i[sortinds[0]]) + cos(alpha[i_trial]) * (j[ind] - j[sortinds[0]])

                                k[ind] = min(round(k[ind] * scaling + center_len), 2*center_len -1)
                                l[ind] = min(round(l[ind] * scaling + center_len), 2*center_len -1)


                                #'''#
                                if (ind==imin or sum([r==s for r,s in zip([k[:ind],l[:ind]], [k[ind], l[ind]])], 0).max() < 2): 
                                        # Exclude "double" counts by comparing current [k[ind],l[ind]] to the whole [k,l] array
                                    if max(k[ind], l[ind]) > 2*center_len - 1:
                                            print "Error: Values outside maze area, for i_netw= %i, i_trial= %i, ind= %i" %(i_netw, i_trial, ind)
                                    else:
                                        rotated_away[k[ind], l[ind]] = max(1, rotated_away[k[ind], l[ind]] + 1) # old
                                        
                                #else:
                                #    print "Excluding double count"

                alpha_all[i_netw, i_trial] = alpha[i_trial]
                seqdist_all[i_netw, i_trial] = seq_dist
                scaling_all[i_netw, i_trial] = scaling
                #k_end[i_netw, i_trial] = k[sortinds[-1]]
                #l_end[i_netw, i_trial] = l[sortinds[-1]]

        figure(figsize=(6,3), dpi=dpisize)
        subplot(121)
        #'''#
        rotated_home -= 0.1*(rotated_home <> 0.1) # old
        print "rotated_home.max() = ", rotated_home.max()
        #print "rotated_home.min() = ", rotated_home.min()
        print "rotated_away.max() = ", rotated_away.max()

        matshow(transpose((1+rotated_home[int(0.5*center_len) : int(1.5*center_len), int(0.5*center_len) : int(1.5*center_len)])), origin ='lower', fignum=False, norm=colors.LogNorm(), cmap=nc.inferno) # old
        ax = gca()
        center_resol = L_maze_cm / float(center_len)
        ax.set_xticks([0.5*center_len]) # [0, , center_len]
        ax.set_xticklabels(['$\longleftarrow$ Perpendicular to Home $\longrightarrow$']) # [-0.5*center_len*center_resol, 0, 0.5*center_len*center_resol]
        ax.set_yticks([0.25*center_len, 0.75*center_len])
        ax.set_yticklabels(['Away from \n $\longleftarrow$  $\quad$ Home', 'Towards $\quad$ \n Home $\longrightarrow$'], rotation='vertical')
        ax.xaxis.set_ticks_position('bottom') #'''
        ax.xaxis.set_tick_params(length=0)
        ax.yaxis.set_tick_params(length=0)
        plot([0.5*center_len, 0.5*center_len], [0, center_len], '--', color='k', linewidth=0.5)
        plot([0, center_len], [0.5*center_len, 0.5*center_len], '--', color='k', linewidth=0.5)
        plot([0, center_len], [0.75*center_len, 0.75*center_len], '--', color='k', linewidth=0.5)

        #colorbar(shrink=0.4)
        for label in ax.get_xticklabels() + ax.get_yticklabels(): 
            label.set_fontsize(8)
        title('Home events rotated \n to Home well location', fontsize=8)


        subplot(122)
        matshow(transpose((1+rotated_away[int(0.5*center_len) : int(1.5*center_len), int(0.5*center_len) : int(1.5*center_len)])), origin ='lower', fignum=False, norm=colors.LogNorm(), cmap=nc.inferno) # old
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
        '''#
        axins = inset_axes(ax, width="5%", height="100%",loc=3, bbox_to_anchor=(1.05, 0., 1, 1), bbox_transform=ax.transAxes, borderpad=0)
        cbar = colorbar(cax=axins) #fraction = 0.05)
        cbar.set_ticks([1, 2, 11, 101, 1001])
        cbar.set_ticklabels([0, 1, 10, 100, 1000])
        cbar.ax.set_ylabel('Number of sequences', fontsize=8, rotation='vertical', position=(0, 0.5), va='bottom')
        axes(cbar.ax)
        yticks(fontsize=8)
        '''

        #if savefigs:
        #savefig('plots/'+'rotated_dataman_12netws.png', dpi=300)
        #    savefig('movement_data/'+'rotated_'+identifier+'_'+str(nTrials)+'.tif', dpi=dpisize_save)


# ------------------------------------------------------------------------------------------------------------

def rotated_plot_bayes(center_mat_array, seqstart, goal_index_array, seqend, center_len, n_valid_nets, nTrials, L_maze_cm, n_spatial_bins, dpisize):
        n_grid = 80 # center_len # 
        rotated_home = zeros([2*center_len, 2*center_len]) # corresponds to an area of [- Lmaze_cm, Lmaze_cm]^2 
        rotated_away = zeros([2*center_len, 2*center_len]) # center_len = 100 (3.5cm - bins)

        size_factor = center_len / float(n_spatial_bins)

        for i_netw in xrange(n_valid_nets): 
            alpha = zeros(nTrials)
            for i_trial in xrange(nTrials): 

                print "i_netw, i_trial = ", i_netw, i_trial

                if mod(i_trial, 2) == 0: # Home trial
                    i,j = nonzero(np.transpose(center_mat_array[i_netw][i_trial]) > center_mat_array[i_netw][i_trial].min() ) # Bayes

                    i_start, j_start = net_init.xypos_maze(seqstart[i_netw][i_trial][0], n_grid, n_spatial_bins) # Bayes with data manager input

                    k = zeros(len(i))
                    l = zeros(len(i))
                    xs, ys = net_init.xypos_maze(seqstart[i_netw][i_trial][0], n_grid, L_maze_cm) # For results from data manager
                    xh, yh = net_init.xypos_maze(goal_index_array[i_netw][0], n_grid, L_maze_cm)
                    xe, ye = net_init.xypos_maze(seqend[i_netw][i_trial][0], n_grid, L_maze_cm)

                    print "seqstart[i_netw][i_trial][0] = ", seqstart[i_netw][i_trial][0]
                    print "xs, ys = ", xs, ys
                    print "xh, yh = ", xh, yh
                    print "xe, ye = ", xe, ye

                    seq_dist = np.sqrt((xe-xs)**2 + (ye-ys)**2)
                    home_dist_start = np.sqrt((xh-xs)**2 + (yh-ys)**2)
                    print "home_dist_start = ", home_dist_start

                    if home_dist_start > 0:
                        alpha[i_trial] = np.arccos( (0 * (xh-xs) + 1*(yh-ys)) / (1*home_dist_start) )
                        scaling = (100 / home_dist_start)     
                    else:
                        alpha[i_trial] = 0
                        scaling = 0
                    if xh - xs > 0:
                        alpha[i_trial] = 2*pi - alpha[i_trial] 

                    alpha[i_trial] *= -1

                    print "alpha[i_trial], scaling = ", alpha[i_trial], scaling

                    for ind in xrange(len(i)):
                        k[ind] = cos(alpha[i_trial]) * (i[ind] - i_start) * size_factor - sin(alpha[i_trial]) * (j[ind] - j_start) * size_factor # Bayes
                        l[ind] = sin(alpha[i_trial]) * (i[ind] - i_start) * size_factor + cos(alpha[i_trial]) * (j[ind] - j_start) * size_factor
                        k[ind] = round(k[ind] * scaling + center_len ) # 
                        l[ind] = round(l[ind] * scaling + center_len ) # 

                        '''#
                        if (ind==0 or sum([r==s for r,s in zip([k[:ind],l[:ind]], [k[ind], l[ind]])], 0).max() < 2): 
                                # Exclude "double" counts by comparing current [k[ind],l[ind]] to the whole [k,l] array

                                if max(k[ind], l[ind]) > 2*center_len - 1:
                                        test = 0
                                        #print "Error: Values outside maze area, for i_netw= %i, i_trial= %i, ind= %i" %(i_netw, i_trial, ind)
                                else:
                                        rotated_home[k[ind], l[ind]] +=  np.transpose(center_mat_array[i_netw][i_trial])[i[ind], j[ind]] # for "-inf" init values ### Bayes
                        #else:
                        #    print "Excluding double count"
                        '''
                        if max(k[ind], l[ind]) <= 2*center_len-1: rotated_home[k[ind], l[ind]] +=  np.transpose(center_mat_array[i_netw][i_trial])[i[ind], j[ind]] # for "-inf" init values ### Bayes

                else: # Random trial
                    i,j = nonzero(np.transpose(center_mat_array[i_netw][i_trial]) > center_mat_array[i_netw][i_trial].min() ) # Bayes

                    i_start, j_start = net_init.xypos_maze(seqstart[i_netw][i_trial][0], n_grid, n_spatial_bins) # Bayes with data manager input

                    print "i_start = ", i_start
                    print "j_start = ", j_start

                    k = zeros(len(i))
                    l = zeros(len(i))
                    xs, ys = net_init.xypos_maze(seqstart[i_netw][i_trial][0], n_grid, L_maze_cm) # new - Bayes
                    xe, ye = net_init.xypos_maze(seqend[i_netw][i_trial][0], n_grid, L_maze_cm)
                    seq_dist = np.sqrt((xe-xs)**2 + (ye-ys)**2)

                    if i_trial > 2: ### old: > 2:
                            xh, yh = net_init.xypos_maze(goal_index_array[i_netw][i_trial-2], n_grid, L_maze_cm) # new - Bayes
                            prev_rand_dist_start = np.sqrt((xh-xs)**2 + (yh-ys)**2)
                            print "prev_rand_dist_start = ", prev_rand_dist_start

                            if prev_rand_dist_start > 0:
                                alpha[i_trial] = np.arccos( (0 * (xh-xs) + 1*(yh-ys)) / (1*prev_rand_dist_start) )
                                scaling = (100.0 / prev_rand_dist_start)     
                            else:
                                alpha[i_trial] = 0
                                scaling = 0    
                            if xh - xs > 0:
                                alpha[i_trial] = 2*pi - alpha[i_trial] 

                            alpha[i_trial] *= -1

                            imin = 0 ## Possibly remove the "start" points for ease of display
                            for ind in xrange(len(i)):
                                k[ind] = cos(alpha[i_trial]) * (i[ind] - i_start) * size_factor - sin(alpha[i_trial]) * (j[ind] - j_start) * size_factor # Bayes
                                l[ind] = sin(alpha[i_trial]) * (i[ind] - i_start) * size_factor + cos(alpha[i_trial]) * (j[ind] - j_start) * size_factor
                                k[ind] = round(k[ind] * scaling + center_len ) # 
                                l[ind] = round(l[ind] * scaling + center_len ) # 
                                '''#
                                if (ind==imin or sum([r==s for r,s in zip([k[:ind],l[:ind]], [k[ind], l[ind]])], 0).max() < 2): 
                                        # Exclude "double" counts by comparing current [k[ind],l[ind]] to the whole [k,l] array
                                    if max(k[ind], l[ind]) > 2*center_len - 1:
                                            print "Error: Values outside maze area, for i_netw= %i, i_trial= %i, ind= %i" %(i_netw, i_trial, ind)
                                    else:
                                        #rotated_away[k[ind], l[ind]] = max(1, rotated_away[k[ind], l[ind]] + 1) # old
                                        rotated_away[k[ind], l[ind]] += np.transpose(center_mat_array[i_netw][i_trial])[i[ind], j[ind]] # for "-inf" init values ### Bayes

                                #else:
                                #    print "Excluding double count"
                                '''
                                if max(k[ind], l[ind]) <= 2*center_len-1: rotated_away[k[ind], l[ind]] += np.transpose(center_mat_array[i_netw][i_trial])[i[ind], j[ind]] # for "-inf" init values ### Bayes 

        figure(figsize=(6,3), dpi=dpisize)
        subplot(121)
        #'''#
	    #rotated_home -= 0.1*(rotated_home <> 0.1) # old
        print "rotated_home.max() = ", rotated_home.max()
        #print "rotated_home.min() = ", rotated_home.min()
        print "rotated_away.max() = ", rotated_away.max()

        matshow(transpose((rotated_home[int(0.5*center_len) : int(1.5*center_len), int(0.5*center_len) : int(1.5*center_len)])), origin ='lower', fignum=False, cmap=nc.inferno)# , norm=colors.LogNorm() ### bayes without logscale
        #matshow(transpose((1 + rotated_home[int(0.5*center_len) : int(1.5*center_len), int(0.5*center_len) : int(1.5*center_len)])), origin ='lower', fignum=False, cmap=nc.inferno , norm=colors.LogNorm()) ### bayes with logscale

        ax = gca()
        center_resol = L_maze_cm / float(center_len)
        ax.set_xticks([0.5*center_len]) # [0, , center_len]
        ax.set_xticklabels(['$\longleftarrow$ Perpendicular to Home $\longrightarrow$']) # [-0.5*center_len*center_resol, 0, 0.5*center_len*center_resol]
        ax.set_yticks([0.25*center_len, 0.75*center_len])
        ax.set_yticklabels(['Away from \n $\longleftarrow$  $\quad$ Home', 'Towards $\quad$ \n Home $\longrightarrow$'], rotation='vertical')
        ax.xaxis.set_ticks_position('bottom') #'''
        ax.xaxis.set_tick_params(length=0)
        ax.yaxis.set_tick_params(length=0)
        plot([0.5*center_len, 0.5*center_len], [0, center_len], '--', color='k', linewidth=0.5)
        plot([0, center_len], [0.5*center_len, 0.5*center_len], '--', color='k', linewidth=0.5)
        plot([0, center_len], [0.75*center_len, 0.75*center_len], '--', color='k', linewidth=0.5)

        colorbar(shrink=0.4)
        for label in ax.get_xticklabels() + ax.get_yticklabels(): 
            label.set_fontsize(8)
        title('Home events rotated \n to Home well location', fontsize=8)


        subplot(122)
        matshow(transpose((rotated_away[int(0.5*center_len) : int(1.5*center_len), int(0.5*center_len) : int(1.5*center_len)])), origin ='lower', fignum=False, cmap=nc.inferno) # , norm=colors.LogNorm() # bayes without logscale
        #matshow(transpose((1  + rotated_away[int(0.5*center_len) : int(1.5*center_len), int(0.5*center_len) : int(1.5*center_len)])), origin ='lower', fignum=False, cmap=nc.inferno, norm=colors.LogNorm()) ### bayes with logscale
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
        '''#
        axins = inset_axes(ax, width="5%", height="100%",loc=3, bbox_to_anchor=(1.05, 0., 1, 1), bbox_transform=ax.transAxes, borderpad=0)
        cbar = colorbar(cax=axins) #fraction = 0.05)
        cbar.set_ticks([1, 2, 11, 101, 1001])
        cbar.set_ticklabels([0, 1, 10, 100, 1000])
        cbar.ax.set_ylabel('Number of sequences', fontsize=8, rotation='vertical', position=(0, 0.5), va='bottom')
        axes(cbar.ax)
        yticks(fontsize=8)
        '''

        #if savefigs:
        #savefig('plots/'+'rotated_Bayes_12netws.png', dpi=300)
        #savefig('plots/'+'rotated_Bayes_subset0_12netws.png', dpi=300)

        savefig('plots/'+'rotated_Bayes_all_'+str(n_valid_nets)+'netw_'+str(nTrials)+'trials.png', dpi=300)
        #savefig('plots/'+'rotated_Bayes_all_'+str(n_valid_nets)+'netw_'+str(nTrials)+'trials_logscale.png', dpi=300)
        #savefig('plots/'+'rotated_Bayes_subset0_'+str(n_valid_nets)+'netw_'+str(nTrials)+'trials_logscale.png', dpi=300)

        #file = open('plots/rotated_Home_data_Bayes_all_'+str(n_valid_nets)+'netw_'+str(nTrials)+'trials.txt', 'w'); pickle.dump(rotated_home,file,0); file.close()
        #file = open('plots/rotated_Away_data_Bayes_all_'+str(n_valid_nets)+'netw_'+str(nTrials)+'trials.txt', 'w'); pickle.dump(rotated_away,file,0); file.close()
        #file = open('plots/rotated_Home_data_Bayes_subset0_'+str(n_valid_nets)+'netw_'+str(nTrials)+'trials.txt', 'w'); pickle.dump(rotated_home,file,0); file.close()
        #file = open('plots/rotated_Away_data_Bayes_subset0_'+str(n_valid_nets)+'netw_'+str(nTrials)+'trials.txt', 'w'); pickle.dump(rotated_away,file,0); file.close()




def rotated_plot_bayes_vector(x_array, y_array, seqstart, goal_index_array, seqend, center_len, n_valid_nets, nTrials, L_maze_cm, n_spatial_bins, dpisize):
        n_grid = 80 # center_len # 
        rotated_home = zeros([2*center_len, 2*center_len]) # corresponds to an area of [- Lmaze_cm, Lmaze_cm]^2 
        rotated_away = zeros([2*center_len, 2*center_len]) # center_len = 100 (3.5cm - bins)
        alpha_all = zeros([n_valid_nets, nTrials])
        seqdist_all = zeros([n_valid_nets, nTrials])
        scaling_all = zeros([n_valid_nets, nTrials])
        k_end = zeros([n_valid_nets, nTrials])
        l_end = zeros([n_valid_nets, nTrials])

        size_factor = 1.0 # center_len / float(n_spatial_bins)

        figure(figsize=(6,3), dpi=dpisize)


        for i_netw in xrange(n_valid_nets): 
            alpha = zeros(nTrials)
            for i_trial in xrange(nTrials): 

                print "i_netw, i_trial = ", i_netw, i_trial

                if mod(i_trial, 2) == 0: # Home trial
                    subplot(121)

                    i_start, j_start = net_init.xypos_maze(seqstart[i_netw][i_trial][0], n_grid, L_maze_cm) # Bayes with data manager input

                    k = zeros(len(x_array[i_netw, i_trial, :]))
                    l = zeros(len(x_array[i_netw, i_trial, :]))
                    xs, ys = net_init.xypos_maze(seqstart[i_netw][i_trial][0], n_grid, L_maze_cm) # For results from data manager
                    xh, yh = net_init.xypos_maze(goal_index_array[i_netw][0], n_grid, L_maze_cm)
                    xe, ye = net_init.xypos_maze(seqend[i_netw][i_trial][0], n_grid, L_maze_cm)

                    seq_dist = np.sqrt((xe-xs)**2 + (ye-ys)**2)
                    home_dist_start = np.sqrt((xh-xs)**2 + (yh-ys)**2)

                    if home_dist_start > 0:
                        alpha[i_trial] = np.arccos( (0 * (xh-xs) + 1*(yh-ys)) / (1*home_dist_start) )
                        scaling = (0.25*center_len / home_dist_start) # ???
                    else:
                        alpha[i_trial] = 0
                        scaling = 0
                    if xh - xs > 0:
                        alpha[i_trial] = 2*pi - alpha[i_trial] 

                    alpha[i_trial] *= -1

                    #print "alpha[i_trial], scaling = ", alpha[i_trial], scaling

                    for ind in xrange(len(x_array[i_netw, i_trial, :])):
                        k[ind] = cos(alpha[i_trial]) * (x_array[i_netw, i_trial, ind] - i_start) * size_factor - sin(alpha[i_trial]) * (y_array[i_netw, i_trial, ind] - j_start) * size_factor # Bayes
                        l[ind] = sin(alpha[i_trial]) * (x_array[i_netw, i_trial, ind] - i_start) * size_factor + cos(alpha[i_trial]) * (y_array[i_netw, i_trial, ind] - j_start) * size_factor
                        k[ind] = k[ind] * scaling + 0.5*center_len # for connected point plots
                        l[ind] = l[ind] * scaling + 0.5*center_len #

                    plot(k, l, 'k-', lw=0.1) # lw=1 # 0.5 # 0.25

                #'''#
                else: # Random trial
                    subplot(122)

                    i_start, j_start = net_init.xypos_maze(seqstart[i_netw][i_trial][0], n_grid, L_maze_cm) # Bayes

                    print "i_start = ", i_start
                    print "j_start = ", j_start

                    k = zeros(len(x_array[i_netw, i_trial, :]))
                    l = zeros(len(x_array[i_netw, i_trial, :]))
                    xs, ys = net_init.xypos_maze(seqstart[i_netw][i_trial][0], n_grid, L_maze_cm) # new - Bayes
                    xe, ye = net_init.xypos_maze(seqend[i_netw][i_trial][0], n_grid, L_maze_cm)
                    seq_dist = np.sqrt((xe-xs)**2 + (ye-ys)**2)

                    if i_trial > 2: ### old: > 2:
                            xh, yh = net_init.xypos_maze(goal_index_array[i_netw][i_trial-2], n_grid, L_maze_cm) # new - Bayes
                            prev_rand_dist_start = np.sqrt((xh-xs)**2 + (yh-ys)**2)
                            print "Previous Random: xh, yh = ", xh, yh
                            print "prev_rand_dist_start = ", prev_rand_dist_start

                            if prev_rand_dist_start > 0:
                                alpha[i_trial] = np.arccos( (0 * (xh-xs) + 1*(yh-ys)) / (1*prev_rand_dist_start) )
                                scaling = (0.25*center_len / prev_rand_dist_start)     
                            else:
                                alpha[i_trial] = 0
                                scaling = 0    
                            if xh - xs > 0:
                                alpha[i_trial] = 2*pi - alpha[i_trial] 

                            alpha[i_trial] *= -1

                            for ind in xrange(len(x_array[i_netw, i_trial, :])):
                                k[ind] = cos(alpha[i_trial]) * (x_array[i_netw, i_trial, ind] - i_start) * size_factor - sin(alpha[i_trial]) * (y_array[i_netw, i_trial, ind] - j_start) * size_factor # Bayes
                                l[ind] = sin(alpha[i_trial]) * (x_array[i_netw, i_trial, ind] - i_start) * size_factor + cos(alpha[i_trial]) * (y_array[i_netw, i_trial, ind] - j_start) * size_factor
                                k[ind] = k[ind] * scaling + 0.5*center_len # for connected point plots
                                l[ind] = l[ind] * scaling + 0.5*center_len #

                    plot(k, l, 'k-', lw=0.1) # lw=1 # 0.5 # 0.25

                #'''
                alpha_all[i_netw, i_trial] = alpha[i_trial]
                seqdist_all[i_netw, i_trial] = seq_dist
                scaling_all[i_netw, i_trial] = scaling
                #k_end[i_netw, i_trial] = k[sortinds[-1]]
                #l_end[i_netw, i_trial] = l[sortinds[-1]]

        #'''#
	    #rotated_home -= 0.1*(rotated_home <> 0.1) # old
        #print "rotated_home.max() = ", rotated_home.max()
        #print "rotated_home.min() = ", rotated_home.min()
        #print "rotated_away.max() = ", rotated_away.max()

        #matshow(transpose((rotated_home[int(0.5*center_len) : int(1.5*center_len), int(0.5*center_len) : int(1.5*center_len)])), origin ='lower', fignum=False, cmap=nc.inferno)# , norm=colors.LogNorm() ### bayes without logscale
        #matshow(transpose((0.0001 + rotated_home[int(0.5*center_len) : int(1.5*center_len), int(0.5*center_len) : int(1.5*center_len)])), origin ='lower', fignum=False, cmap=nc.inferno , norm=colors.LogNorm()) ### bayes with logscale

        subplot(121)
        ax = gca()
        center_resol = L_maze_cm / float(center_len)
        ax.set_xticks([0.5*center_len]) # [0, , center_len]
        ax.set_xticklabels(['$\longleftarrow$ Perpendicular to Home $\longrightarrow$']) # [-0.5*center_len*center_resol, 0, 0.5*center_len*center_resol]
        ax.set_yticks([0.25*center_len, 0.75*center_len])
        ax.set_yticklabels(['Away from \n $\longleftarrow$  $\quad$ Home', 'Towards $\quad$ \n Home $\longrightarrow$'], rotation='vertical')
        ax.xaxis.set_ticks_position('bottom') #'''
        ax.xaxis.set_tick_params(length=0)
        ax.yaxis.set_tick_params(length=0)
        ax.set_aspect('equal')
        plot([0.5*center_len, 0.5*center_len], [0, center_len], '--', color='k', linewidth=0.5)
        plot([0, center_len], [0.5*center_len, 0.5*center_len], '--', color='k', linewidth=0.5)
        plot([0, center_len], [0.75*center_len, 0.75*center_len], '--', color='k', linewidth=0.5)
        xlim([0, center_len])
        ylim([0, center_len])

        #colorbar(shrink=0.4)
        for label in ax.get_xticklabels() + ax.get_yticklabels(): 
            label.set_fontsize(8)
        title('Home events rotated \n to Home well location', fontsize=8)


        subplot(122)
        #matshow(transpose((1+rotated_away[int(0.5*center_len) : int(1.5*center_len), int(0.5*center_len) : int(1.5*center_len)])), origin ='lower', fignum=False, norm=colors.LogNorm(), cmap=nc.inferno) # old
        #matshow(transpose((rotated_away[int(0.5*center_len) : int(1.5*center_len), int(0.5*center_len) : int(1.5*center_len)])), origin ='lower', fignum=False, cmap=nc.inferno) # , norm=colors.LogNorm() ### bayes
        ax = gca()
        ax.set_xticks([0.5*center_len]) # [0, , center_len]
        ax.set_xticklabels(['$\longleftarrow$ Perpendicular to prev. Random $\longrightarrow$']) # [-0.5*center_len*center_resol, 0, 0.5*center_len*center_resol]
        ax.set_yticks([0.2*center_len, 0.8*center_len])
        ax.set_yticklabels(['$\quad$ Away from \n $\longleftarrow$ prev. Random' , 'Towards $\qquad \quad$ \n prev. Random $\longrightarrow$'], rotation='vertical')
        ax.xaxis.set_ticks_position('bottom') #'''
        ax.xaxis.set_tick_params(length=0)
        ax.yaxis.set_tick_params(length=0)
        ax.set_aspect('equal')
        plot([0.5*center_len, 0.5*center_len], [0, center_len], '--', color='k', linewidth=0.5)
        plot([0, center_len], [0.5*center_len, 0.5*center_len], '--', color='k', linewidth=0.5)
        plot([0, center_len], [0.75*center_len, 0.75*center_len], '--', color='k', linewidth=0.5)
        xlim([0, center_len])
        ylim([0, center_len])

        for label in ax.get_xticklabels() + ax.get_yticklabels(): 
            label.set_fontsize(8)
        title('Away events rotated \n  to previous random location', fontsize=8)
        '''#
        axins = inset_axes(ax, width="5%", height="100%",loc=3, bbox_to_anchor=(1.05, 0., 1, 1), bbox_transform=ax.transAxes, borderpad=0)
        cbar = colorbar(cax=axins) #fraction = 0.05)
        cbar.set_ticks([1, 2, 11, 101, 1001])
        cbar.set_ticklabels([0, 1, 10, 100, 1000])
        cbar.ax.set_ylabel('Number of sequences', fontsize=8, rotation='vertical', position=(0, 0.5), va='bottom')
        axes(cbar.ax)
        yticks(fontsize=8)
        '''

        #if savefigs:
        savefig('plots/'+'rotated_Bayes_vector_subset0_'+str(n_valid_nets)+'netw_'+str(nTrials)+'trials.png', dpi=300)
        #savefig('plots/'+'rotated_Bayes_vector_all_'+str(n_valid_nets)+'netw_'+str(nTrials)+'trials.png', dpi=300)

        #file = open('plots/rotated_Home_data_Bayes_subset0.txt', 'w'); pickle.dump(rotated_home,file,0); file.close()
        #file = open('plots/rotated_Home_data_Bayes_vector_subset0_'+str(n_valid_nets)+'netw_'+str(nTrials)+'trials.txt', 'w'); pickle.dump(rotated_home,file,0); file.close()



def replot_bayes():
    ion()
    center_len = 100
    dpisize=300

    file = open('plots/rotated_Home_data_Bayes_subset0_12netw_40trials.txt')
    rh=load(file)
    file.close()
    file = open('plots/rotated_Away_data_Bayes_subset0_12netw_40trials.txt')
    ra=load(file)
    file.close()

    figure(figsize=(6,3), dpi=dpisize)
    subplot(121)
    matshow(transpose((1 + rh[int(0.5*center_len) : int(1.5*center_len), int(0.5*center_len) : int(1.5*center_len)])), origin ='lower', fignum=False, cmap=nc.inferno , norm=colors.LogNorm()) ### bayes with logscale
    ax = gca()
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
    axins = inset_axes(ax, width="5%", height="100%",loc=3, bbox_to_anchor=(-0.35, 0., 1, 1), bbox_transform=ax.transAxes, borderpad=0)
    cbar = colorbar(cax=axins) #fraction = 0.05)
    print "rh.max() = ", rh.max()
    cbar.set_ticks([1, 2, 11, 101, 1])
    cbar.set_ticklabels([0, 1, 10, 100, 1000])
    #cbar.ax.set_ylabel('Posterior Probability sum', fontsize=8, rotation='vertical', position=(0, 0.5), va='bottom')
    axes(cbar.ax)
    yticks(fontsize=8)



    subplot(122)
    matshow(transpose((1  + ra[int(0.5*center_len) : int(1.5*center_len), int(0.5*center_len) : int(1.5*center_len)])), origin ='lower', fignum=False, cmap=nc.inferno, norm=colors.LogNorm()) ### bayes with logscale
    ax = gca()
    ax.set_xticks([0.5*center_len]) # [0, , center_len]
    ax.set_xticklabels(['$\longleftarrow$ Perpendicular to prev. Random $\longrightarrow$']) 
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
    #'''#
    axins = inset_axes(ax, width="5%", height="100%",loc=3, bbox_to_anchor=(1.05, 0., 1, 1), bbox_transform=ax.transAxes, borderpad=0)
    cbar = colorbar(cax=axins) #fraction = 0.05)
    print "ra.max() = ", ra.max()
    cbar.set_ticks([1, 2, 11, 101, 1001])
    cbar.set_ticklabels([0, 1, 10, 100, 1000])
    cbar.ax.set_ylabel('Posterior Probability sum', fontsize=8, rotation='vertical', position=(0, 0.5), va='bottom')
    axes(cbar.ax)
    yticks(fontsize=8)
    #'''

    #if savefigs:
    #savefig('plots/'+'rotated_Bayes_12netws.png', dpi=300)
    #savefig('plots/'+'rotated_Bayes_subset0_12netws.png', dpi=300)
    savefig('plots/'+'rotated_Bayes_subset0_12netw_40trials_new.png', dpi=300)

    ioff()
    show()





























