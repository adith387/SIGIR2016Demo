import numpy as np
import pickle
import matplotlib
#matplotlib.use('Agg')
import matplotlib.patches
import matplotlib.pyplot as plt
import sys
import argparse
import scipy
import scipy.stats
from matplotlib.font_manager import FontProperties

fontP = FontProperties()
fontP.set_size('14')

def plot_mse_stderr(PKL, approach_list, plot_estimate, color_string = 'bmgrcky'):
    x = np.array(PKL[2]) + 1
        
    legendList = []
    legendHandles = []
    
    if plot_estimate:
        l = plt.plot(x, PKL[-1]*np.ones(np.shape(x)), color='gold', rasterized=True, linewidth=2.0)
        legendHandles.append(matplotlib.patches.Patch(color = 'gold', label = 'Target'))
    
    for approachIndex, approach in enumerate(approach_list):
        snipPoint = 0
        prefix = ''
        if '-SN' in approach:
            prefix = 'sn'

        approachName = None
        if 'Direct' in approach:
            approachName = 'RP'
            limit = approach.split('_', 2)[2]
            approachName += '(' + limit +')'
            limit = int(limit)
            limit = limit + int(limit / 10)
            snipPoint = np.where(x == limit)[0]
        elif 'IPS' in approach:
            approachName = 'IPS'
        elif 'Truth' == approach:
            approachName = 'OnPolicy'
        elif 'Gamma' in approach:
            approachName = 'Slates'
        elif 'DoublyRobust' in approach:
            approachName = 'DR'
        else:
            approachName = approach.replace('_','-')

        print(approach, prefix+approachName)
        legendList.append(prefix + approachName)

        approachDict = PKL[1]

        trials = PKL[0]
        print("Approach: %s NumTrials: %d" % (approach, trials), flush = True)
        mu = None
        if plot_estimate:
            mu = PKL[4][approachDict[approach],snipPoint:]
        else:
            values = PKL[3][approachDict[approach],0:trials,snipPoint:]
            values = np.log10(np.sqrt(values))
            mu = np.mean(values,axis=0)
            
        currColor = color_string[approachIndex]
        l = plt.plot(x[snipPoint:], mu, color=currColor, rasterized=True, linewidth=2.0)
        legendHandles.append(matplotlib.patches.Patch(color = currColor, label = prefix + approachName))
            
    plt.legend(handles = legendHandles, loc='best', prop = fontP, ncol = 3)
    plt.xlabel('Number of samples (n)')
    if plot_estimate:
        plt.ylabel('Estimate')
    else:
        plt.ylabel('log(RMSE)')
    ax = plt.gca()
    
    #ax.set_xscale("log")
    
    
if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', action='store',
                        help='pkl file path with data')
    parser.add_argument('--save', dest='save', action='store_true')
    parser.set_defaults(save=False)
    parser.add_argument('--mode', dest='mode', action='store')
    parser.set_defaults(mode='estimate')
    
    Args = parser.parse_args()
    PKL = pickle.load(open(Args.path,"rb"))
    
    plotEstimate = False
    if Args.mode == 'estimate':
        plotEstimate = True
    
    titleString = None
    if plotEstimate:
        titleString = 'REVENUE \t 3 slots, 7 candidates'
    else:
        titleString = 'Avg. Error over %d trials \t 3 slots, 7 candidates' % PKL[0]
    
    plt.rc('font', size = 16)
    plt.rc('text', usetex=True)
    plt.rc('font', family = 'serif')
    plt.axes([.13,.13,.78,.78])

    plt.suptitle(titleString)
    approachList = ['Truth', 'Direct_ridge_100', 'Direct_ridge_1000', 'Direct_ridge_10000']#, 'Direct_ridge_100000']
    plot_mse_stderr(PKL,approachList,plotEstimate)
    if Args.save:
        plt.savefig(Args.path+"RP.png", format="png", dpi=100)
    else:
        plt.show()
    plt.clf()
    
    plt.suptitle(titleString)
    approachList = ['Truth', 'Direct_ridge_100', 'Direct_ridge_1000', 'Direct_ridge_10000', 'IPS']
    plot_mse_stderr(PKL,approachList,plotEstimate)
    if Args.save:
        plt.savefig(Args.path+"IPS.png", format="png", dpi=100)
    else:
        plt.show()
    plt.clf()
        
    plt.suptitle(titleString)
    approachList = ['IPS','IPS-SN']
    plot_mse_stderr(PKL,approachList,plotEstimate,'km')
    if Args.save:
        plt.savefig(Args.path+"wIPS.png", format="png", dpi=100)
    else:
        plt.show()
    plt.clf()
    
    plt.suptitle(titleString)
    approachList = ['IPS-SN','Gamma','Gamma-SN']
    plot_mse_stderr(PKL,approachList,plotEstimate,'mgc')
    if Args.save:
        plt.savefig(Args.path+"Gamma.png", format="png", dpi=100)
    else:
        plt.show()
    plt.clf()
    
    plt.suptitle(titleString)
    approachList = ['Direct_ridge_100','Direct_ridge_1000', 'Direct_ridge_10000', 'IPS', 'DoublyRobust_ridge']
    plot_mse_stderr(PKL,approachList,plotEstimate,'mgcrky')
    if Args.save:
        plt.savefig(Args.path+"DR.png", format="png", dpi=100)
    else:
        plt.show()
