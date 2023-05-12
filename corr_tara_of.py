from __future__ import print_function, division

import numpy as np
import pickle as pkl
from collections import Counter

import random
from scipy import *
from scipy.stats import *
import scipy.io
import sys, os

import time
import copy

from scipy.optimize import minimize

input_dir = "Tara_OF/"
behave_dir = "Tara_OF/"

# Pellet reaching
rodent_files = ["BinOasis__1205_performing",
               "BinOasis__1217_performing",
               "BinOasis__1232_observing",
               "BinOasis__1244_observing"]

rodent_behave = ['Behavioral annotation_1205_new_behaviours',
                'Behavioral annotation_1217_New_behaviours_n',
                'Behavioral annotation_1232_new',
                'Behavioral annotation_1244']


infile = sys.argv[1]
print("Name: ", infile)

file_index = np.array([int(sys.argv[2]), int(sys.argv[3])])
which_files = np.arange(file_index[0], file_index[1]+1, 1)
print("Which files: ", which_files)

captureframerate = sys.argv[4]
print("Capture frame rate: ", captureframerate)

pellet_condition = sys.argv[5]
print(pellet_condition, int(pellet_condition)==1)

if int(pellet_condition)==1:
    PERFORMANCE = True
else:
    PERFORMANCE = False

basename = '%s_Correlations'%(infile)

if(PERFORMANCE):
    basename = '%s_PERF'%basename
else:
    basename = '%s_OBSV'%basename


# Open field task
rodent_tasks = ['Running (CW)', 'Running (CCW)', 'Turning (CW)', 'Turning (CCW)', 'Tail touch', 'Foraging', 'Grooming', 'Eating/social touch', 'Looking down on the edges', 'Stationary', 'Pitch up', 'Squirming', 'Twitching', 'Running(CW)', 'Running(CCW)']

# Load binarized activities
S = np.array([])
for i in list(which_files):
    if S.shape[0]>0:
        S = np.append(S, np.loadtxt("%s%s.csv"%(input_dir,rodent_files[i]), delimiter=',')[:,1:], axis=0)
    else:
        S = np.loadtxt("%s%s.csv"%(input_dir,rodent_files[i]), delimiter=',')[:,1:]
S = S.T
print(S.shape)


# Make cellnames
cellnames = [infile+'_%d'%(i+1) for i in np.arange(S.shape[0])]

# Load behaviors
binning_time = captureframerate
rodent_behaviours = []
possiblecovariates = {}

for i in list(which_files):
    expt_times = np.loadtxt("%s%s.csv"%(input_dir,rodent_files[0]), delimiter=',')[:,0]
    timestamps = np.zeros(len(expt_times))
    
    behave_names = pkl.load( open(input_dir+rodent_behave[i]+'_names.pkl','rb') )
    behave_times = pkl.load( open(input_dir+rodent_behave[i]+'_times.pkl','rb') )
    
    print(rodent_files[i], behave_names)
    for names in behave_names:
        if names in rodent_tasks:
            task_index = behave_names.index(names)

            if names == "Running(CW)": names = "Running (CW)"
            if names == "Running(CCW)": names = "Running (CCW)"

            t_start = [behave_times[task_index][t][0] for t in np.arange(len(behave_times[task_index])) if behave_times[task_index][t][1]==1]
            t_stop = [behave_times[task_index][t][0] for t in np.arange(len(behave_times[task_index])) if behave_times[task_index][t][1]==2]
            
            start_index = np.around((np.around(t_start,2)/0.05),0).astype('int')
            stop_index = np.around((np.around(t_stop,2)/0.05),0).astype('int')
            
            # this is to delete some
            if ((len(start_index)!=0) and (np.array(behave_times[task_index])[:,1][-1]==1)):
                start_index = np.delete(start_index,-1)
            
            if ((len(start_index)!=0) and (np.array(behave_times[task_index])[:,1][0]==2)):
                stop_index = np.delete(stop_index,0)
            
            if (len(start_index)>len(stop_index)):
                zero_index = np.where(np.diff(np.array(behave_times[task_index])[:,1])==0)[0]
                for z_index in zero_index:
                    start_index = np.delete(start_index,np.where(np.around(t_start,2)==np.around(behave_times[task_index][z_index][0],2))[0])
            
            if (len(start_index)<len(stop_index)):
                zero_index = np.where(np.diff(np.array(behave_times[task_index])[:,1])==0)[0]
                for z_index in zero_index:
                    stop_index = np.delete(stop_index,np.where(np.around(t_stop,2)==np.around(behave_times[task_index][z_index+1][0],2))[0])
            
            for behave_time_index in np.arange(len(start_index)):
                timestamps[start_index[behave_time_index]:stop_index[behave_time_index]] = rodent_tasks.index(names)+1

            try:
                dummy = timestamps.copy()
                cov_array = np.zeros_like(dummy)
                cov_array[np.where(dummy == rodent_tasks.index(names)+1)[0]] = 1
                    
                possiblecovariates[names].append(cov_array)
                possiblecovariates[names] = np.array(possiblecovariates[names]).flatten()
                
            except:
                possiblecovariates[names] = []
                    
                dummy = timestamps.copy()
                cov_array = np.zeros_like(dummy)
                cov_array[np.where(dummy == rodent_tasks.index(names)+1)[0]] = 1
                    
                possiblecovariates[names].append(cov_array)
                possiblecovariates[names] = np.array(possiblecovariates[names]).flatten()

    rodent_behaviours.append(timestamps)

for names in possiblecovariates.keys():
    print(names, len(possiblecovariates[names]))



whichcovariates = {}

#rodent_tasks = ['Running (CW)', 'Running (CCW)', 'Turning (CW)', 'Turning (CCW)', 'Tail touch', 'Foraging', 'Grooming', 'Eating/social touch',
#'Looking down on the edges', 'Stationary', 'Pitch up', 'Squirming', 'Twitching']

# Open field
whichcombinations = {}
if PERFORMANCE:
   whichcombinations['Running (CW)'] = ['Running (CW)']
   whichcombinations['Running (CCW)'] = ['Running (CCW)']
   whichcombinations['Turning (CCW)'] = ['Turning (CCW)']
   whichcombinations['Turning (CCW)'] = ['Turning (CCW)']
   whichcombinations['Tail touch'] = ['Tail touch']
   whichcombinations['Foraging'] = ['Foraging']
   whichcombinations['Grooming'] = ['Grooming']
   whichcombinations['Eating/social touch'] = ['Eating/social touch']
   whichcombinations['Looking down on the edges'] = ['Looking down on the edges']
   whichcombinations['Stationary'] = ['Stationary']
   whichcombinations['Pitch up'] = ['Pitch up']
else:
   whichcombinations['Running (CW)'] = ['Running (CW)']
   whichcombinations['Running (CCW)'] = ['Running (CCW)']
   whichcombinations['Squirming'] = ['Squirming']

squirm_name = 'Squirming'


listofthingsthatshouldbeprocessed = []
for wc in list(whichcombinations.keys()):
    ara = whichcombinations[wc]
    for guy in ara:
        if(guy not in listofthingsthatshouldbeprocessed):
            listofthingsthatshouldbeprocessed.append(guy)

print(listofthingsthatshouldbeprocessed)



print('Here are the keys in the mat file:')
ckeys = sort(list(possiblecovariates.keys()))
for k in ckeys:
    print(k)

ckeys = list(possiblecovariates.keys())

allresultskeys = zeros((len(cellnames), len(list(whichcombinations.keys()))))

outputdict = {}
outputdict['filename'] = infile
outputdict['cellnames'] = cellnames

kkk = sort(list(whichcombinations.keys()))
for j in range(len(kkk)):
    print('Going for', kkk[j])
    thekeyname = kkk[j]
    outputdict['cellnames'] = cellnames
    goodkeys = whichcombinations[thekeyname]

    covs = []
    chkkey = goodkeys[0]
    covs = possiblecovariates[chkkey]
    if len(goodkeys)>1:
        for i in range(1,len(goodkeys),1):
            chkkey = goodkeys[i]
            covs = covs + possiblecovariates[chkkey]
    covs = covs.astype('bool')
    print('Non-zero values: ', np.sum(covs))
    
    if(PERFORMANCE):
        outputdict['mean-%s'%kkk[j]] = np.sum(S[:,np.where(covs)[0]],axis=0)/float(len(np.where(covs)[0]))
        outputdict['corr-%s'%kkk[j]] = np.corrcoef(S[:,np.where(covs)[0]])
    else:
        outputdict['mean-%s'%kkk[j]] = np.sum(S[:,np.where(covs)[0]],axis=0)/float(len(np.where(covs)[0]))
        outputdict['corr-%s'%kkk[j]] = np.corrcoef(S[:,np.where(covs)[0]])

        covs_nosquirm = covs*(possiblecovariates[squirm_name]==0)
        outputdict['mean-%s-nosquirm'%kkk[j]] = np.sum(S[:,np.where(covs_nosquirm)[0]],axis=0)/float(len(np.where(covs_nosquirm)[0]))
        outputdict['corr-%s-nosquirm'%kkk[j]] = np.corrcoef(S[:,np.where(covs_nosquirm)[0]])

scipy.io.savemat('%s.mat'%(basename), outputdict)
