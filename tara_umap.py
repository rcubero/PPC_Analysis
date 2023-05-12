# to make things py2 and py3 compatible
from __future__ import print_function, division

# major packages needed
import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
import matplotlib as mpl
import itertools, os, sys
import pickle as pkl

from sklearn import manifold
from collections import Counter
from itertools import combinations, product
from scipy import io, ndimage, optimize, signal, linalg, spatial
from scipy.interpolate import interp1d

input_dir = "/Users/rcubero/Google Drive/Seven_1604-1704/"
output_dir = "/Users/rcubero/Dropbox/Tuce_results/Seven_results/"

seven_files = ["Seven_1704_S1_BinOasis__1242_performing__renumbered", "Seven_1704_S1_BinOasis__1253_performing__renumbered",
               "Seven_1604_S2_BinOasis__1428_performing__renumbered", "Seven_1604_S2_BinOasis__1439_performing__renumbered",
               "Seven_1604_S3_BinOasis__1953_performing__renumbered", "Seven_1604_S3_BinOasis__2005_performing__renumbered"]

firingrate_std = 5
downsamplelength = 10
perplexity_value = 50
zscore_cutoff = 3.0

randomized_range_mean_std = np.loadtxt("%s%s.rmsr"%(output_dir,"randomized_range_mean_std_longer_linear"))
f_mean = interp1d(randomized_range_mean_std[0], randomized_range_mean_std[1], kind="cubic")
f_std = interp1d(randomized_range_mean_std[0], randomized_range_mean_std[2], kind="cubic")

relevance = {}
N_spikes = {}
relevance_zscore = {}
filtered_significant_neuron = {}
for filenames in seven_files:
    binoasis = pd.read_csv("%s%s.csv"%(input_dir,filenames), header=-1)
    relevance[filenames] = np.loadtxt("%s%s.msr"%(output_dir,filenames))
    N_spikes[filenames] = binoasis.sum(axis=0).values[1:]
    relevance_zscore[filenames] = []
    filtered_significant_neuron[filenames] = []
    for i in np.arange(len(N_spikes[filenames])):
        if N_spikes[filenames][i] >= 20:
            z_score = (relevance[filenames][i]-f_mean(N_spikes[filenames][i]))/(f_std(N_spikes[filenames][i]))
            relevance_zscore[filenames].append(z_score)
            if z_score >= zscore_cutoff:
                filtered_significant_neuron[filenames].append(1)
            else:
                filtered_significant_neuron[filenames].append(0)
        else:
            relevance_zscore[filenames].append(0)
            relevance[filenames][i] = 0
            filtered_significant_neuron[filenames].append(0)

significant_neurons = np.zeros((len(seven_files),len(relevance[seven_files[0]])))
for i in np.arange(len(seven_files)):
    significant_neurons[i] = filtered_significant_neuron[seven_files[i]]
    significant_neuron_index = np.where(np.sum(significant_neurons,axis=0)==6)[0]+1


binoasis = pd.read_csv("%s%s.csv"%(input_dir,seven_files[0]), header=-1)
task_index = np.ones(len(binoasis))
for j in np.arange(1, binoasis.shape[1], 1):
    binoasis.iloc[:,j] = ndimage.filters.gaussian_filter1d(binoasis.iloc[:,j], firingrate_std)
    # for normalisation, comment if unnecessary
    # binoasis.iloc[:,j] = (binoasis.iloc[:,j] - np.amin(binoasis.iloc[:,j]))/(np.amax(binoasis.iloc[:,j]) - np.amin(binoasis.iloc[:,j]))
for i in np.arange(1,len(seven_files)):
    appended_oasis = pd.read_csv("%s%s.csv"%(input_dir,seven_files[i]), header=-1)
    for j in np.arange(1, appended_oasis.shape[1], 1):
        appended_oasis.iloc[:,j] = ndimage.filters.gaussian_filter1d(appended_oasis.iloc[:,j], firingrate_std)
        # for normalisation, comment if unnecessary
        # appended_oasis.iloc[:,j] = (appended_oasis.iloc[:,j] - np.amin(appended_oasis.iloc[:,j]))/(np.amax(appended_oasis.iloc[:,j]) - np.amin(appended_oasis.iloc[:,j]))
    binoasis = binoasis.append(appended_oasis, ignore_index=True)
    task_index = np.append(task_index, (i+1)*np.ones(len(appended_oasis)))

time_data = np.array(binoasis.iloc[:, significant_neuron_index])
downsampling = (np.mod(np.arange(len(task_index)),downsamplelength)==0).astype('int')

firing_rate_cutoff = [0.1, 0.15, 0.2, 0.25, 0.30]
significant_cutoff = [3, 4, 5, 6]
fs_cutoff = list(product(firing_rate_cutoff, significant_cutoff))

for n in np.arange(len(fs_cutoff)):
    FR_threshold, count_threshold = fs_cutoff[n]
    
    Sfilt = (time_data >= FR_threshold)*1.
    Scounts = np.sum(Sfilt, 1)
    Skept = (Scounts >= count_threshold).astype('int')
    kept_index = downsampling*Skept
    print(fs_cutoff[n], np.sum(kept_index))
    
    if np.sum(kept_index)>2:
        tsne = manifold.TSNE(n_components=2, perplexity=perplexity_value, init='pca', random_state=0)
        Y_tsne = tsne.fit_transform(np.array(binoasis.iloc[np.where(kept_index)[0], significant_neuron_index]))
        basename = 'Seven_tSNE_NonNormed_FRStd_%04d_DwnSamp_%04d_FRCut_%04d_Count_%04d'%(firingrate_std, downsamplelength, 100*FR_threshold, count_threshold)
        np.savetxt('%s/FiringRateVsCounts/%s_%s.txt'%(output_dir,basename,"TSNE_coordinates"),Y_tsne)
    else:
        np.savetxt('%s/FiringRateVsCounts/%s_%s.txt'%(output_dir,basename,"TSNE_coordinates"),np.array([]))

