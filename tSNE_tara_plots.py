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

input_dir = "/Users/rcubero/Google Drive/Tara_7x10mins_recordings_perfromanceonly/"
output_dir = "/Users/rcubero/Dropbox/Tuce_results/Tara_results/"

seven_files = ["Tara_S1_BinOasis__1157_performing__renumbered", "Tara_S1_BinOasis__1208_performing__renumbered",
               "Tara_S3_BinOasis__1205_performing__renumbered", "Tara_S3_BinOasis__1217_performing__renumbered",
               "Tara_S4_BinOasis__1727_performing__renumbered"]

seven_titles = ["Tara_S1_1157", "Tara_S1_1208",
                "Tara_S3_1205", "Tara_S3_1217",
                "Tara_S4_1727"]

seven_colors = ["darkgreen", "lightgreen",
                "darkblue", "lightblue",
                "black"]

seven_colors2 = ["lightgrey", "lightcoral", "sienna", "orange", "goldenrod", "olive", "lawngreen", "seagreen","turquoise","teal","deepskyblue","deepskyblue","slategray","slateblue","indigo","violet","crimson", "red", "orange", "yellow", "blue", "green"]

seven_labels = ["unclassified", "Nosepoke (no food)", "Nosepoke/grasping to eat", "Eating on haunches", "Eating all 4's", "Grooming", "Turning (CW)", "Turning (CCW)", "Rearing", "Reaching", "Running (CW)", "Running (CCW)", "Tail touch", "Foraging", "Eating/social touch", "Looking down on the edges", "Stationary", "Pitch up", "Nosepoke", "Grasping", "Scratching"]

firingrate_std = 10
downsamplelength = 20
perplexity_value = 50
zscore_cutoff = 3.0

seven_behave = ['behavioral_annotation_1157', 'behavioral_annotation_1208', 'behavioral_annotation_1205', 'behavioral_annotation_1217', 'behavioral_annotation_1727']
seven_tasks = ["Nosepoke (no food)", "Nosepoke/grasping to eat", "Eating on haunches", "Eating all 4's", "Grooming", "Turning (CW)", "Turning (CCW)", "Rearing", "Reaching", "Running (CW)", "Running (CCW)", "Tail touch", "Foraging", "Eating/social touch", "Looking down on the edges", "Stationary", "Pitch up", "Nosepoke", "Grasping", "Scratching"]

binning_time = 0.05
seven_behaviours = []
for i in np.arange(len(seven_behave)):
    expt_times = pd.read_csv("%s%s.csv"%(input_dir,seven_files[i]), header=-1)[0].values
    timestamps = np.zeros(len(expt_times))
    
    behave_names = pkl.load(open(input_dir+seven_behave[i]+'_names.pkl','rb'), encoding='latin1')
    behave_times = pkl.load(open(input_dir+seven_behave[i]+'_times.pkl','rb'), encoding='latin1')
    for names in behave_names:
        if names in seven_tasks:
            if names == "Social touch": names = "Eating/social touch"
            if names == "Turning CCW": names = "Turning (CCW)"
            if names == "Turning CW": names = "Turning (CW)"
            task_index = behave_names.index(names)
            
            t_start = [behave_times[task_index][t][0] for t in np.arange(len(behave_times[task_index])) if behave_times[task_index][t][1]==1]
            t_stop = [behave_times[task_index][t][0] for t in np.arange(len(behave_times[task_index])) if behave_times[task_index][t][1]==2]
            
            start_index = np.around((np.around(t_start,2)/0.05),0).astype('int')
            stop_index = np.around((np.around(t_stop,2)/0.05),0).astype('int')
            
            if(len(start_index)!=len(stop_index)):
                zero_index = np.where(np.diff(np.array(behave_times[task_index])[:,1])==0)[0][0]
                start_index = np.delete(start_index,np.where(np.around(t_start,2)==np.around(behave_times[task_index][zero_index][0],2))[0])
            
            for behave_time_index in np.arange(len(start_index)):
                timestamps[start_index[behave_time_index]:stop_index[behave_time_index]] = seven_tasks.index(names)+1
    seven_behaviours.append(timestamps)
seven_behaviours = np.array(seven_behaviours).flatten()

randomized_range_mean_std = np.loadtxt("%s%s.rmsr"%(output_dir,"randomized_range_mean_std_longer_linear"))
f_mean = interp1d(randomized_range_mean_std[0], randomized_range_mean_std[1], kind="cubic")
f_std = interp1d(randomized_range_mean_std[0], randomized_range_mean_std[2], kind="cubic")

relevance = {}
N_spikes = {}
relevance_zscore = {}
for filenames in seven_files:
    binoasis = pd.read_csv("%s%s.csv"%(input_dir,filenames), header=-1)
    relevance[filenames] = np.loadtxt("%s%s.msr"%(output_dir,filenames))
    N_spikes[filenames] = binoasis.sum(axis=0).values[1:]
    relevance_zscore[filenames] = []
    for i in np.arange(len(N_spikes[filenames])):
        if N_spikes[filenames][i] >= 20:
            z_score = (relevance[filenames][i]-f_mean(N_spikes[filenames][i]))/(f_std(N_spikes[filenames][i]))
            relevance_zscore[filenames].append(z_score)
        else:
            relevance_zscore[filenames].append(0)
            relevance[filenames][i] = 0

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

downsampleindex = np.where(np.mod(np.arange(len(task_index)),downsamplelength)==0)[0]


#significant_cutoff = [3.0, 3.5, 4.0, 4.5, 5.0]
#perplexity_vals = [5, 30, 50, 100, 200, 500]
#sp_cutoff = list(product(significant_cutoff, perplexity_vals))
#
## Non-Normed Session
#fig, axs = plt.subplots(len(significant_cutoff), len(perplexity_vals), dpi=300)
#fig.set_size_inches(45, 30)
#
#n=0
#for i, ax in enumerate(fig.axes):
#    zscore_cutoff, perplexity_value = sp_cutoff[n]
#    basename = 'Tara_tSNE_NonNormed_FRStd_%04d_DwnSamp_%04d_ZCut_%04d_Perp_%04d'%(firingrate_std, downsamplelength, 10*zscore_cutoff, perplexity_value)
#    Y_tsne = np.loadtxt('%s/SignificanceVsPerplexity/%s_%s.txt'%(output_dir,basename,"TSNE_coordinates"))
#    for i in np.arange(len(seven_files)):
#        ax.scatter(Y_tsne[task_index[downsampleindex]==(i+1),0], Y_tsne[task_index[downsampleindex]==(i+1),1], s=10, color=seven_colors[i], label=seven_titles[i], rasterized=True)
#    ax.legend(loc="upper right")
#    ax.set_title('z-score: '+str(sp_cutoff[n][0])+', perplexity: '+ str(sp_cutoff[n][1]) )
#    n += 1
#plt.savefig('%sSignificanceVsPerplexity/%s.pdf'%(output_dir,"tSNE_significanceVsperplexity_NonNormed_sessions"), bbox_inches="tight", dpi=300)
#plt.close()
#
## Non-Normed Behaviour
#fig, axs = plt.subplots(len(significant_cutoff), len(perplexity_vals), dpi=300)
#fig.set_size_inches(45, 30)
#
#n=0
#for i, ax in enumerate(fig.axes):
#    zscore_cutoff, perplexity_value = sp_cutoff[n]
#    basename = 'Tara_tSNE_NonNormed_FRStd_%04d_DwnSamp_%04d_ZCut_%04d_Perp_%04d'%(firingrate_std, downsamplelength, 10*zscore_cutoff, perplexity_value)
#    Y_tsne = np.loadtxt('%s/SignificanceVsPerplexity/%s_%s.txt'%(output_dir,basename,"TSNE_coordinates"))
#    for i in np.arange(len(seven_labels)):
#        ax.scatter(Y_tsne[seven_behaviours[downsampleindex]==i,0], Y_tsne[seven_behaviours[downsampleindex]==i,1], s=10, color=seven_colors2[i], label=seven_labels[i], rasterized=True)
#    ax.legend(loc="upper right", fontsize=8)
#    ax.set_title('z-score: '+str(sp_cutoff[n][0])+', perplexity: '+ str(sp_cutoff[n][1]) )
#    n += 1
#plt.savefig('%sSignificanceVsPerplexity/%s.pdf'%(output_dir,"tSNE_significanceVsperplexity_NonNormed_behaviours"), bbox_inches="tight", dpi=300)
#plt.close()
#
## Normed Session
#fig, axs = plt.subplots(len(significant_cutoff), len(perplexity_vals), dpi=300)
#fig.set_size_inches(45, 30)
#
#n=0
#for i, ax in enumerate(fig.axes):
#    zscore_cutoff, perplexity_value = sp_cutoff[n]
#    basename = 'Tara_tSNE_Normed_FRStd_%04d_DwnSamp_%04d_ZCut_%04d_Perp_%04d'%(firingrate_std, downsamplelength, 10*zscore_cutoff, perplexity_value)
#    Y_tsne = np.loadtxt('%s/SignificanceVsPerplexity/%s_%s.txt'%(output_dir,basename,"TSNE_coordinates"))
#    for i in np.arange(len(seven_files)):
#        ax.scatter(Y_tsne[task_index[downsampleindex]==(i+1),0], Y_tsne[task_index[downsampleindex]==(i+1),1], s=10, color=seven_colors[i], label=seven_titles[i], rasterized=True)
#    ax.legend(loc="upper right")
#    ax.set_title('z-score: '+str(sp_cutoff[n][0])+', perplexity: '+ str(sp_cutoff[n][1]) )
#    n += 1
#plt.savefig('%sSignificanceVsPerplexity/%s.pdf'%(output_dir,"tSNE_significanceVsperplexity_Normed_sessions"), bbox_inches="tight", dpi=300)
#plt.close()
#
## Normed Behaviour
#fig, axs = plt.subplots(len(significant_cutoff), len(perplexity_vals), dpi=300)
#fig.set_size_inches(45, 30)
#
#n=0
#for i, ax in enumerate(fig.axes):
#    zscore_cutoff, perplexity_value = sp_cutoff[n]
#    basename = 'Tara_tSNE_Normed_FRStd_%04d_DwnSamp_%04d_ZCut_%04d_Perp_%04d'%(firingrate_std, downsamplelength, 10*zscore_cutoff, perplexity_value)
#    Y_tsne = np.loadtxt('%s/SignificanceVsPerplexity/%s_%s.txt'%(output_dir,basename,"TSNE_coordinates"))
#    for i in np.arange(len(seven_labels)):
#        ax.scatter(Y_tsne[seven_behaviours[downsampleindex]==i,0], Y_tsne[seven_behaviours[downsampleindex]==i,1], s=10, color=seven_colors2[i], label=seven_labels[i], rasterized=True)
#    ax.legend(loc="upper right", fontsize=8)
#    ax.set_title('z-score: '+str(sp_cutoff[n][0])+', perplexity: '+ str(sp_cutoff[n][1]) )
#    n += 1
#plt.savefig('%sSignificanceVsPerplexity/%s.pdf'%(output_dir,"tSNE_significanceVsperplexity_Normed_behaviours"), bbox_inches="tight", dpi=300)
#plt.close()


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
    significant_neuron_index = np.where(np.sum(significant_neurons,axis=0)==5)[0]+1


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
sp_cutoff = list(product(firing_rate_cutoff, significant_cutoff))

# Non-Normed Session
fig, axs = plt.subplots(len(firing_rate_cutoff), len(significant_cutoff), dpi=300)
fig.set_size_inches(45, 30)

n=0
for i, ax in enumerate(fig.axes):
    zscore_cutoff, perplexity_value = sp_cutoff[n]
    
    Sfilt = (time_data >= zscore_cutoff)*1.
    Scounts = np.sum(Sfilt, 1)
    Skept = (Scounts >= perplexity_value).astype('int')
    kept_index = np.where(downsampling*Skept)[0]
    
    basename = 'Tara_tSNE_NonNormed_FRStd_%04d_DwnSamp_%04d_FRCut_%04d_Count_%04d'%(firingrate_std, downsamplelength, 100*zscore_cutoff, perplexity_value)
    Y_tsne = np.loadtxt('%s/FiringRateVsCounts/%s_%s.txt'%(output_dir,basename,"TSNE_coordinates"))
    for i in np.arange(len(seven_files)):
        ax.scatter(Y_tsne[task_index[kept_index]==(i+1),0], Y_tsne[task_index[kept_index]==(i+1),1], s=10, color=seven_colors[i], label=seven_titles[i], rasterized=True)
    ax.legend(loc="upper right")
    ax.set_title('FR cutoff: '+str(sp_cutoff[n][0])+', Neuron Cutoff: '+ str(sp_cutoff[n][1]) )
    n += 1
plt.savefig('%sFiringRateVsCounts/%s.pdf'%(output_dir,"tSNE_firingratevscounts_NonNormed_sessions"), bbox_inches="tight", dpi=300)
plt.close()

# Non-Normed Behaviour
fig, axs = plt.subplots(len(firing_rate_cutoff), len(significant_cutoff), dpi=300)
fig.set_size_inches(45, 30)

n=0
for i, ax in enumerate(fig.axes):
    zscore_cutoff, perplexity_value = sp_cutoff[n]
    
    Sfilt = (time_data >= zscore_cutoff)*1.
    Scounts = np.sum(Sfilt, 1)
    Skept = (Scounts >= perplexity_value).astype('int')
    kept_index = np.where(downsampling*Skept)[0]
    
    basename = 'Tara_tSNE_NonNormed_FRStd_%04d_DwnSamp_%04d_FRCut_%04d_Count_%04d'%(firingrate_std, downsamplelength, 100*zscore_cutoff, perplexity_value)
    Y_tsne = np.loadtxt('%s/FiringRateVsCounts/%s_%s.txt'%(output_dir,basename,"TSNE_coordinates"))
    for i in np.arange(len(seven_labels)):
        ax.scatter(Y_tsne[seven_behaviours[kept_index]==i,0], Y_tsne[seven_behaviours[kept_index]==i,1], s=10, color=seven_colors2[i], label=seven_labels[i], rasterized=True)
    ax.legend(loc="upper right", fontsize=8)
    ax.set_title('FR cutoff: '+str(sp_cutoff[n][0])+', Neuron Cutoff: '+ str(sp_cutoff[n][1]) )
    n += 1
plt.savefig('%sFiringRateVsCounts/%s.pdf'%(output_dir,"tSNE_firingratevscounts_NonNormed_behaviours"), bbox_inches="tight", dpi=300)
plt.close()

# Normed Session
fig, axs = plt.subplots(len(firing_rate_cutoff), len(significant_cutoff), dpi=300)
fig.set_size_inches(45, 30)

n=0
for i, ax in enumerate(fig.axes):
    zscore_cutoff, perplexity_value = sp_cutoff[n]
    
    Sfilt = (time_data >= zscore_cutoff)*1.
    Scounts = np.sum(Sfilt, 1)
    Skept = (Scounts >= perplexity_value).astype('int')
    kept_index = np.where(downsampling*Skept)[0]
    
    basename = 'Tara_tSNE_Normed_FRStd_%04d_DwnSamp_%04d_FRCut_%04d_Count_%04d'%(firingrate_std, downsamplelength, 100*zscore_cutoff, perplexity_value)
    Y_tsne = np.loadtxt('%s/FiringRateVsCounts/%s_%s.txt'%(output_dir,basename,"TSNE_coordinates"))
    for i in np.arange(len(seven_files)):
        ax.scatter(Y_tsne[task_index[kept_index]==(i+1),0], Y_tsne[task_index[kept_index]==(i+1),1], s=10, color=seven_colors[i], label=seven_titles[i], rasterized=True)
    ax.legend(loc="upper right")
    ax.set_title('FR cutoff: '+str(sp_cutoff[n][0])+', Neuron Cutoff: '+ str(sp_cutoff[n][1]) )
    n += 1
plt.savefig('%sFiringRateVsCounts/%s.pdf'%(output_dir,"tSNE_firingratevscounts_Normed_sessions"), bbox_inches="tight", dpi=300)
plt.close()

# Normed Behaviour
fig, axs = plt.subplots(len(firing_rate_cutoff), len(significant_cutoff), dpi=300)
fig.set_size_inches(45, 30)

n=0
for i, ax in enumerate(fig.axes):
    zscore_cutoff, perplexity_value = sp_cutoff[n]
    
    Sfilt = (time_data >= zscore_cutoff)*1.
    Scounts = np.sum(Sfilt, 1)
    Skept = (Scounts >= perplexity_value).astype('int')
    kept_index = np.where(downsampling*Skept)[0]
    
    basename = 'Tara_tSNE_Normed_FRStd_%04d_DwnSamp_%04d_FRCut_%04d_Count_%04d'%(firingrate_std, downsamplelength, 100*zscore_cutoff, perplexity_value)
    Y_tsne = np.loadtxt('%s/FiringRateVsCounts/%s_%s.txt'%(output_dir,basename,"TSNE_coordinates"))
    for i in np.arange(len(seven_labels)):
        ax.scatter(Y_tsne[seven_behaviours[kept_index]==i,0], Y_tsne[seven_behaviours[kept_index]==i,1], s=10, color=seven_colors2[i], label=seven_labels[i], rasterized=True)
    ax.legend(loc="upper right", fontsize=8)
    ax.set_title('FR cutoff: '+str(sp_cutoff[n][0])+', Neuron Cutoff: '+ str(sp_cutoff[n][1]) )
    n += 1
plt.savefig('%sFiringRateVsCounts/%s.pdf'%(output_dir,"tSNE_firingratevscounts_Normed_behaviours"), bbox_inches="tight", dpi=300)
plt.close()

