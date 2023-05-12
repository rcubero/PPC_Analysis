# to make things py2 and py3 compatible
from __future__ import print_function, division

# major packages needed
import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
import matplotlib as mpl
import itertools, os, sys

from scipy import io, ndimage, optimize, signal, linalg, spatial
from scipy.stats import spearmanr, pearsonr, rankdata
from scipy.interpolate import interp1d
from sklearn import manifold
from random import shuffle
from collections import Counter

from mpl_toolkits.mplot3d import Axes3D

input_dir = "/Users/rcubero/Google Drive/Tara_7x10mins_recordings_perfromanceonly/"
output_dir = "/Users/rcubero/Dropbox/Tuce_results/"

tara_files = ["Tara_S1_BinOasis__1157_performing__renumbered", "Tara_S1_BinOasis__1208_performing__renumbered",
              "Tara_S2_BinOasis__1707_performing__renumbered", "Tara_S2_BinOasis__1718_performing__renumbered",
              "Tara_S3_BinOasis__1205_performing__renumbered", "Tara_S3_BinOasis__1217_performing__renumbered",
              "Tara_S4_BinOasis__1727_performing__renumbered"]

tara_titles = ["Tara_S1_1157", "Tara_S1_1208",
               "Tara_S2_1707", "Tara_S2_1718",
               "Tara_S3_1205", "Tara_S3_1217", "Tara_S4_1727"]

tara_colors = ["darkgreen", "lightgreen",
               "red", "pink",
               "darkblue", "lightblue", "black"]

randomized_range_mean_std = np.loadtxt("%s%s.rmsr"%(output_dir,"randomized_range_mean_std_longer_linear"))

f_mean = interp1d(randomized_range_mean_std[0], randomized_range_mean_std[1], kind="cubic")
f_std = interp1d(randomized_range_mean_std[0], randomized_range_mean_std[2], kind="cubic")

relevance = {}
N_spikes = {}
filtered_significant_neuron = {}
filtered_significant_neuron_zscore = {}
zscore_cutoff = float(sys.argv[3])
for filenames in tara_files:
    binoasis = pd.read_csv("%s%s.csv"%(input_dir,filenames), header=-1)
    relevance[filenames] = np.loadtxt("%s%s.msr"%(output_dir,filenames))
    N_spikes[filenames] = binoasis.sum(axis=0).values[1:]
    filtered_significant_neuron_zscore[filenames] = []
    filtered_significant_neuron[filenames] = []
    for i in np.arange(len(N_spikes[filenames])):
        if N_spikes[filenames][i] >= 20:
            z_score = (relevance[filenames][i]-f_mean(N_spikes[filenames][i]))/(f_std(N_spikes[filenames][i]))
            filtered_significant_neuron_zscore[filenames].append(z_score)
            if z_score >= zscore_cutoff:
                filtered_significant_neuron[filenames].append(1)
            else:
                filtered_significant_neuron[filenames].append(0)
                relevance[filenames][i] = 0
        else:
            filtered_significant_neuron_zscore[filenames].append(0)
            filtered_significant_neuron[filenames].append(0)
            relevance[filenames][i] = 0

#significant_neurons = np.zeros((len(tara_files),len(relevance[tara_files[0]])))
#for i in np.arange(len(tara_files)):
#    significant_neurons[i] = filtered_significant_neuron[tara_files[i]]
#significant_neuron_index = np.where(np.sum(significant_neurons,axis=0)==7)[0]+1

def rank_neurons(array, ascending=False):
    if not ascending:
        return rankdata(-array)-1
    else:
        return rankdata(array)-1

rank_cutoff = float(sys.argv[4])
#ranking_neurons = np.zeros((len(tara_files),len(relevance[tara_files[0]])))
#for i in np.arange(len(tara_files)):
#    ranking_neurons[i] = rank_neurons(relevance[tara_files[i]],ascending=False)
#average_ranking_neuron = np.sum(ranking_neurons,axis=0)
#average_ranking_neuron_index = np.where(rank_neurons(average_ranking_neuron, ascending=True)<rank_cutoff)[0]+1

# upload smoothened firing rates
firingrate_std = float(sys.argv[1])
binoasis = pd.read_csv("%s%s.csv"%(input_dir,tara_files[0]), header=-1)
task_index = np.ones(len(binoasis))
for j in np.arange(1, binoasis.shape[1], 1):
    binoasis.iloc[:,j] = ndimage.filters.gaussian_filter1d(binoasis.iloc[:,j], firingrate_std)
for i in np.arange(1,len(tara_files)):
    appended_oasis = pd.read_csv("%s%s.csv"%(input_dir,tara_files[i]), header=-1)
    for j in np.arange(1, appended_oasis.shape[1], 1):
        appended_oasis.iloc[:,j] = ndimage.filters.gaussian_filter1d(appended_oasis.iloc[:,j], firingrate_std)
    binoasis = binoasis.append(appended_oasis, ignore_index=True)
    task_index = np.append(task_index, (i+1)*np.ones(len(appended_oasis)))

downsamplelength = float(sys.argv[2])
downsampleindex = np.where(np.mod(np.arange(len(task_index)),downsamplelength)==0)[0]

n_components = 3

def plot_results(Y, title, name):
    fig, ax = plt.subplots(dpi=600)
    fig.set_size_inches(20,20)
    for i in np.arange(len(tara_titles)):
        ax.scatter(Y[task_index[downsampleindex]==(i+1),0], Y[task_index[downsampleindex]==(i+1),1], s=50, color=tara_colors[i], label=tara_titles[i])
    ax.set_title(title)
    ax.legend(loc="upper right")
    plt.savefig('%s_%s.pdf'%(basename,name), bbox_inches="tight", dpi=600)
    plt.close()


def plot3D_results(Y, title, name):
    fig = plt.figure(dpi=600)
    ax = fig.gca(projection='3d')
    fig.set_size_inches(20,20)
    for i in np.arange(len(tara_titles)):
        ax.scatter(Y[task_index[downsampleindex]==(i+1),0], Y[task_index[downsampleindex]==(i+1),1], Y[task_index[downsampleindex]==(i+1),2], s=50, color=tara_colors[i], label=tara_titles[i])
    ax.set_title(title)
    ax.legend(loc="upper right")
    plt.savefig('%s_%s.pdf'%(basename,name), bbox_inches="tight", dpi=600)
    plt.close()


for zscore_cutoff in [3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0]:
    filtered_significant_neuron = {}
    for filenames in tara_files:
        filtered_significant_neuron[filenames] = []
        for i in np.arange(len(N_spikes[filenames])):
            if N_spikes[filenames][i] >= 20:
                z_score = (relevance[filenames][i]-f_mean(N_spikes[filenames][i]))/(f_std(N_spikes[filenames][i]))
                if z_score >= zscore_cutoff:
                    filtered_significant_neuron[filenames].append(1)
                else:
                    filtered_significant_neuron[filenames].append(0)
            else:
                filtered_significant_neuron[filenames].append(0)

    significant_neurons = np.zeros((len(tara_files),len(relevance[tara_files[0]])))
    for i in np.arange(len(tara_files)):
        significant_neurons[i] = filtered_significant_neuron[tara_files[i]]
    significant_neuron_index = np.where(np.sum(significant_neurons,axis=0)==7)[0]+1

    print(zscore_cutoff, len(significant_neuron_index))
    basename = 'MSR_3D_perplexity_DR_FRStd_%04d_DwnSamp_%04d_Zscore_%04d_Rank_%04d'%(firingrate_std, downsamplelength, 10*zscore_cutoff, rank_cutoff)
    tsne = manifold.TSNE(n_components=n_components, perplexity=15, init='pca', random_state=0)
    Y_tsne = tsne.fit_transform(np.array(binoasis.iloc[downsampleindex,significant_neuron_index]))
#    plot_results(Y_tsne, "TSNE: Significant Neuron", "TSNE_significantneuron")
    plot3D_results(Y_tsne, "TSNE: Significant Neuron", "TSNE_significantneuron")
    np.savetxt('%s_%s.d'%(basename,"TSNE_coordinates_significantneuron"),Y_tsne)

# WORK! correlation distances
#tsne = manifold.TSNE(n_components=n_components, perplexity=15, init='pca', random_state=0)
#Y_tsne = tsne.fit_transform(np.array(binoasis.iloc[downsampleindex,significant_neuron_index]))
#plot_results(Y_tsne, "TSNE: Significant Neuron", "TSNE_significantneuron")
#np.savetxt('%s_%s.d'%(basename,"TSNE_coordinates_significantneuron"),Y_tsne)

#for j in np.arange(len(tara_files)):
#    relevance_ranking_neuron_index = np.where(rank_neurons(relevance[tara_files[j]],ascending=False)<rank_cutoff)[0]+1
#    significant_ranking_neuron_index = np.where(filtered_significant_neuron[tara_files[j]])[0]+1
#    Y_tsne = tsne.fit_transform(np.array(binoasis.iloc[downsampleindex,significant_ranking_neuron_index]))
#    plot_results(Y_tsne, 'TSNE: Relevant Neuron (%s)'%(tara_titles[j]), 'TSNE_relevantneuron_%s'%(tara_titles[j]))
#    np.savetxt('%s_%s.d'%(basename,'TSNE_coordinates_significantneuron_%s'%(tara_titles[j])),Y_tsne)

#Y_tsne = tsne.fit_transform(np.array(binoasis.iloc[downsampleindex,average_ranking_neuron_index]))
#plot_results(Y_tsne, "TSNE: Relevant Neuron", "TSNE_relevantneuron")
#np.savetxt('%s_%s.d'%(basename,"TSNE_coordinates_relevantneuron"),Y_tsne)
#
#for j in np.arange(len(tara_files)):
#    relevance_ranking_neuron_index = np.where(rank_neurons(relevance[tara_files[j]],ascending=False)<rank_cutoff)[0]+1
#    Y_tsne = tsne.fit_transform(np.array(binoasis.iloc[downsampleindex,relevance_ranking_neuron_index]))
#    plot_results(Y_tsne, 'TSNE: Relevant Neuron (%s)'%(tara_titles[j]), 'TSNE_relevantneuron_%s'%(tara_titles[j]))
#    np.savetxt('%s_%s.d'%(basename,'TSNE_coordinates_relevantneuron_%s'%(tara_titles[j])),Y_tsne)

#significant_ranking_neuron_index = np.sort([115, 44, 187, 36, 85, 17, 129, 89, 9, 59, 60, 73, 146, 152, 150, 32, 94, 58, 185, 95])
#significant_ranking_neuron_index = np.arange(len(relevance[tara_files[0]]))+1
#Y_tsne = tsne.fit_transform(np.array(binoasis.iloc[downsampleindex,significant_ranking_neuron_index]))
#plot_results(Y_tsne, 'TSNE: All Neuron', 'TSNE_allneuron2')
#np.savetxt('%s_%s.d'%(basename,'TSNE_coordinates_allneuron'),Y_tsne)

