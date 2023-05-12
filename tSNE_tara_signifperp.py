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

input_dir = "/Users/rcubero/Google Drive/Lucy/"
output_dir = "/Users/rcubero/Dropbox/Tuce_results/Lucy_results/"

lucy_files = ["Lucy_1404_S1_BinOasis__1530_performing__renumbered", "Lucy_1404_S1_BinOasis__1622_performing__renumbered",
              "Lucy_1504_S1_BinOasis__1120_performing__renumbered", "Lucy_1504_S1_BinOasis__1140_performing__renumbered",
              "Lucy_1504_S2_BinOasis__1730_performing__renumbered", "Lucy_1504_S2_BinOasis__1745_performing__renumbered",
              "Lucy_1704_S1_BinOasis__1141_performing__renumbered", "Lucy_1704_S1_BinOasis__1152_performing__renumbered"]

lucy_titles = ["Session 1 (R)", "Session 2 (R)",
               "Session 3 (R)", "Session 4 (R)",
               "Session 5 (R)", "Session 6 (R)",
               "Session 7 (OF)", "Session 8 (OF)"]

lucy_colors = ["saddlebrown", "burlywood",
               "darkgreen", "lightgreen",
               "red", "pink",
               "darkblue", "lightblue"]

lucy_colors2 = ["lightgrey", "blue", "red", "lightblue", "darkviolet",
                "lightgreen", "green", "orange", "lightpink", "orchid"]
lucy_labels = ['unclassified', 'Nosepoke', 'Reaching', 'Eating', 'Grooming',
               'Turning (CW)', 'Turning (CCW)', 'Rearing', 'Running (CW)', 'Running (CCW)']

firingrate_std = 5
downsamplelength = 5
zscore_cutoff = 3.5
FR_threshold = 0.1
count_threshold = 2

lucy_behave = ['behavioral_annotation_1530', 'behavioral_annotation_1622',
               'behavioral_annotation_1120', 'behavioral_annotation_1140',
               'behavioral_annotation_1730', 'behavioral_annotation_1745',
               'behavioral_annotation_1141', 'behavioral_annotation_1152']
lucy_tasks = ['Nosepoke', 'Grasping', 'Eating on haunches', 'Grooming', 'Turning (CW)',
              'Turning (CCW)', 'Rearing', 'Running (CW)', 'Running (CCW)',
              'Turning CW', 'Turning CCW', 'Nosepoke / no grabbing', 'Nosepoke / grabbing food', 'Haunches', "All4s", "Eating / sniffing (All 4's)", "Eating / haunches"]

binning_time = 0.05
lucy_behaviours = []
for i in np.arange(len(lucy_behave)):
    expt_times = pd.read_csv("%s%s.csv"%(input_dir,lucy_files[i]), header=-1)[0].values
    timestamps = np.zeros(len(expt_times))
    
    if Path(input_dir+lucy_behave[i]+'.pkl').exists():
        behave_names = pkl.load(open(input_dir+lucy_behave[i]+'_names.pkl','rb'), encoding='latin1')
        behave_times = pkl.load(open(input_dir+lucy_behave[i]+'_times.pkl','rb'), encoding='latin1')
        
        for names in behave_names:
            if names in lucy_tasks:
                task_index = behave_names.index(names)
                
                if names in ["Turning CW"]: names = "Turning (CW)"
                if names in ["Turning CCW"]: names = "Turning (CCW)"
                #                 if names in ["Turning CW", "Turning (CW)", "Turning (CCW)"]: names = "Turning CCW"
                #                 if names == "Running (CW)": names = "Running (CCW)"
                if names in ["Nosepoke", "Nosepoke / no grabbing", "Nosepoke / grabbing food"]: names = "Nosepoke"
                if names in ["Haunches", "All4s", "Eating / sniffing (All 4's)", "Eating / haunches"]: names = "Eating on haunches"
                
                t_start = [behave_times[task_index][t][0] for t in np.arange(len(behave_times[task_index])) if behave_times[task_index][t][1]==1]
                t_stop = [behave_times[task_index][t][0] for t in np.arange(len(behave_times[task_index])) if behave_times[task_index][t][1]==2]
                
                start_index = np.around((np.around(t_start,2)/0.05),0).astype('int')
                stop_index = np.around((np.around(t_stop,2)/0.05),0).astype('int')
                
                # this is to delete some
                if(len(start_index)!=len(stop_index)):
                    zero_index = np.where(np.diff(np.array(behave_times[task_index])[:,1])==0)[0][0]
                    start_index = np.delete(start_index,np.where(np.around(t_start,2)==np.around(behave_times[task_index][zero_index][0],2))[0])

                for behave_time_index in np.arange(len(start_index)):
                    timestamps[start_index[behave_time_index]:stop_index[behave_time_index]] = lucy_tasks.index(names)+1
    lucy_behaviours.append(timestamps)
lucy_behaviours = np.array(lucy_behaviours).flatten()

randomized_range_mean_std = np.loadtxt("%s%s.rmsr"%(output_dir,"randomized_range_mean_std_longer_linear"))
f_mean = interp1d(randomized_range_mean_std[0], randomized_range_mean_std[1], kind="cubic")
f_std = interp1d(randomized_range_mean_std[0], randomized_range_mean_std[2], kind="cubic")

relevance = {}
N_spikes = {}
relevance_zscore = {}
for filenames in lucy_files:
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

significant_neurons = np.zeros((len(lucy_files),len(relevance[lucy_files[0]])))
for i in np.arange(len(lucy_files)):
    significant_neurons[i] = filtered_significant_neuron[lucy_files[i]]
significant_neuron_index = np.where(np.sum(significant_neurons,axis=0)==8)[0]+1


binoasis = pd.read_csv("%s%s.csv"%(input_dir,lucy_files[0]), header=-1)
task_index = np.ones(len(binoasis))
task_names = np.around(binoasis[0].values,2)
task_titles = np.array([lucy_titles[0]]*len(binoasis)).astype('str')
for j in np.arange(1, binoasis.shape[1], 1):
    binoasis.iloc[:,j] = ndimage.filters.gaussian_filter1d(binoasis.iloc[:,j], firingrate_std)
for i in np.arange(1,len(lucy_files)):
    appended_oasis = pd.read_csv("%s%s.csv"%(input_dir,lucy_files[i]), header=-1)
    for j in np.arange(1, appended_oasis.shape[1], 1):
        appended_oasis.iloc[:,j] = ndimage.filters.gaussian_filter1d(appended_oasis.iloc[:,j], firingrate_std)
    binoasis = binoasis.append(appended_oasis, ignore_index=True)
    task_index = np.append(task_index, (i+1)*np.ones(len(appended_oasis)))
    task_names = np.append(task_names, np.around(appended_oasis[0].values,2))
    task_titles = np.append(task_titles, np.array([lucy_titles[i]]*len(appended_oasis)).astype('str'))

downsampleindex = np.where(np.mod(np.arange(len(task_index)),downsamplelength)==0)[0]


FR_threshold = 0.2
count_threshold = 2

time_data = np.array(binoasis.iloc[:, significant_neuron_index])
downsampling = (np.mod(np.arange(len(task_index)),downsamplelength)==0).astype('int')

Sfilt = (time_data > FR_threshold)*1.
Scounts = np.sum(Sfilt, 1)
Skept = (Scounts > count_threshold).astype('int')
kept_index = np.where(downsampling*Skept)[0]


perplexity_vals = [10, 25, 50, 70, 100]
learning_rate = [50, 100, 150, 200]
early_exaggeration = [12, 50, 100, 200, 400]


for perp_val in perplexity_vals:
    for l_rate in learning_rate:
        for e_exag in early_exaggeration:
            basename = 'Lucy_tSNE_Z3p5F5D5FRp2C2Perp%dL%dE%d'%(perp_val, l_rate, e_exag)

            tsne = manifold.TSNE(n_components=2, perplexity=perplexity_value, init='pca', random_state=0)
            Y_tsne = tsne.fit_transform(np.array(binoasis.iloc[downsampleindex, significant_neuron_index]))
            np.savetxt('%stSNE_explore/%s.txt'%(output_dir,basename,"TSNE_coordinates"),Y_tsne)

            fig = plt.figure(dpi=300)
            fig.set_size_inches(13,5)

            xmin, xmax = np.amin(Y_tsne[:,0])-np.abs(np.amin(Y_tsne[:,0]))*0.1, np.amax(Y_tsne[:,0])+np.abs(np.amax(Y_tsne[:,0]))*0.1
            ymin, ymax = np.amin(Y_tsne[:,1])-np.abs(np.amin(Y_tsne[:,1]))*0.1, np.amax(Y_tsne[:,1])+np.abs(np.amax(Y_tsne[:,1]))*0.1

            ax = fig.add_subplot(121)
            for i in np.arange(len(lucy_titles)):
                ax.scatter(Y_tsne[task_index[kept_index]==(i+1),0],
                           Y_tsne[task_index[kept_index]==(i+1),1],
                           s=5, color=lucy_colors[i], label=lucy_titles[i])
            ax.set_title('Session Mapping')
            ax.legend(loc="upper right", fontsize=5)
            ax.set_xlim(left=xmin, right=xmax)
            ax.set_ylim(bottom=ymin, top=ymax)

            ax = fig.add_subplot(122)
            for i in np.delete(np.unique(lucy_behaviours),0).astype('int'):
                ax.scatter(Y_tsne[lucy_behaviours[kept_index]==i,0],
                           Y_tsne[lucy_behaviours[kept_index]==i,1],
                           s=5, color=lucy_colors2[i],
                           label=lucy_labels[i], rasterized=True)
            ax.set_title('Behavioural Mapping')
            ax.legend(loc="upper right", fontsize=5, bbox_to_anchor=(1.13, 1.015))
            ax.set_xlim(left=xmin, right=xmax)
            ax.set_ylim(bottom=ymin, top=ymax)

            plt.savefig('%stSNE_explore/%s.pdf'%(output_dir,basename,"TSNE_coordinates"), bbox_inches="tight", dpi=300)
            plt.close()
