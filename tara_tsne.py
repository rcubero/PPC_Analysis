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

tara_files = ["Tara_S1_BinOasis__1157_performing__renumbered", "Tara_S1_BinOasis__1208_performing__renumbered",
              "Tara_S3_BinOasis__1205_performing__renumbered", "Tara_S3_BinOasis__1217_performing__renumbered",
              "Tara_S4_BinOasis__1727_performing__renumbered"]

tara_oasis = ["Tara_S1_Oasis__1157_performing__renumbered", "Tara_S1_Oasis__1208_performing__renumbered",
              "Tara_S3_Oasis__1205_performing__renumbered", "Tara_S3_Oasis__1217_performing__renumbered",
              "Tara_S4_Oasis__1727_performing__renumbered"]

tara_titles = ["Tara_S1_1157", "Tara_S1_1208",
               "Tara_S3_1205", "Tara_S3_1217",
               "Tara_S4_1727"]

tara_colors = ["darkgreen", "lightgreen",
               "darkblue", "lightblue",
               "black"]

tara_behave = ['behavioral_annotation_1157', 'behavioral_annotation_1208',
               'behavioral_annotation_1205', 'behavioral_annotation_1217',
               'behavioral_annotation_1727']
tara_tasks = ["Nosepoke (no food)", "Nosepoke/grasping to eat", "Eating on haunches", "Eating all 4's",
              "Grooming", "Turning (CW)", "Turning (CCW)", "Rearing", "Reaching", "Running (CW)", "Running (CCW)",
              "Pitch up", "Nosepoke", "Grasping", "Turning CW", "Turning CCW"]

binning_time = 0.05
tara_behaviours = []
for i in np.arange(len(tara_behave)):
    expt_times = pd.read_csv("%s%s.csv"%(input_dir,tara_files[i]), header=-1)[0].values
    timestamps = np.zeros(len(expt_times))
    
    behave_names = pkl.load(open(input_dir+tara_behave[i]+'_names.pkl','rb'), encoding='latin1')
    behave_times = pkl.load(open(input_dir+tara_behave[i]+'_times.pkl','rb'), encoding='latin1')
    print(tara_titles[i], behave_names)
    
    for names in behave_names:
        if names in tara_tasks:
            task_index = behave_names.index(names)
            
            # these are just to rename the behaviours
            if names in ["Turning CW", "Turning CCW", "Turning (CW)"]: names = "Turning (CCW)"
            if names in ["Running (CW)"]: names = "Running (CCW)"
            if names in ["Nosepoke (no food)", "Nosepoke"]: names = "Nosepoke"
            if names in ["Nosepoke/grasping to eat", "Grasping", "Reaching"]: names = "Reaching"
            if names in ["Eating on haunches", "Eating all 4's"]: names = "Eating on haunches"
            
            t_start = [behave_times[task_index][t][0] for t in np.arange(len(behave_times[task_index])) if behave_times[task_index][t][1]==1]
            t_stop = [behave_times[task_index][t][0] for t in np.arange(len(behave_times[task_index])) if behave_times[task_index][t][1]==2]
            
            start_index = np.around((np.around(t_start,2)/0.05),0).astype('int')
            stop_index = np.around((np.around(t_stop,2)/0.05),0).astype('int')
            
            if(len(start_index)!=len(stop_index)):
                zero_index = np.where(np.diff(np.array(behave_times[task_index])[:,1])==0)[0][0]
                start_index = np.delete(start_index,np.where(np.around(t_start,2)==np.around(behave_times[task_index][zero_index][0],2))[0])
            
            for behave_time_index in np.arange(len(start_index)):
                timestamps[start_index[behave_time_index]:stop_index[behave_time_index]] = tara_tasks.index(names)+1
                # this is to deactivate the turning behaviours in Tara's session
                if ((i in [2,3]) and (names == "Turning (CCW)")):
                    timestamps[start_index[behave_time_index]:stop_index[behave_time_index]] = 0

tara_behaviours.append(timestamps)

tara_behaviours = np.array(tara_behaviours).flatten()

tara_colors2 = ["lightgrey", "blue", "red", "lightblue", "lightblue",
                "darkviolet", "green", "green", "orange", "red",
                "lightpink", "lightpink", "sienna", "blue", "red"]
tara_labels = ["unclassified", "Nosepoke", "Reaching", "Eating", "Eating",
               "Grooming", "Turning", "Turning", "Rearing", "Reaching",
               "Running", "Running", "Pitch up", "Nosepoke", "Reaching"]

zscore_cutoff = 5.0

randomized_range_mean_std = np.loadtxt("%s%s.rmsr"%(output_dir,"randomized_range_mean_std_longer_linear"))
f_mean = interp1d(randomized_range_mean_std[0], randomized_range_mean_std[1], kind="cubic")
f_std = interp1d(randomized_range_mean_std[0], randomized_range_mean_std[2], kind="cubic")

relevance = {}
N_spikes = {}
filtered_significant_neuron = {}
filtered_significant_neuron_zscore = {}
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

significant_neurons = np.zeros((len(tara_files),len(relevance[tara_files[0]])))
for i in np.arange(len(tara_files)):
    significant_neurons[i] = filtered_significant_neuron[tara_files[i]]
significant_neuron_index = np.where(np.sum(significant_neurons,axis=0)==5)[0]+1

firingrate_std = 2
downsamplelength = 1

# upload smoothened firing rates
binoasis = pd.read_csv("%s%s.csv"%(input_dir,tara_oasis[0]), header=-1)
task_index = np.ones(len(binoasis))
task_names = np.around(binoasis[0].values,2)
task_titles = np.array([seven_titles[0]]*len(binoasis)).astype('str')
for j in np.arange(1, binoasis.shape[1], 1):
    binoasis.iloc[:,j] = ndimage.filters.gaussian_filter1d(binoasis.iloc[:,j], firingrate_std)
for i in np.arange(1,len(seven_files)):
    appended_oasis = pd.read_csv("%s%s.csv"%(input_dir,tara_oasis[i]), header=-1)
    for j in np.arange(1, appended_oasis.shape[1], 1):
        appended_oasis.iloc[:,j] = ndimage.filters.gaussian_filter1d(appended_oasis.iloc[:,j], firingrate_std)
    binoasis = binoasis.append(appended_oasis, ignore_index=True)
    task_index = np.append(task_index, (i+1)*np.ones(len(appended_oasis)))
    task_names = np.append(task_names, np.around(appended_oasis[0].values,2))
    task_titles = np.append(task_titles, np.array([seven_titles[i]]*len(appended_oasis)).astype('str'))

downsampleindex = np.where(np.mod(np.arange(len(task_index)),downsamplelength)==0)[0]

FR_threshold = 0.1
count_threshold = 2

time_data = np.array(binoasis.iloc[:, significant_neuron_index])
downsampling = (np.mod(np.arange(len(task_index)),downsamplelength)==0).astype('int')

Sfilt = (time_data > FR_threshold)*1.
Scounts = np.sum(Sfilt, 1)
Skept = (Scounts > count_threshold).astype('int')
kept_index = np.where(downsampling*Skept)[0]

tsne = manifold.TSNE(n_components=2, perplexity=50, init='pca',
                     random_state=0, metric='euclidean')
Y_tsne = tsne.fit_transform(np.array(binoasis.iloc[kept_index, significant_neuron_index]))

xmin, xmax = np.amin(Y_tsne[:,0])-np.abs(np.amin(Y_tsne[:,0]))*0.1, np.amax(Y_tsne[:,0])+np.abs(np.amax(Y_tsne[:,0]))*0.1
ymin, ymax = np.amin(Y_tsne[:,1])-np.abs(np.amin(Y_tsne[:,1]))*0.1, np.amax(Y_tsne[:,1])+np.abs(np.amax(Y_tsne[:,1]))*0.1

X, Y = np.mgrid[xmin:xmax:101j, ymin:ymax:101j]
positions = np.vstack([X.ravel(), Y.ravel()])
kernel = stats.gaussian_kde(Y_tsne.T, bw_method='silverman')
Z_session = np.reshape(kernel(positions).T, X.shape)

X, Y = np.mgrid[xmin:xmax:101j, ymin:ymax:101j]
positions = np.vstack([X.ravel(), Y.ravel()])
kernel = stats.gaussian_kde(Y_tsne[seven_behaviours[kept_index]!=0].T, bw_method='silverman')
Z_behaviour = np.reshape(kernel(positions).T, X.shape)


fig = plt.Figure(dpi=300)
fig.set_size_inches(15,5)

ax1 = fig.add_subplot(131)
ax1.imshow(np.power(np.rot90(Z_session),2), cmap=plt.cm.viridis, extent=[xmin, xmax, ymin, ymax])
ax1.set_title('Session Mapping')
ax1.set_xlim(left=xmin, right=xmax)
ax1.set_ylim(bottom=ymin, top=ymax)

ax2 = fig.add_subplot(132)
ax2.imshow(np.power(np.rot90(Z_behaviour),2), cmap=plt.cm.viridis, extent=[xmin, xmax, ymin, ymax])
ax2.set_title('Behavioural Mapping')
ax2.set_xlim(left=xmin, right=xmax)
ax2.set_ylim(bottom=ymin, top=ymax)

for t_index in np.arange(len(Y_tsne)):
    ax1_scatter = ax1.scatter(Y_tsne[t_index,0], Y_tsne[t_index,1], s=10, color='r', marker='o')
    ax2_scatter = ax2.scatter(Y_tsne[t_index,0], Y_tsne[t_index,1], s=10, color='r', marker='o')
    
    video_title = task_titles[kept_index][t_index]
    timepoint = task_names[kept_index][t_index]
    video_name = "/Users/rcubero/Downloads/Seven_behavioral_videos/"+video_title+".avi"
    
    cap = cv2.VideoCapture(video_name)
    
    frame_number = int(timepoint/0.05)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number-1)
    res, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    ax3 = fig.add_subplot(133)
    ax3.imshow(gray, cmap='binary_r')
    ax3.set_title(video_title+" "+str(timepoint)+" s", fontsize=10)
    ax3.axis('off')
    ax3.tick_params(axis='both', left='off', top='off', right='off', bottom='off', labelleft='off', labeltop='off', labelright='off', labelbottom='off')
    
    if (seven_behaviours[kept_index][t_index]!=0):
        ax3.set_ylabel(seven_labels[int(seven_behaviours[kept_index][t_index])], fontsize=10)
    
    canvas = FigureCanvas(fig)
    canvas.print_figure(output_dir+'tSNE_forvideo/img%08d'%t_index+'.png', bbox_inches="tight", dpi=100)
    ax1_scatter.remove()
    del ax1_scatter
    ax2_scatter.remove()
    del ax2_scatter
    ax3.cla()


