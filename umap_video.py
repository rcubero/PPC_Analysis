# to make things py2 and py3 compatible
from __future__ import print_function, division

import numpy as np
import os, sys
from scipy import io
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure


video_dir = sys.argv[1]
results = io.loadmat("%s/%s.mat"%(video_dir, sys.argv[2]))
seven_labels = results['seven_labels'].flatten()
task_index = results['task_index'].flatten()
task_timestamp = results['task_timestamp'].flatten()
task_titles = results['task_titles'].flatten()
kept_index = results['kept_index'].flatten()
downsampleindex = results['downsampleindex'].flatten()
closest_neighbor_kept = results['closest_neighbor_kept'].flatten()

results = io.loadmat("%s/%s.mat"%(video_dir, sys.argv[3]))
Y_umap = results['coordinates']
seven_behaviours = results['behave_labels'][0]
kept_index = results['kept_index'][0]

video_title = sys.argv[4]
output_dir = sys.argv[5]

video_indices = np.array([inds for inds in np.where(task_titles==video_title)[0] if inds in downsampleindex])
video_unlabeled = np.array([inds for inds in kept_index if seven_behaviours[inds]==0])
video_startind = np.where(task_titles==video_title)[0][0]
umap_ind = 0

fig = plt.figure()
fig.set_size_inches(12,5)
fig.subplots_adjust(hspace=0.01)

ax1 = fig.add_subplot(121)

xmin, xmax = np.amin(Y_umap[:,0])-np.abs(np.amin(Y_umap[:,0]))*0.1, np.amax(Y_umap[:,0])+np.abs(np.amax(Y_umap[:,0]))*0.1
ymin, ymax = np.amin(Y_umap[:,1])-np.abs(np.amin(Y_umap[:,1]))*0.1, np.amax(Y_umap[:,1])+np.abs(np.amax(Y_umap[:,1]))*0.1

ax1.scatter(Y_umap[:,0], Y_umap[:,1], s=8, color='lightgrey', rasterized=True, alpha=0.1)

ax1.set_xlim(left=xmin, right=xmax)
ax1.set_ylim(bottom=ymin, top=ymax)
ax1.tick_params(axis='both', left='off', top='off', right='off', bottom='off', labelleft='off', labeltop='off', labelright='off', labelbottom='off')
ax1.axis('off')

for t_index in video_indices:
    if t_index in kept_index:
        umap_ind = np.where(kept_index==t_index)[0][0]
        ax1_scatter = ax1.scatter(Y_umap[umap_ind,0], Y_umap[umap_ind,1], s=50, color='blue')
    #else:
    #    umap_ind = closest_neighbor_kept[t_index]
    #    ax1_scatter = ax1.scatter(Y_umap[umap_ind,0], Y_umap[umap_ind,1], s=50, color='lightblue')

    ax3 = fig.add_subplot(122)
    gray = plt.imread('%s/%s/%s_%06d.png'%(video_dir, video_title, video_title, t_index-video_startind+1))
    ax3.imshow(gray, cmap='binary_r')
    ax3.text(10,len(gray)-10, str(task_timestamp[t_index])+" s", fontsize=15, color='k', horizontalalignment='left')
    ax3.axis('off')
    ax3.tick_params(axis='both', left='off', top='off', right='off', bottom='off', labelleft='off', labeltop='off', labelright='off', labelbottom='off')
    
    canvas = FigureCanvas(fig)
    canvas.print_figure('%s/%s_%06d'%(output_dir, video_title, t_index)+'.png', bbox_inches="tight", dpi=300)
    if t_index in kept_index:
        ax1_scatter.remove()
        del ax1_scatter
    ax3.cla()

plt.close()
