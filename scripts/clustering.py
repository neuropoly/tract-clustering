# #!/usr/bin/env python

# Apply clustering on processed rat atlas metrics.

import os
import seaborn as sns
import pickle
import numpy as np
import scipy as sp
import nibabel as nib
import pandas as pd
import numpy.ma as ma
from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from mpl_toolkits.mplot3d import Axes3D
from PIL import Image
from scipy import misc
from scipy.cluster.hierarchy import dendrogram
from scipy import ndimage
from scipy import signal
from sklearn.datasets import load_iris
from scipy import stats
from collections import Counter
from pandas import DataFrame as df
from sklearn import datasets
from sklearn import metrics
from sklearn.cluster import *
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
from sklearn.feature_extraction.image import grid_to_graph
from sklearn.metrics import silhouette_samples, silhouette_score, pairwise_distances
from sklearn.metrics import davies_bouldin_score

import params


# seaborn fig params
sns.set(font_scale=1.4)
sns.set_style("whitegrid", {'axes.grid' : False})

np.set_printoptions(threshold=np.inf)


def generate_clustering_per_region(region):
    """
    Generate clustering from a series of 2D slices pertaining to a region (e.g. cervical)
    :param region:
    :param levels: list of levels
    :return:
    """
    # open file
    nii = nib.load(params.file_prefix + region + ext)
    data = nii.get_data()

# SCRIPT STARTS HERE
# ======================================================================================================================

ext = '.nii'

os.chdir(params.folder_concat_region)

# Load files per region
for region, levels in params.regions.items():
    generate_clustering_per_region(region)







# nibfile = nib.load("/Users/julien/Desktop/Tracts_testing_2/S4/Volume4D_sym_cleaned.nii.gz")
# data = nibfile.get_data()
mask = nib.load("/Users/hanam/Documents/Tracts_testing_2/all_levels/S4/BW_mask.nii.gz")
maskdata = mask.get_data()

# In[2944]:


img = plt.imshow(maskdata)

# In[2945]:


maskdata = np.rot90(maskdata, 1, (1, 0))

# In[2946]:


plt.imshow(maskdata, cmap="gray")

# In[2947]:


maskdata = maskdata[:, :maskdata.shape[1] // 2]

# In[2948]:


plt.imshow(maskdata, cmap="gray")

# In[2949]:


maskdata.shape

# In[2950]:


data.shape


# In[2951]:


def crop_center(img, cropx, cropy):
    y, x = img.shape
    startx = x // 2 - (cropx // 2)
    starty = y // 2 - (cropy // 2)
    return img[starty:starty + cropy, startx:startx + cropx]


# In[2952]:


feature_names = [
    "number_axons",
    "axon_equiv_diameter",
    "avf",
    "g_ratio",
    "myelin_thickness",
    "mvf",
    ]

# In[2953]:


data = data.transpose((1, 0, 2, 3))

# Ignore solidity, eccent., orient., etc.
# data = data[..., 0:6]
data = data.squeeze(axis=2)

# In[2954]:


data.shape

# In[2955]:


data = data[:, :data.shape[1] // 2, :]

# In[2956]:


data.shape

# In[2957]:


original_shape = data.shape[0:2]

# printoriginal_shape

# In[2958]:


h1 = 35
h2 = 115
w1 = 0
w2 = 180


# In[2959]:


def colorbar(mappable):
    ax = mappable.axes
    fig = ax.figure
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    return fig.colorbar(mappable, cax=cax)


# In[2960]:


fig = plt.figure(figsize=(20, 20))
fig.subplots_adjust(hspace=0.1, wspace=0.1)

for i in range(1, data.shape[2] + 1):
    # image= np.where(maskdata, data[...,i-1], 0)
    data[..., i - 1] = np.where(maskdata, data[..., i - 1], 0)
    im_crop = crop_center(data[..., i - 1], w2 - w1, h2 - h1)
    # print
    # data.shape

    ax = fig.add_subplot(5, 4, i)

    # ndimage.median_filter(data,size=20)

    # signal.medfilt(data, 5)
    # im_crop[i , ...]= im_crop(mask[...])

    img = ax.imshow(im_crop)

    # Center line
    # ax.axvline(original_shape[1] // 2)
    plt.title(feature_names[i - 1])
    colorbar(img)

plt.tight_layout(h_pad=1)

# In[2961]:


connectivity = grid_to_graph(n_x=data.shape[0],
                             n_y=data.shape[1])

# In[2962]:


connectivity

# In[2963]:


data = data.reshape(-1, data.shape[2])

# In[2964]:


scaler = StandardScaler()
data = scaler.fit_transform(data)

# In[2965]:


data.shape

# # t-SNE

# In[2966]:


# data_embedded = TSNE(n_components=2, verbose=3, perplexity=40).fit_transform(data)


# In[2967]:


num_clusters = 7
clustering = AgglomerativeClustering(linkage="ward", n_clusters=num_clusters,
                                     connectivity=connectivity)
clustering.fit(data)
labels = clustering.labels_

# silhouette= silhouette_score(connectivity, labels)


# In[2968]:


# plt.figure(figsize=(12, 8))
# plt.scatter(data_embedded[:, 0], data_embedded[:, 1],
#             cmap="Spectral", s=10, c=clustering.labels_)
# plt.title("t-SNE without spatial information")
# plt.show()


# In[2969]:


labels.shape

# In[2970]:


labels

# In[2971]:


frequent = stats.mode(labels)[0]
frequent = frequent[0]
# print
# frequent

# In[2972]:


silhouette = silhouette_score(data, labels != frequent)
# print
# silhouette

# In[2973]:


labels = labels.reshape(*original_shape)

# In[2974]:


labels.shape

# In[2975]:


labels = labels.astype(float)

# In[2976]:


labels[labels == frequent] = np.nan

# In[2977]:


labels

# In[2978]:


plt.imshow(labels, cmap="Spectral")
plt.title("Agglomerative Clustering with Connectivity (K={})".format(num_clusters), pad=18)
plt.tight_layout()

# In[2979]:


## Plot Dendogram

# This example plots the corresponding dendrogram of a hierarchical clustering
# using AgglomerativeClustering and the dendrogram method available in scipy.


# def plot_dendrogram(model, **kwargs):

#     # Children of hierarchical clustering
#     children = model.children_

#     # Distances between each pair of children
#     # Since we don't have this information, we can use a uniform one for plotting
#     distance = np.arange(children.shape[0])

#     # The number of observations contained in each cluster level
#     no_of_observations = np.arange(2, children.shape[0]+2)

#     # Create linkage matrix and then plot the dendrogram
#     linkage_matrix = np.column_stack([children, distance, no_of_observations]).astype(float)

#     # Plot the corresponding dendrogram
#     dendrogram(linkage_matrix, **kwargs)


# iris = load_iris()
# x = iris.data[:20]
# model = AgglomerativeClustering(n_clusters=3)

# model = model.fit(x)
# plt.title('Hierarchical Clustering Dendrogram')
# plot_dendrogram(model, labels=model.labels_)
# plt.show()


# In[2980]:


x = labels != frequent

# In[2981]:


x

# In[2982]:


fig = plt.figure(figsize=(20, 20))
fig.subplots_adjust(hspace=0.1, wspace=0.1)

for i in range(1, 26):
    clustering = AgglomerativeClustering(linkage="ward", n_clusters=i + 4,
                                         connectivity=connectivity)
    clustering.fit(data)
    labels = clustering.labels_
    frequent = stats.mode(labels)[0]
    frequent = frequent[0]

    # Find the foreground rows
    frgrd_idx = [idx for idx, l in enumerate(labels) if l != frequent]
    # Extract the foreground rows from data
    frgrd_data = data[frgrd_idx, :]
    # Extract the foreground rows from labels
    frgrd_labels = labels[frgrd_idx]

    # Compute the silhouette_score on the foreground voxels
    #     silhouette= silhouette_score(frgrd_data,labels=frgrd_labels)

    #     calinski= metrics.calinski_harabasz_score(data, labels)
    #     print("calinski-harabasz score=", calinski)

    #     davies= davies_bouldin_score(frgrd_data,labels=frgrd_labels)

    #     scores = pd.DataFrame(silhouette, columns= 'Silhouette')
    #     scores = df.reindex(columns=list('Silhouette Davies'), fill_value= davies)

    # sub_image = full_image[y_start: y_end, x_start:x_end]

    #     print silhouette, davies

    labels = labels.reshape(*original_shape)
    labels = labels.astype(float)
    labels[labels == frequent] = np.nan
    im_crop = crop_center(labels, w2 - w1, h2 - h1)

    ax = fig.add_subplot(5, 5, i)
    img = ax.imshow(im_crop, cmap="Spectral")
    plt.title("Clusters = {}".format(i + 4), pad=18)
    plt.tight_layout()

plt.suptitle("Agglomerative Clustering with Connectivity, different cluster numbers from 5 to 29")
plt.tight_layout(h_pad=1)

