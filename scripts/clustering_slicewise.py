# #!/usr/bin/env python

# Perform clustering slicewise and generate figure comparing clustering across slices with Paxinos

# TODO: create folder for outputs: clustering_slicewise

import os
import sys
import seaborn as sns
import numpy as np
import logging
from matplotlib.pylab import *
from matplotlib import pyplot as plt
from matplotlib import colors
import nibabel as nib
from sklearn.cluster import *
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.image import grid_to_graph
from sklearn.metrics import mutual_info_score

from utils import get_best_matching_color_with_paxinos
import params


# Initialize logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)  # default: logging.DEBUG, logging.INFO
hdlr = logging.StreamHandler(sys.stdout)
logging.root.addHandler(hdlr)

# seaborn fig params
sns.set(font_scale=1.4)
sns.set_style("whitegrid", {'axes.grid': False})
np.set_printoptions(threshold=np.inf)

ext = '.nii'

os.chdir(os.path.join(params.FOLDER, params.OUTPUT_FOLDER))

# Define levels from params.
# print (params.regions['cervical'])
levels = []
for region in params.regions.keys():
    levels = levels + params.regions[region]

# levels = ['C1','C2','C3']

# Loop across spinal levels
for level in levels:
    # Load data
    # This data has the following content for the 4th dimension:
    # TODO: complete below
    # 0: XX
    # 1: XX
    # 5: WM mask
    # 6: Paxinos tract 1
    # 7: Paxinos tract 2
    # ..
    # 13: Paxinos tract 8
    logger.info("\nLoad data for level: " + level)
    nii = nib.load(params.file_prefix_all + level + ext)

    data = nii.get_fdata()
    # print ('Data size: ' + str(data.shape))

    # Crop around spinal cord, and only keep half of it.
    # The way the atlas was built, the right and left sides are perfectly symmetrical (mathematical average). Hence,
    # we can discard one half, without loosing information.
    xmin, xmax = (30, 75)
    ymin, ymax = (40, 105)
    data_crop = data[xmin:xmax, ymin:ymax, :]
    # print ('Data size: ' + str(data_crop.shape))

    # DEBUG: print fig
    # from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
    # from matplotlib.figure import Figure
    # fig = Figure()
    # FigureCanvas(fig)
    # ax = fig.add_subplot(111)
    # ax.matshow(data_crop[:, :, 0, 5], cmap='gray')
    # fig.savefig('fig_data_crop5.png')

    # Reshape to 1d
    # Note: 3rd dimension is a singleton (because single slice)
    # Note: mask of WM corresponds to the 5th vector along the 4th dimension
    mask_crop = data_crop[:, :, 0, 5]
    mask_crop = mask_crop.astype(bool)
    ind_mask = np.where(mask_crop)
    mask1d = np.squeeze(mask_crop.reshape(-1, 1))

    # Load Paxinos atlas
    # TODO: no need to load, we can use data
    nii_paxinos = nib.load(os.path.join(params.FOLDER, params.file_paxinos + '.nii.gz'))
    paxinos3d = nii_paxinos.get_fdata()
    # Crop Paxinos
    paxinos3d = paxinos3d[xmin:xmax, ymin:ymax, :]
    # print ('Paxinos size: ' + str(paxinos3d.shape))
    paxinos3d = np.clip(paxinos3d, 0, 1)
    paxinos2d_complete = np.zeros((paxinos3d.shape[0], paxinos3d.shape[1]))
    for tract in range (0, paxinos3d.shape[3]):
        paxinos2d = (paxinos3d[:, :, levels.index(level), tract])
        paxinos2d[paxinos2d > 0] = tract + 1
        # fig = plt.figure(figsize=(20, 20))
        # plt.imshow(paxinos2d, cmap='Spectral')
        # plt.title('Paxinos: ' + level + str(tract + 1))
        # plt.tight_layout()
        # fig.savefig('paxinos_level-{}_tract-{}.png'.format(level,tract + 1))
        # fig.clear()
        # plt.close()
        # paxinos2d_complete = paxinos2d_complete + paxinos2d
        paxinos2d_complete = np.where(paxinos2d_complete != 0, paxinos2d_complete, paxinos2d)

    fig = plt.figure(figsize=(20, 20))
    plt.imshow(paxinos2d_complete, cmap='Spectral')
    plt.title('Paxinos complete ' + level)
    plt.tight_layout()
    fig.savefig('paxinos_complete_level-{}.png'.format(level))
    fig.clear()
    plt.close()

    # Reshape data used for clustering
    # Here, we will perform clustering on the first 5 images (ie: selection on the 4th dimension)
    data2d = data_crop[:, :, 0, 0:5].reshape(-1, 5)

    # Standardize data
    logger.info("Standardize data...")
    scaler = StandardScaler()
    data2d_norm = scaler.fit_transform(data2d)
    del data2d

    # Build connectivity matrix
    logger.info("Build connectivity matrix...")
    connectivity = grid_to_graph(n_x=data_crop.shape[0],
                                 n_y=data_crop.shape[1],
                                 n_z=data_crop.shape[2],
                                 mask=mask_crop)
    del data_crop

    # Perform clustering
    logger.info("Run clustering...")
    num_clusters = [8, 10]  # [5, 6, 7, 8, 9, 10, 11]
    for n_cluster in num_clusters:
        logger.info("Number of clusters: {}".format(n_cluster))
        clustering = AgglomerativeClustering(linkage="ward", n_clusters=n_cluster, connectivity=connectivity)
        clustering.fit(data2d_norm[mask1d, :])
        logger.info("Reshape labels...")
        labels = np.zeros_like(mask_crop, dtype=np.int)
        labels[ind_mask] = clustering.labels_ + 1  # we add a the +1 because sklearn's first label has value "0", and we are now going to use "0" as the background (i.e. not a label)
        del clustering

        logger.info("Generate figure...")
        fig = plt.figure(figsize=(10, 5))
        # Display Paxinos
        # fig.tight_layout()
        ax = fig.add_subplot(1, 2, 2)
        im = ax.imshow(rot90(paxinos2d_complete), cmap='Spectral')
        ax.set_title('Paxinos complete ' + level)

        # Find label color corresponding best to the Paxinos atlas
        list_color = get_best_matching_color_with_paxinos(im=labels, imref=paxinos2d_complete)

        # cb_ax = fig.add_axes([0.83, 0.1, 0.02, 0.8])
        # fig.colorbar(im, cax=cb_ax)

        # Display clustering results
        ax1 = fig.add_subplot(1, 2, 1)
        ax1.imshow(rot90(labels[:, :]), cmap='Spectral')
        ax1.set_title('Clustering_results_ncluster{}_{}'.format(n_cluster, level))
        # fig.title(level)

        fig.savefig('clustering_results_ncluster{}_{}.png'.format(n_cluster, level))
        fig.clear()
        plt.close()

        # TODO: flip paxinos R-L, remove numbers, bring the two figs closer together

        # TODO: adjust colormap of clustering to match paxinos

        # del data2d_norm
        logger.info("Done!")

    # Cluster across metrics (dim 0 -> 4)
    # --> generates 2d*n (n = numb of clusters)

    # save in variable

# display figure (matrix of slices) showing clusters on the right, and paxinos on the left, for each slice.

# # TODO: __main__ to make it callable via CLI