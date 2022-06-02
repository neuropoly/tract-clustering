# #!/usr/bin/env python

# Apply clustering on processed rat atlas metrics.

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
logging.basicConfig(level=params.logging_mode)

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
    use_mask = True

    # Load data
    logging.info("Load data...")
    nii = nib.load(params.file_prefix_all + region + ext)
    data = nii.get_fdata()

    # Crop around spinal cord, and only keep half of it.
    # The way the atlas was built, the right and left sides are perfectly symmetrical (mathematical average). Hence,
    # we can discard one half, without loosing information.
    # TODO: parametrize this, and find center automatically
    # TODO: find cropping values per region
    if region == 'cervical' or region == 'lumbar':
        xmin, xmax = (45, 110)
        ymin, ymax = (75, 114)
    elif region == 'thoracic':
        xmin, xmax = (53, 100)
        ymin, ymax = (75, 105)
    else:
        xmin, xmax = (55, 94)
        ymin, ymax = (75, 95)
    data_crop = data[xmin:xmax, ymin:ymax, :]
    del data

    # If we have a mask of the white matter, we load it and crop it according to the data_crop shape.
    if use_mask:
        # Load data
        nii_mask = nib.load(params.file_mask_prefix + region + ext)
        mask = nii_mask.get_data()
        # Crop, binarize
        mask_crop = mask[xmin:xmax, ymin:ymax, :] > 0.5
    else:
        mask_crop = np.ones(data_crop.shape[0:3]) > 0.5
    # Reshape
    ind_mask = np.where(mask_crop)
    mask1d = np.squeeze(mask_crop.reshape(-1, 1))

    # Standardize data
    logging.info("Standardize data...")
    # original_shape = data_crop.shape[0:3]
    data2d = data_crop.reshape(-1, data_crop.shape[3])
    scaler = StandardScaler()
    data2d_norm = scaler.fit_transform(data2d)
    del data2d

    # Build connectivity matrix
    logging.info("Build connectivity matrix...")
    connectivity = grid_to_graph(n_x=data_crop.shape[0],
                                 n_y=data_crop.shape[1],
                                 n_z=data_crop.shape[2],
                                 mask=mask_crop)

    del data_crop

    # Process Paxinos atlas for display
    nii_paxinos = nib.load(params.file_paxinos + '_' + region + ext)
    paxinos3d = np.mean(nii_paxinos.get_data(), axis=2)
    # Crop data
    paxinos3d = paxinos3d[xmin:xmax, ymin:ymax, :]
    # clip between 0 and 1.
    # note: we don't want to normalize, otherwise the background (which should be 0) will have a non-zero value.
    paxinos3d = np.clip(paxinos3d, 0, 1)
    # TODO: crop Paxinos

    # Perform clustering
    logging.info("Run clustering...")
    num_clusters = [8, 10]  # [5, 6, 7, 8, 9, 10, 11]

    for n_cluster in num_clusters:
        logging.info("Number of clusters: {}".format(n_cluster))
        clustering = AgglomerativeClustering(linkage="ward", n_clusters=n_cluster, connectivity=connectivity)
        clustering.fit(data2d_norm[mask1d, :])
        logging.info("Reshape labels...")
        labels = np.zeros_like(mask_crop, dtype=np.int)
        labels[ind_mask] = clustering.labels_ + 1  # we add a the +1 because sklearn's first label has value "0", and we are now going to use "0" as the background (i.e. not a label)
        del clustering

        # Display clustering results
        logging.info("Generate figures...")
        fig = plt.figure(figsize=(20, 20))
        fig.subplots_adjust(hspace=0.1, wspace=0.1)
        for i in range(len(levels)):
            ax = fig.add_subplot(4, 4, i+1)
            ax.imshow(labels[:, :, i], cmap='Spectral')
            plt.title("iz = {}".format(i), pad=18)
            plt.tight_layout()
        fig.savefig('clustering_results_ncluster{}_{}.png'.format(n_cluster, region))
        fig.clear()

        # Create 4D array: last dimension corresponds to the cluster number. Cluster value is converted to 1.
        a = list(labels.shape)
        a.append(n_cluster)
        labels4d = np.zeros(a)
        for i_label in range(n_cluster):
            ind_label = np.argwhere(labels == i_label + 1)
            for i in ind_label:
                labels4d[i[0], i[1], i[2], i_label] = 1

        # Average across Z. Each cluster is coded between 0 and 1.
        labels3d = np.mean(labels4d, axis=2)

        # Display result of averaging
        logging.info("Generate figures...")
        fig = plt.figure(figsize=(7, 7))
        fig.suptitle('Averaged clusters (N={}) | Region: {}'.format(n_cluster, region), fontsize=20)

        # Display Paxinos
        # TODO: generalize BASE_COLORS for more than 8 labels
        ax = fig.add_subplot(1, 2, 1)
        ax.set_facecolor((1, 1, 1))
        for i_label in range(paxinos3d.shape[2]):
            labels_rgb = np.zeros([paxinos3d.shape[0], paxinos3d.shape[1], 4])
            for ix in range(paxinos3d.shape[0]):
                for iy in range(paxinos3d.shape[1]):
                    ind_color = list(params.colors.keys())[i_label]
                    labels_rgb[ix, iy] = colors.to_rgba(params.colors[ind_color], paxinos3d[ix, iy, i_label])
            ax.imshow(labels_rgb)
        plt.axis('off')
        plt.title("Paxinos atlas", pad=18)
        plt.tight_layout()

        # Find label color corresponding best to the Paxinos atlas
        list_color = get_best_matching_color_with_paxinos(im=labels3d, imref=paxinos3d)

        # Display clustering
        ax = fig.add_subplot(1, 2, 2)
        for i_label in range(n_cluster):
            labels_rgb = np.zeros([labels3d.shape[0], labels3d.shape[1], 4])
            for ix in range(labels3d.shape[0]):
                for iy in range(labels3d.shape[1]):
                    labels_rgb[ix, iy] = colors.to_rgba(params.colors[list_color[i_label]], labels3d[ix, iy, i_label])
            ax.imshow(labels_rgb)
        plt.axis('off')
        plt.title("Cluster map", pad=18)
        plt.tight_layout()
        fig.subplots_adjust(hspace=0, wspace=0.1)
        fig.savefig('clustering_results_avgz_{}_ncluster{}.png'.format(region, n_cluster))

    del data2d_norm

    logging.info("Done!")


# SCRIPT STARTS HERE
# ======================================================================================================================

ext = '.nii'

os.chdir(os.path.join(params.folder, params.folder_output, params.folder_concat_region))

# Load files per region
for region, levels in params.regions.items():
    logging.info('\nProcessing region: {}'.format(region))
    generate_clustering_per_region(region)
