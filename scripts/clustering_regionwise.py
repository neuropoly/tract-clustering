# #!/usr/bin/env python
# Apply clustering on processed rat atlas metrics.


import os
import sys
import numpy as np
import logging
from matplotlib.pylab import *
from matplotlib import pyplot as plt
from matplotlib import colors
import nibabel as nib
import shutil
from sklearn.cluster import *
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.image import grid_to_graph
from sklearn.metrics import mutual_info_score

from plotting import plot_clustering
import params


# Initialize logging
logging.basicConfig(level=params.logging_mode)

np.set_printoptions(threshold=np.inf)


def generate_clustering_per_region(region, levels):
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
    xmin, xmax = params.xminmax
    ymin, ymax = params.yminmax
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
    # TODO: use code from slicewise for paxinos
    nii_paxinos = nib.load(params.file_paxinos + '_' + region + ext)
    paxinos3d = np.mean(nii_paxinos.get_fdata(), axis=2)
    # Crop data
    paxinos3d = paxinos3d[xmin:xmax, ymin:ymax, :]
    # clip between 0 and 1.
    # note: we don't want to normalize, otherwise the background (which should be 0) will have a non-zero value.
    paxinos3d = np.clip(paxinos3d, 0, 1)
    # TODO: crop Paxinos

    # Perform clustering
    logging.info("Run clustering...")
    num_clusters = [7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30]  # [5, 6, 7, 8, 9, 10, 11]

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
            ax.imshow(np.transpose(labels[:, :, i]), cmap='Spectral')
            plt.title("iz = {}".format(i), pad=18)
            plt.tight_layout()
        fig.savefig(os.path.join(params.folder, params.folder_output, params.folder_clustering_regionwise,
                                 'clustering_allslices_ncluster{}_{}.png'.format(n_cluster, region)))
        fig.clear()
        plt.close()

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

        plot_clustering(labels3d, paxinos3d, n_cluster, region, path_output)

    logging.info("Done!")


# SCRIPT STARTS HERE
# ======================================================================================================================

ext = '.nii'

# Deal with output folder
path_output = os.path.join(params.folder, params.folder_output, params.folder_clustering_regionwise)
if os.path.exists(path_output):
    shutil.rmtree(path_output)
os.makedirs(path_output)

os.chdir(os.path.join(params.folder, params.folder_output, params.folder_concat_region))

# Load files per region
for region, levels in params.regions.items():
    logging.info('\nProcessing region: {}'.format(region))
    generate_clustering_per_region(region, levels)
