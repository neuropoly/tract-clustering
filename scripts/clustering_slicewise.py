# #!/usr/bin/env python
# Perform clustering slicewise and generate figure comparing clustering across slices with Watson atlas

import logging
import os
from matplotlib.pylab import *
from matplotlib import pyplot as plt
from matplotlib import colors
import nibabel as nib
from sklearn.cluster import *
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.image import grid_to_graph
import shutil

from plotting import plot_clustering
from utils import get_best_matching_color_with_paxinos
import params


# Initialize logging
logging.basicConfig(level=params.logging_mode)

np.set_printoptions(threshold=np.inf)

ext = '.nii'

# Deal with output folder
path_output_folder_results_clustering = os.path.join(params.folder, params.folder_output, params.folder_clustering_slicewise)
if os.path.exists(path_output_folder_results_clustering):
    shutil.rmtree(path_output_folder_results_clustering)
os.makedirs(path_output_folder_results_clustering)
os.chdir(os.path.join(params.folder, params.folder_output, params.folder_processing))

# Define spinal levels from params.
levels = []
for region in params.regions.keys():
    levels = levels + params.regions[region]

# Loop across spinal levels
for level in levels:
    # Load data
    # This data has the following content for the 4th dimension:
    # 0: Axon Density
    # 1: Axon Equivalent diameter
    # 2: Axon volume Fraction
    # 3: G-Ratio
    # 4: Myelin Thickness
    # 5: WM mask
    # 6: Paxinos tract 1
    # 7: Paxinos tract 2
    # ..
    # 12: Paxinos tract 7
    ind_metrics = [0, 2]
    logging.info("\nLoad data for level: " + level)
    nii = nib.load(params.file_prefix_all + level + ext)

    data = nii.get_fdata()

    # Crop around spinal cord, and only keep half of it.
    # The way the atlas was built, the right and left sides are perfectly symmetrical (mathematical average). Hence,
    # we can discard one half, without loosing information.
    xmin, xmax = params.xminmax
    ymin, ymax = params.yminmax
    data_crop = data[xmin:xmax, ymin:ymax, :]

    # Reshape to 1d
    # Note: 3rd dimension is a singleton (because single slice)
    # Note: mask of WM corresponds to the 5th vector along the 4th dimension
    mask_crop = data_crop[:, :, 0, 5]
    mask_crop = mask_crop.astype(bool)
    ind_mask = np.where(mask_crop)
    mask1d = np.squeeze(mask_crop.reshape(-1, 1))

    # Load Paxinos atlas
    paxinos3d = np.squeeze(data_crop[:, :, 0, 6:13])
    # clip between 0 and 1.
    # note: we don't want to normalize, otherwise the background (which should be 0) will have a non-zero value.
    paxinos3d = np.clip(paxinos3d, 0, 1)

    # Reshape data used for clustering
    # Here, we will perform clustering on the first 5 volumes (ie: selection on the 4th dimension)

    data2d = data_crop[:, :, 0, ind_metrics].reshape(-1, len(ind_metrics))

    # Standardize data
    logging.info("Standardize data...")
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

    # Perform clustering
    logging.info("Run clustering...")
    num_clusters = [5, 6, 7, 8, 9, 10, 11]
    for n_cluster in num_clusters:
        logging.info("Number of clusters: {}".format(n_cluster))
        clustering = AgglomerativeClustering(linkage="ward", n_clusters=n_cluster, connectivity=connectivity)
        clustering.fit(data2d_norm[mask1d, :])
        logging.info("Reshape labels...")
        labels = np.zeros_like(mask_crop, dtype=np.int)
        labels[ind_mask] = clustering.labels_ + 1  # we add a the +1 because sklearn's first label has value "0", and we are now going to use "0" as the background (i.e. not a label)
        del clustering

        # Create 3D array: last dimension corresponds to the cluster number. Cluster value is converted to 1.
        a = list(labels.shape)
        a.append(n_cluster)
        labels3d = np.zeros(a)
        for i_label in range(n_cluster):
            ind_label = np.argwhere(labels == i_label + 1)
            for i in ind_label:
                labels3d[i[0], i[1], i_label] = 1

        plot_clustering(labels3d, paxinos3d, n_cluster, level, path_output_folder_results_clustering)

    logging.info("Done!")
