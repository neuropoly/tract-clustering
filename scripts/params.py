# #!/usr/bin/env python
#
# Parameters for the tract-clustering project.

import logging


# PARAMETERS THAT ARE SPECIFIC TO THE INPUT DATA
# ======================================================================================================================

folder = '/Users/julien/temp/tract-clustering/AtlasRat'  # Path to folder that contains the atlas
folder_output = 'results_clustering'  # Main output folder for tract-clustering pipeline
folder_processing = 'processing_temp'  # Where to put intermediate processed images
folder_concat_region = 'processing_regionwise'

folder_clustering_slicewise = 'results_clustering_slicewise'  # Where to put results of slicewise clustering
folder_clustering_regionwise = 'results_clustering_regionwise'  # Where to put results of region-wise clustering

file_prefix = 'AtlasRat_'

# list of 3d input files
input_file_prefix = [
    'AtlasRat_AD',
    'AtlasRat_AED',
    'AtlasRat_GR',
    'AtlasRat_MT',
    'AtlasRat_MVF',
    'AtlasRat_mask_WM',
]

# Paxinos atlas. Do not add the extension. If defined as empty list, the code will not process this file.
file_paxinos = 'AtlasRat_Paxinos'

metrics = [
    'AD',
    'AED',
    'GR',
    'MT',
    'MVF'
    ]

# Metric used as reference for estimating the warping field, which will then be applied to all other metrics
input_file_prefix_reference = 'AtlasRat_AD'

input_file_ext = '.nii.gz'

regions = {
    'cervical': ["C1", "C2", "C3", "C4", "C5", "C6", "C7", "C8"],
    'thoracic': ["T1", "T2", "T3", "T4", "T5", "T6", "T7", "T8", "T9", "T10", "T11", "T12", "T13"],
    'lumbar': ["L1", "L2", "L3", "L4", "L5", "L6"],
    'sacral': ["S1", "S2", "S3", "S4"],
    }

# Reference destination slice for the registration, per region
reference_level = {
    'cervical': 'C4',
    'thoracic': 'T7',
    'lumbar': 'L3',
    'sacral': 'S2',
    }

xminmax = (30, 75)
yminmax = (40, 105)


# PARAMETERS THAT ARE USED INTERNALLY BY THE CODE
# ======================================================================================================================

logging_mode = logging.INFO  # logging.DEBUG ; logging.INFO

file_prefix_all = 'AtlasRat_AllMetrics_'
file_mask_prefix = 'AtlasRat_mask_WM_'

# colors used to display the clustered tracts
colors = {
    'k': (0, 0, 0),  # 0
    'g': (0, 0.5, 0),  # 1
    'r': (1, 0, 0),  # 2
    'c': (0, 0.75, 0.75),  # 3
    'y': (0.75, 0.75, 0),  # 4
    'm': (0.75, 0, 0.75),  # 5
    'b': (0, 0, 1),  # 6
    'w': (1, 1, 1),  # 7
    'indigo': (0.29411764705882354, 0.0, 0.5098039215686274),  # 8
    'orange': (1.0, 0.6470588235294118, 0.0),  # 9
    'brown': (0.6470588235294118, 0.16470588235294117, 0.16470588235294117),  # 10
    'lightpink': (1.0, 0.7137254901960784, 0.7568627450980392),  # 11
    }

# Mapping of color index between cluster map and paxinos, depending on cluster number
clust2pax = {
    8: [2, 7, 10, 6, 3, 11, 5, 7],
    10: [7, 6, 2, 8, 3, 9, 10, 7, 1, 11],
    }
