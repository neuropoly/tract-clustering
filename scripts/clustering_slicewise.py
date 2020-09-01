# #!/usr/bin/env python

# Perform clustering slicewise and generate figure comparing clustering across slices with Paxinos

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

import params


# Initialize logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)  # default: logging.DEBUG, logging.INFO
hdlr = logging.StreamHandler(sys.stdout)
logging.root.addHandler(hdlr)

# seaborn fig params
sns.set(font_scale=1.4)
sns.set_style("whitegrid", {'axes.grid' : False})
np.set_printoptions(threshold=np.inf)

ext = '.nii'

os.chdir(os.path.join(params.FOLDER, params.OUTPUT_FOLDER))
use_mask = True

# # Load mask
# logger.info("Load mask...")
# nii_mask = nib.load(os.path.join(os.path.join(params.FOLDER), params.input_file_prefix[5] + '.nii.gz'))
# data_mask = nii_mask.get_fdata()

# Loop across spinal levels
# TODO; define levels from params.
levels = ['C1', 'C2']
for level in levels:
    # Load data
    # This data has the following content for the 4th dimension:
    # 0: XX
    # 1: XX
    # 5: WM mask
    logger.info("Load data...")
    nii = nib.load(params.file_prefix_all + level + ext)
    #
    data = nii.get_fdata()
    # Crop around spinal cord, and only keep half of it.
    # The way the atlas was built, the right and left sides are perfectly symmetrical (mathematical average). Hence,
    # we can discard one half, without loosing information.
    xmin, xmax = (30, 75)
    ymin, ymax = (40, 120)
    data_crop = data[xmin:xmax, ymin:ymax, :]

    # DEBUG: print fig
    from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
    from matplotlib.figure import Figure
    fig = Figure()
    FigureCanvas(fig)
    ax = fig.add_subplot(111)
    ax.matshow(data_crop[:, :, 0, 5], cmap='gray')
    fig.savefig('fig_data_crop5.png')

    # Reshape to 1d
    ind_mask = np.where(data_crop[:, :, 0, 5])
    mask1d = np.squeeze(data_crop.reshape(-1, 1))

    # Standardize data
    logger.info("Standardize data...")
    # original_shape = data_crop.shape[0:3]
    data2d = data_crop.reshape(-1, data_crop.shape[3])
    scaler = StandardScaler()
    data2d_norm = scaler.fit_transform(data2d)
    del data2d

    # Cluster across metrics (dim 0 -> 4)
    # --> generates 2d*n (n = numb of clusters)

    # save in variable

# display figure (matrix of slices) showing clusters on the right, and paxinos on the left, for each slice.





# TODO: __main__ to make it callable via CLI