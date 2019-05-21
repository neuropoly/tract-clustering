# This script loops across slices and registers slice i-1 on slice i and
# slice i+1 on slice i. Then, it applies the transformations on the neighbor
# slices and outputs a 3d volume (x, y, 3) for each metric.

import seaborn as sns
import pickle

import numpy as np
import scipy as sp
import nibabel as nib
import pandas as pd
import cv2
import os
import subprocess
import matplotlib


matplotlib.use('TkAgg')

from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from mpl_toolkits.mplot3d import Axes3D
from PIL import Image
from scipy import misc
from subprocess import check_output

from sklearn import datasets
from sklearn.cluster import *
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
from sklearn.feature_extraction.image import grid_to_graph

from os import listdir
from os.path import isfile, join
from os import walk
from collections import OrderedDict
from collections import Counter

sns.set(font_scale=1.4)
sns.set_style("whitegrid", {'axes.grid': False})


# Parameters
# FOLDER: folder that contains all slices and all metrics
Folder = "/Users/hanam/Documents/Tracts_testing_2/all_levels/"


# METRICS: list of metrics to register, e.g. [avf.nii.gz, ad.nii.gz]
Metrics = [
    "Axon_Density",
    "axon_equiv_diameter",
    "AVF_corrected",
    "GR_corrected",
    "Myelin_thickness",
    "MVF_corrected"
]

# METRIC_REF: file name of metric to use as reference for registration
# Metric_ref= nib.load(Folder/C1/Sample1_AD_nii.gz)

# Read FOLDER that contains all slices and all metrics
list_levels = next(walk(Folder))[1]
list_levels[0]

# Create array of levels to have levels in proper order
Cervical = ["C1",
            "C2",
            "C3",
            "C4",
            "C5",
            "C6",
            "C7",
            "C8",
            ]

Thoracic = ["T1",
            "T2",
            "T3",
            "T4",
            "T5",
            "T6",
            "T7",
            "T8",
            "T9",
            "T10",
            "T11",
            "T12",
            "T13",
            ]

Lumbar = ["L1",
          "L2",
          "L3",
          "L4",
          "L5",
          "L6",
          ]

Sacral = ["S1",
          "S2",
          "S3",
          "S4",
          ]

Cervical_list = []
for folder_name in Cervical:
    folder_path = os.path.join(Folder, folder_name)
    filename = os.path.join(folder_path, "Volume4D_sym_cleaned.nii.gz")
    Cervical_list.append(filename)

print Cervical_list

Thoracic_list = []
for folder_name in Thoracic:
    folder_path = os.path.join(Folder, folder_name)
    filename = os.path.join(folder_path, "Volume4D_sym_cleaned.nii.gz")
    Thoracic_list.append(filename)

print Thoracic_list


Lumbar_list = []
for folder_name in Lumbar:
    folder_path = os.path.join(Folder, folder_name)
    filename = os.path.join(folder_path, "Volume4D_sym_cleaned.nii.gz")
    Lumbar_list.append(filename)

Sacral_list = []
for folder_name in Sacral:
    folder_path = os.path.join(Folder, folder_name)
    filename = os.path.join(folder_path, "Volume4D_sym_cleaned.nii.gz")
    Sacral_list.append(filename)

print Sacral_list


def preprocess_file(moving, fixed):
    # 1. Open each file
    x = nib.load(level[moving])
    y = nib.load(level[fixed])

    # 2. Filter the metric you want
    moving = x.get_data()
    moving = moving.transpose((1, 0, 2, 3))
    moving = moving.squeeze(axis=2)
    moving = moving[..., metric]

    fixed = y.get_data()
    fixed = fixed.transpose((1, 0, 2, 3))
    fixed = fixed.squeeze(axis=2)
    fixed = fixed[..., metric]



    # 3. Create a new file for each file, name it "{}_pre.nii.gz".format(p)
    tmp_moving = nib.Nifti1Image(moving, x.affine, x.header)
    tmp_fixed = nib.Nifti1Image(fixed, y.affine, y.header)

    nib.save(tmp_moving, "/Users/hanam/Documents/Tracts_testing_2/all_levels/temporary_moving.nii.gz")
    nib.save(tmp_fixed, "/Users/hanam/Documents/Tracts_testing_2/all_levels/temporary_fixed.nii.gz")

    return tmp_moving, tmp_fixed



#   Register METRIC_REF(i-x) --> METRIC_REF(i) <--- METRIC_REF(i+x) for Cervical
#     # outputs warp(i-1->1)

level = Cervical_list


for i in range(len(level)):
    metric = 0
    if i in range(1, 4):
        previous = 3 - i
        print previous
        current = 3
        print current

        preprocess_file(previous,
                        current,
                        )
        # #  Register METRIC_REF(i-x) --> METRIC_REF(i)
        subprocess.call(['antsRegistration', "--dimensionality", "2", "--transform", "Affine[0.5]",
                            "--metric", "MeanSquares[/Users/hanam/Documents/Tracts_testing_2/all_levels/temporary_moving.nii.gz,/Users/hanam/Documents/Tracts_testing_2/all_levels/temporary_fixed.nii.gz, 1, 5]",
                            "--convergence", "100x100", "--shrink-factors", "8x4", "--smoothing-sigmas", "1x2vox",
                            "--transform", "BSplineSyN[0.5,2]",
                            "--metric", "MeanSquares[/Users/hanam/Documents/Tracts_testing_2/all_levels/temporary_moving.nii.gz,/Users/hanam/Documents/Tracts_testing_2/all_levels/temporary_fixed.nii.gz, 1, 4]",
                            "--convergence", "100x100x100x100", "--shrink-factors", "8x4x2x1",
                            "--smoothing-sigmas", "0x0x0x0vox",
                            "--output", "/Users/hanam/Documents/Tracts_testing_2/all_levels/warp_tp_tc",
                            "--interpolation", "BSpline[3]"])
        print i

        previous_current = nib.load("/Users/hanam/Documents/Tracts_testing_2/all_levels/warp_tp_tc1Warp.nii.gz")
        nib.save(previous_current, "/Users/hanam/Documents/Tracts_testing_2/all_levels/warp_"+str(previous) + "_"+str(current)+"_cervical.nii.gz")

        for m in range(1, len(Metrics)):
            metric = m
            preprocess_file(previous,
                            current,
                            )
            subprocess.call(['antsApplyTransforms', "--dimensionality",  "2",
                             "--input",  "/Users/hanam/Documents/Tracts_testing_2/all_levels/temporary_fixed.nii.gz",
                             "--output", "/Users/hanam/Documents/Tracts_testing_2/all_levels/Applied_warp_"+str(Metrics[m])+"_"+str(Cervical[previous])+"_on_"+str(Cervical[current])+".nii.gz",
                             "--transform", "/Users/hanam/Documents/Tracts_testing_2/all_levels/warp_"+str(previous) + "_"+str(current)+"_cervical.nii.gz",
                             "/Users/hanam/Documents/Tracts_testing_2/all_levels/warp_tp_tc0GenericAffine.mat",
                             "--reference-image", "/Users/hanam/Documents/Tracts_testing_2/all_levels/temporary_moving.nii.gz"])

    if i == 3:
        continue

    if i > 3:
        print i
        nex = i
        print nex
        current = 3
        print current
        preprocess_file(nex,
                        current,
                        )

      # #  Register METRIC_REF(i+1) --> METRIC_REF(i)
        subprocess.call(['antsRegistration', "--dimensionality", "2", "--transform", "Affine[0.5]",
                         "--metric", "MeanSquares[/Users/hanam/Documents/Tracts_testing_2/all_levels/temporary_moving.nii.gz,/Users/hanam/Documents/Tracts_testing_2/all_levels/temporary_fixed.nii.gz, 1, 5]",
                         "--convergence", "100x100", "--shrink-factors", "8x4", "--smoothing-sigmas", "1x2vox",
                         "--transform", "BSplineSyN[0.5,2]",
                         "--metric", "MeanSquares[/Users/hanam/Documents/Tracts_testing_2/all_levels/temporary_moving.nii.gz,/Users/hanam/Documents/Tracts_testing_2/all_levels/temporary_fixed.nii.gz, 1, 4]",
                         "--convergence", "100x100x100x100", "--shrink-factors", "8x4x2x1",
                         "--smoothing-sigmas", "0x0x0x0vox",
                         "--output", "/Users/hanam/Documents/Tracts_testing_2/all_levels/warp_tn_tc",
                         "--interpolation", "BSpline[3]"])

        next_current = nib.load("/Users/hanam/Documents/Tracts_testing_2/all_levels/warp_tn_tc1Warp.nii.gz")
        nib.save(next_current, "/Users/hanam/Documents/Tracts_testing_2/all_levels/warp_"+str(nex)+"_"+str(current)+"_cervical.nii.gz")

        for m in range(1, len(Metrics)):
            metric = m
            preprocess_file(nex,
                            current,
                            )
            subprocess.call(['antsApplyTransforms', "--dimensionality",  "2",
                             "--input",  "/Users/hanam/Documents/Tracts_testing_2/all_levels/temporary_fixed.nii.gz",
                             "--output", "/Users/hanam/Documents/Tracts_testing_2/all_levels/Applied_warp_"+str(Metrics[m])+"_"+str(Cervical[nex])+"_on_"+str(Cervical[current])+".nii.gz",
                             "--transform", "/Users/hanam/Documents/Tracts_testing_2/all_levels/warp_"+str(nex) + "_"+str(current)+"_cervical.nii.gz",
                             "/Users/hanam/Documents/Tracts_testing_2/all_levels/warp_tp_tc0GenericAffine.mat",
                             "--reference-image", "/Users/hanam/Documents/Tracts_testing_2/all_levels/temporary_moving.nii.gz"])

    for mm in range(1, len(Metrics)):

        x = nib.load("/Users/hanam/Documents/Tracts_testing_2/all_levels/Applied_warp_"+str(Metrics[mm])+"_"+str(Cervical[i])+"_on_"+str(Cervical[3])+".nii.gz")

        Volume_4D = np.zeros((151, 151, 1, 6))
        Volume_4D = Volume_4D[:, :, i, x]
        nib.save(Volume_4D, "/Users/hanam/Documents/Tracts_testing_2/all_levels/Volume_4D_cervical.nii.gz")

#   Register METRIC_REF(i-x) --> METRIC_REF(i) <--- METRIC_REF(i+x) for Thoracic
#     # outputs warp(i-1->1)
level = Thoracic_list
metric = 0


for ii in range(len(level)):
    if ii in range(1, 7):
        previous = 6 - ii
        print previous
        current = 6
        print current

        preprocess_file(previous,
                        current,
                        )
        # #  Register METRIC_REF(i-x) --> METRIC_REF(i)
        subprocess.call(['antsRegistration', "--dimensionality", "2", "--transform", "Affine[0.5]",
                         "--metric",
                         "MeanSquares[/Users/hanam/Documents/Tracts_testing_2/all_levels/temporary_moving.nii.gz,/Users/hanam/Documents/Tracts_testing_2/all_levels/temporary_fixed.nii.gz, 1, 5]",
                         "--convergence", "100x100", "--shrink-factors", "8x4", "--smoothing-sigmas", "1x2vox",
                         "--transform", "BSplineSyN[0.5,2]",
                         "--metric",
                         "MeanSquares[/Users/hanam/Documents/Tracts_testing_2/all_levels/temporary_moving.nii.gz,/Users/hanam/Documents/Tracts_testing_2/all_levels/temporary_fixed.nii.gz, 1, 4]",
                         "--convergence", "100x100x100x100", "--shrink-factors", "8x4x2x1",
                         "--smoothing-sigmas", "0x0x0x0vox",
                         "--output", "/Users/hanam/Documents/Tracts_testing_2/all_levels/warp_tp_tc",
                         "--interpolation", "BSpline[3]"])

        previous_current = nib.load("/Users/hanam/Documents/Tracts_testing_2/all_levels/warp_tp_tc1Warp.nii.gz")
        nib.save(previous_current, "/Users/hanam/Documents/Tracts_testing_2/all_levels/warp_" + str(previous) + "_" + str(current) + "_thoracic.nii.gz")

        for m in range(1, len(Metrics)):
            metric = m
            preprocess_file(previous,
                            current,
                            )
            subprocess.call(['antsApplyTransforms', "--dimensionality",  "2",
                             "--input",  "/Users/hanam/Documents/Tracts_testing_2/all_levels/temporary_fixed.nii.gz",
                             "--output", "/Users/hanam/Documents/Tracts_testing_2/all_levels/Applied_warp_"+str(Metrics[m])+"_"+str(Thoracic[previous])+"_on_"+str(Thoracic[current])+".nii.gz",
                             "--transform", "/Users/hanam/Documents/Tracts_testing_2/all_levels/warp_"+str(previous) + "_"+str(current)+"_thoracic.nii.gz",
                             "/Users/hanam/Documents/Tracts_testing_2/all_levels/warp_tp_tc0GenericAffine.mat",
                             "--reference-image", "/Users/hanam/Documents/Tracts_testing_2/all_levels/temporary_moving.nii.gz"])

    if ii == 6:
        continue



    if ii > 6:
        print ii
        current = 6
        print current
        nex = ii
        print nex

        preprocess_file(nex,
                        current,
                        )

        # #  Register METRIC_REF(i+1) --> METRIC_REF(i)
        subprocess.call(['antsRegistration', "--dimensionality", "2", "--transform", "Affine[0.5]",
                         "--metric",
                         "MeanSquares[/Users/hanam/Documents/Tracts_testing_2/all_levels/temporary_moving.nii.gz,/Users/hanam/Documents/Tracts_testing_2/all_levels/temporary_fixed.nii.gz, 1, 5]",
                         "--convergence", "100x100", "--shrink-factors", "8x4", "--smoothing-sigmas", "1x2vox",
                         "--transform", "BSplineSyN[0.5,2]",
                         "--metric",
                         "MeanSquares[/Users/hanam/Documents/Tracts_testing_2/all_levels/temporary_moving.nii.gz,/Users/hanam/Documents/Tracts_testing_2/all_levels/temporary_fixed.nii.gz, 1, 4]",
                         "--convergence", "100x100x100x100", "--shrink-factors", "8x4x2x1",
                         "--smoothing-sigmas", "0x0x0x0vox",
                         "--output", "/Users/hanam/Documents/Tracts_testing_2/all_levels/warp_tn_tc",
                         "--interpolation", "BSpline[3]"])
    
        print ii
    
        next_current = nib.load("/Users/hanam/Documents/Tracts_testing_2/all_levels/warp_tn_tc1Warp.nii.gz")
        nib.save(next_current, "/Users/hanam/Documents/Tracts_testing_2/all_levels/warp_" + str(nex) + "_" + str(current) + "_thoracic.nii.gz")

        for m in range(1, len(Metrics)):
            metric = m
            preprocess_file(nex,
                            current,
                            )
            subprocess.call(['antsApplyTransforms', "--dimensionality",  "2",
                             "--input",  "/Users/hanam/Documents/Tracts_testing_2/all_levels/temporary_fixed.nii.gz",
                             "--output", "/Users/hanam/Documents/Tracts_testing_2/all_levels/Applied_warp_"+str(Metrics[m])+"_"+str(Thoracic[nex])+"_on_"+str(Thoracic[current])+".nii.gz",
                             "--transform", "/Users/hanam/Documents/Tracts_testing_2/all_levels/warp_"+str(nex) + "_"+str(current)+"_thoracic.nii.gz",
                             "/Users/hanam/Documents/Tracts_testing_2/all_levels/warp_tp_tc0GenericAffine.mat",
                             "--reference-image", "/Users/hanam/Documents/Tracts_testing_2/all_levels/temporary_moving.nii.gz"])

#   Register METRIC_REF(i-x) --> METRIC_REF(i) <--- METRIC_REF(i+x) for Thoracic
#     # outputs warp(i-1->1)
level = Lumbar_list
metric = 0


for iii in range(len(level)):
    if iii in range(1, 3):
        previous = 2 - iii
        print previous
        current = 2
        print current

        preprocess_file(previous,
                        current,
                        )
        # #  Register METRIC_REF(i-x) --> METRIC_REF(i)
        subprocess.call(['antsRegistration', "--dimensionality", "2", "--transform", "Affine[0.5]",
                         "--metric",
                         "MeanSquares[/Users/hanam/Documents/Tracts_testing_2/all_levels/temporary_moving.nii.gz,/Users/hanam/Documents/Tracts_testing_2/all_levels/temporary_fixed.nii.gz, 1, 5]",
                         "--convergence", "100x100", "--shrink-factors", "8x4", "--smoothing-sigmas", "1x2vox",
                         "--transform", "BSplineSyN[0.5,2]",
                         "--metric",
                         "MeanSquares[/Users/hanam/Documents/Tracts_testing_2/all_levels/temporary_moving.nii.gz,/Users/hanam/Documents/Tracts_testing_2/all_levels/temporary_fixed.nii.gz, 1, 4]",
                         "--convergence", "100x100x100x100", "--shrink-factors", "8x4x2x1",
                         "--smoothing-sigmas", "0x0x0x0vox",
                         "--output", "/Users/hanam/Documents/Tracts_testing_2/all_levels/warp_tp_tc",
                         "--interpolation", "BSpline[3]"])

        previous_current = nib.load("/Users/hanam/Documents/Tracts_testing_2/all_levels/warp_tp_tc1Warp.nii.gz")
        nib.save(previous_current, "/Users/hanam/Documents/Tracts_testing_2/all_levels/warp_" + str(previous) + "_" + str(current) + "_lumbar.nii.gz")

        for m in range(1, len(Metrics)):
            metric = m
            preprocess_file(previous,
                            current,
                            )
            subprocess.call(['antsApplyTransforms', "--dimensionality",  "2",
                             "--input",  "/Users/hanam/Documents/Tracts_testing_2/all_levels/temporary_fixed.nii.gz",
                             "--output", "/Users/hanam/Documents/Tracts_testing_2/all_levels/Applied_warp_"+str(Metrics[m])+"_"+str(Lumbar[previous])+"_on_"+str(Lumbar[current])+".nii.gz",
                             "--transform", "/Users/hanam/Documents/Tracts_testing_2/all_levels/warp_"+str(previous) + "_"+str(current)+"_lumbar.nii.gz",
                             "/Users/hanam/Documents/Tracts_testing_2/all_levels/warp_tp_tc0GenericAffine.mat",
                             "--reference-image", "/Users/hanam/Documents/Tracts_testing_2/all_levels/temporary_moving.nii.gz"])

    if iii == 2:
        continue



    if iii > 2:
        print iii
        current = 2
        print current
        nex = iii
        print nex

        preprocess_file(nex,
                        current,
                        )
    
        # #  Register METRIC_REF(i+1) --> METRIC_REF(i)
        subprocess.call(['antsRegistration', "--dimensionality", "2", "--transform", "Affine[0.5]",
                         "--metric",
                         "MeanSquares[/Users/hanam/Documents/Tracts_testing_2/all_levels/temporary_moving.nii.gz,/Users/hanam/Documents/Tracts_testing_2/all_levels/temporary_fixed.nii.gz, 1, 5]",
                         "--convergence", "100x100", "--shrink-factors", "8x4", "--smoothing-sigmas", "1x2vox",
                         "--transform", "BSplineSyN[0.5,2]",
                         "--metric",
                         "MeanSquares[/Users/hanam/Documents/Tracts_testing_2/all_levels/temporary_moving.nii.gz,/Users/hanam/Documents/Tracts_testing_2/all_levels/temporary_fixed.nii.gz, 1, 4]",
                         "--convergence", "100x100x100x100", "--shrink-factors", "8x4x2x1",
                         "--smoothing-sigmas", "0x0x0x0vox",
                         "--output", "/Users/hanam/Documents/Tracts_testing_2/all_levels/warp_tn_tc",
                         "--interpolation", "BSpline[3]"])
        
        print iii
    
        next_current = nib.load("/Users/hanam/Documents/Tracts_testing_2/all_levels/warp_tn_tc1Warp.nii.gz")
        nib.save(next_current, "/Users/hanam/Documents/Tracts_testing_2/all_levels/warp_" + str(nex) + "_" + str(current) + "_lumbar.nii.gz")

        for m in range(1, len(Metrics)):
            metric = m
            preprocess_file(nex,
                            current,
                            )
            subprocess.call(['antsApplyTransforms', "--dimensionality",  "2",
                             "--input",  "/Users/hanam/Documents/Tracts_testing_2/all_levels/temporary_fixed.nii.gz",
                             "--output", "/Users/hanam/Documents/Tracts_testing_2/all_levels/Applied_warp_"+str(Metrics[m])+"_"+str(Lumbar[nex])+"_on_"+str(Lumbar[current])+".nii.gz",
                             "--transform", "/Users/hanam/Documents/Tracts_testing_2/all_levels/warp_"+str(nex) + "_"+str(current)+"_lumbar.nii.gz",
                             "/Users/hanam/Documents/Tracts_testing_2/all_levels/warp_tp_tc0GenericAffine.mat",
                             "--reference-image", "/Users/hanam/Documents/Tracts_testing_2/all_levels/temporary_moving.nii.gz"])


#   Register METRIC_REF(i-x) --> METRIC_REF(i) <--- METRIC_REF(i+x) for Thoracic
#     # outputs warp(i-1->1)
level = Sacral_list
metric = 0


for iiii in range(len(level)):
    if iiii in range(1):
        previous = 1 - iiii
        current = 1

        preprocess_file(previous,
                        current,
                        )
        # #  Register METRIC_REF(i-x) --> METRIC_REF(i)
        subprocess.call(['antsRegistration', "--dimensionality", "2", "--transform", "Affine[0.5]",
                         "--metric",
                         "MeanSquares[/Users/hanam/Documents/Tracts_testing_2/all_levels/temporary_moving.nii.gz,/Users/hanam/Documents/Tracts_testing_2/all_levels/temporary_fixed.nii.gz, 1, 5]",
                         "--convergence", "100x100", "--shrink-factors", "8x4", "--smoothing-sigmas", "1x2vox",
                         "--transform", "BSplineSyN[0.5,2]",
                         "--metric",
                         "MeanSquares[/Users/hanam/Documents/Tracts_testing_2/all_levels/temporary_moving.nii.gz,/Users/hanam/Documents/Tracts_testing_2/all_levels/temporary_fixed.nii.gz, 1, 4]",
                         "--convergence", "100x100x100x100", "--shrink-factors", "8x4x2x1",
                         "--smoothing-sigmas", "0x0x0x0vox",
                         "--output", "/Users/hanam/Documents/Tracts_testing_2/all_levels/warp_tp_tc",
                         "--interpolation", "BSpline[3]"])

        previous_current = nib.load("/Users/hanam/Documents/Tracts_testing_2/all_levels/warp_tp_tc1Warp.nii.gz")
        nib.save(previous_current, "/Users/hanam/Documents/Tracts_testing_2/all_levels/warp_" + str(previous) + "_" + str(current) + "_sacral.nii.gz")

        for m in range(1, len(Metrics)):
            metric = m
            preprocess_file(previous,
                            current,
                            )
            subprocess.call(['antsApplyTransforms', "--dimensionality",  "2",
                             "--input",  "/Users/hanam/Documents/Tracts_testing_2/all_levels/temporary_fixed.nii.gz",
                             "--output", "/Users/hanam/Documents/Tracts_testing_2/all_levels/Applied_warp_"+str(Metrics[m])+"_"+str(Sacral[previous])+"_on_"+str(Sacral[current])+".nii.gz",
                             "--transform", "/Users/hanam/Documents/Tracts_testing_2/all_levels/warp_"+str(previous) + "_"+str(current)+"_sacral.nii.gz",
                             "/Users/hanam/Documents/Tracts_testing_2/all_levels/warp_tp_tc0GenericAffine.mat",
                             "--reference-image", "/Users/hanam/Documents/Tracts_testing_2/all_levels/temporary_moving.nii.gz"])
    if i == 1:
        continue



    if i > 1:
        current = 1
        nex = iiii

        preprocess_file(nex,
                        current,
                        )

        # #  Register METRIC_REF(i+1) --> METRIC_REF(i)
        subprocess.call(['antsRegistration', "--dimensionality", "2", "--transform", "Affine[0.5]",
                         "--metric",
                         "MeanSquares[/Users/hanam/Documents/Tracts_testing_2/all_levels/temporary_moving.nii.gz,/Users/hanam/Documents/Tracts_testing_2/all_levels/temporary_fixed.nii.gz, 1, 5]",
                         "--convergence", "100x100", "--shrink-factors", "8x4", "--smoothing-sigmas", "1x2vox",
                         "--transform", "BSplineSyN[0.5,2]",
                         "--metric",
                         "MeanSquares[/Users/hanam/Documents/Tracts_testing_2/all_levels/temporary_moving.nii.gz,/Users/hanam/Documents/Tracts_testing_2/all_levels/temporary_fixed.nii.gz, 1, 4]",
                         "--convergence", "100x100x100x100", "--shrink-factors", "8x4x2x1",
                         "--smoothing-sigmas", "0x0x0x0vox",
                         "--output", "/Users/hanam/Documents/Tracts_testing_2/all_levels/warp_tn_tc",
                         "--interpolation", "BSpline[3]"])

        next_current = nib.load("/Users/hanam/Documents/Tracts_testing_2/all_levels/warp_tn_tc1Warp.nii.gz")
        nib.save(next_current, "/Users/hanam/Documents/Tracts_testing_2/all_levels/warp_" + str(nex) + "_" + str(current) + "_sacral.nii.gz")

        for m in range(1, len(Metrics)):
            metric = m
            preprocess_file(nex,
                            current,
                            )
            subprocess.call(['antsApplyTransforms', "--dimensionality",  "2",
                             "--input",  "/Users/hanam/Documents/Tracts_testing_2/all_levels/temporary_fixed.nii.gz",
                             "--output", "/Users/hanam/Documents/Tracts_testing_2/all_levels/Applied_warp_"+str(Metrics[m])+"_"+str(Sacral[nex])+"_on_"+str(Sacral[current])+".nii.gz",
                             "--transform", "/Users/hanam/Documents/Tracts_testing_2/all_levels/warp_"+str(nex) + "_"+str(current)+"_sacral.nii.gz",
                             "/Users/hanam/Documents/Tracts_testing_2/all_levels/warp_tp_tc0GenericAffine.mat",
                             "--reference-image", "/Users/hanam/Documents/Tracts_testing_2/all_levels/temporary_moving.nii.gz"])

# Concatenate the volumes per section
for i in len(Cervical_list):

    Volume_3D = np.zeros((151, 151, 6))
    nib.save(Volume_4D, "/Users/hanam/Documents/Tracts_testing_2/all_levels/ref_4D")




