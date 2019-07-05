# This script loops across slices per level and registers slice i on the median slice for that level and.
# Then, it applies the transformations on the same slices for each metric
# outputs a 3d volume (x, y, i) for each metric.

import seaborn as sns
import pickle

import numpy as np
import scipy as sp
import nibabel as nib
import os
import subprocess

from os import listdir
from os.path import isfile, join
from os import walk



# Parameters
# FOLDER: folder that contains all slices and all metrics
Folder = "/volumes/projects/tract_clustering/data/all_levels/"
os.chdir(Folder)

# METRICS: list of metrics to register, e.g. [avf.nii.gz, ad.nii.gz]
Metrics = [
    "Axon_Density",
    "axon_equiv_diameter",
    "AVF_corrected",
    "GR_corrected",
    "Myelin_thickness",
    "MVF_corrected"
]

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

# Create array of levels to have levels in proper order
array_levels = ["C1",
                "C2",
                "C3",
                "C4",
                "C5",
                "C6",
                "C7",
                "C8",
                "T1",
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
                "L1",
                "L2",
                "L3",
                "L4",
                "L5",
                "L6",
                "S1",
                "S2",
                "S3",
                "S4"
]



filelist= []
for folder_name in array_levels:
    folder_path = os.path.join(Folder, folder_name)
    filename = os.path.join(folder_path, "Volume4D_sym_cleaned.nii.gz")
    for i in Cervical:
        Cervical_list.append(filename)

# TODO: remove dupl.

    Cervical_list = range(filelist[0], filelist[7])

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
    # TODO: ADD HEADER TO SAY WHAT YOU DO HERE
    # 1. Open each file
    x = nib.load(level[moving])
    y = nib.load(level[fixed])

    # 2. Select the metric you want
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

    nib.save(tmp_moving, "/temporary_moving.nii.gz")
    nib.save(tmp_fixed, "/temporary_fixed.nii.gz")

    return tmp_moving, tmp_fixed


#   Register METRIC_REF(i-x) --> METRIC_REF(i) <--- METRIC_REF(i+x) for Cervical
# TODO: don<t duplicate levels: use same code for cervical, thoracic, etc.
#     # outputs warp(i-1->1)

# TODO: chdir in working dir, remove all the absolute path (clarity),
level = Cervical_list

# TODO: unify code
for i in range(len(filelist)):
    if i in range(len(filelist[0]), len(filelist[7])):






# TODO: comment
for i in range(len(level)):
    metric = 0

    if i in range(1, ):  # TODO: no hardcoded indices: find them using list length
        moving_image = 3 - i
        print moving_image
        fixed_image = 3
        print fixed_image

        preprocess_file(moving_image,
                        fixed_image,
                        )
        # #  Register METRIC_REF(i-x) --> METRIC_REF(i)
        subprocess.call(['antsRegistration', "--dimensionality", "2", "--transform", "Affine[0.5]",
                            "--metric", "MeanSquares[", "temporary_fixed.nii.gz,", "temporary_moving.nii.gz, 1, 5]",
                            "--convergence", "100x100", "--shrink-factors", "8x4", "--smoothing-sigmas", "2x2vox",
                            "--transform", "BSplineSyN[0.5,2]",
                            "--metric", "MeanSquares[", "temporary_fixed.nii.gz,", "temporary_moving.nii.gz, 1, 4]",
                            "--convergence", "100x100x100x100", "--shrink-factors", "8x4x2x1",
                            "--smoothing-sigmas", "0x0x0x0vox",
                            "--output", "[warp_tp_tc_first_"+str(moving_image) + "_" + str(fixed_image) + "_cervical, temporary_moving_out"+str(moving_image)+"to"+str(fixed_image)+".nii.gz]",
                            "--interpolation", "BSpline[3]"])
        # TODO: display syntax for easy copy/paste on Terminal

        #
        # subprocess.call(['antsRegistration', "--dimensionality", "2", "--transform", "Affine[0.5]",
        #                     "--metric", "MeanSquares[", Folder+"temporary_fixed.nii.gz,", Folder+"temporary_moving.nii.gz, 1, 4]",
        #                     "--convergence", "0x0", "--shrink-factors", "8x4", "--smoothing-sigmas", "2x2vox",
        #                     "--output", Folder+"warpinit_",
        #                     "--interpolation", "BSpline[3]"])
        #
        # subprocess.call(['antsRegistration', "--dimensionality", "2", "--transform", "Affine[0.5]",
        #                     "--metric", "MeanSquares[", Folder+"temporary_fixed.nii.gz,", Folder+"temporary_moving.nii.gz, 1, 4]",
        #                     "--convergence", "100x100", "--shrink-factors", "8x4", "--smoothing-sigmas", "5x2vox",
        #                     "--transform", "BSplineSyN[0.5,2]",
        #                     "--metric", "MeanSquares[", Folder+"temporary_fixed.nii.gz,", Folder+"temporary_moving.nii.gz, 1, 4]",
        #                     "--convergence", "100x100x100x100", "--shrink-factors", "8x4x2x1",
        #                     "--smoothing-sigmas", "0x0x0x0vox",
        #                     "--output", Folder+"warp_tp_tc_second_"+str(previous) + "_" + str(current) + "_cervical",
        #                     "--interpolation", "BSpline[3]"])

        # subprocess.call(['sct_concat_transfo', "-d", Folder+"temporary_fixed.nii.gz", "-w", Folder+"warp_tp_tc_first_" + str(previous) + "_" + str(current) + "_cervical0GenericAffine.mat", Folder+"warp_tp_tc_second_" + str(previous) + "_" + str(current) + "_cervical1Warp.nii.gz", "-o", Folder+"warp_tp_tc_combined_"+ str(previous) + "_" + str(current) +".nii.gz"])

        print i



        for m in range(0, len(Metrics)):
            metric = m
            preprocess_file(previous,
                            current,
                            )
            subprocess.call(['antsApplyTransforms', "--dimensionality",  "2",
                             "--input", "temporary_fixed.nii.gz",
                             "--output", "/Applied_warp_"+str(Metrics[m])+"_"+str(Cervical[previous])+"_on_"+str(Cervical[current])+".nii.gz",
                             "--transform", "warpinit_0GenericAffine.mat", "warp_tp_tc_combined_"+ str(previous) + "_" + str(current) +".nii.gz",
                             "--reference-image", "temporary_fixed.nii.gz"])

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
                         "--metric", "MeanSquares[", Folder + "temporary_fixed.nii.gz,",
                         Folder + "temporary_moving.nii.gz, 1, 5]",
                         "--convergence", "100x100", "--shrink-factors", "8x4", "--smoothing-sigmas", "2x2vox",
                         "--transform", "BSplineSyN[0.5,2]",
                         "--metric", "MeanSquares[", Folder + "temporary_fixed.nii.gz,",
                         Folder + "temporary_moving.nii.gz, 1, 4]",
                         "--convergence", "100x100x100x100", "--shrink-factors", "8x4x2x1",
                         "--smoothing-sigmas", "0x0x0x0vox",
                         "--output", Folder + "warp_tn_tc_first_" + str(nex) + "_" + str(current) + "_cervical",
                         "--interpolation", "BSpline[3]"])

        subprocess.call(['antsRegistration', "--dimensionality", "2", "--transform", "Affine[0.5]",
                         "--metric", "MeanSquares[", Folder + "temporary_fixed.nii.gz,",
                         Folder + "temporary_moving.nii.gz, 1, 4]",
                         "--convergence", "0x0", "--shrink-factors", "8x4", "--smoothing-sigmas", "2x2vox",
                         "--output", Folder + "warpinit_",
                         "--interpolation", "BSpline[3]"])

        subprocess.call(['antsRegistration', "--dimensionality", "2", "--transform", "Affine[0.5]",
                         "--metric", "MeanSquares[", Folder + "temporary_fixed.nii.gz,",
                         Folder + "temporary_moving.nii.gz, 1, 4]",
                         "--convergence", "100x100", "--shrink-factors", "8x4", "--smoothing-sigmas", "2x2vox",
                         "--transform", "BSplineSyN[0.5,2]",
                         "--metric", "MeanSquares[", Folder + "temporary_fixed.nii.gz,",
                         Folder + "temporary_moving.nii.gz, 1, 4]",
                         "--convergence", "100x100x100x100", "--shrink-factors", "8x4x2x1",
                         "--smoothing-sigmas", "0x0x0x0vox",
                         "--output", Folder + "warp_tn_tc_second_" + str(nex) + "_" + str(current) + "_cervical",
                         "--interpolation", "BSpline[3]"])



        subprocess.call(['sct_concat_transfo', "-d", Folder + "temporary_fixed.nii.gz",
                         "-w", Folder + "warp_tn_tc_first_" + str(nex) + "_" + str(current) + "_cervical0GenericAffine.mat",
                         Folder + "warp_tp_tc_second_" + str(nex) + "_" + str(current) + "_cervical1Warp.nii.gz",
                         "-o", Folder + "warp_tn_tc_combined_" + str(nex) + "_" + str(current) + ".nii.gz"])



        for m in range(0, len(Metrics)):
            metric = m
            preprocess_file(nex,
                            current,
                            )
            subprocess.call(['antsApplyTransforms', "--dimensionality",  "2",
                             "--input", Folder+"temporary_fixed.nii.gz",
                             "--output", Folder+"Applied_warp_"+str(Metrics[m])+"_"+str(Cervical[nex])+"_on_"+str(Cervical[current])+".nii.gz",
                             "--transform", Folder+"warpinit_0GenericAffine.mat", Folder+"warp_tp_tc_combined_"+ str(nex) + "_" + str(current) +".nii.gz",
                             "--reference-image", Folder+"temporary_fixed.nii.gz"])

    # for mm in range(1, len(Metrics)):
    #
    #     # x = nib.load("/Users/hanam/Documents/Tracts_testing_2/all_levels/Applied_warp_"+str(Metrics[mm])+"_"+str(Cervical[i])+"_on_"+str(Cervical[3])+".nii.gz")
    #     #
    #     # Volume_4D = np.zeros((151, 151, 1, 6))
    #     # Volume_4D = Volume_4D[:, :, i, x]
    #     # nib.save(Volume_4D, "/Users/hanam/Documents/Tracts_testing_2/all_levels/Volume_4D_cervical.nii.gz")

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
                            "--metric", "MeanSquares[", Folder+"temporary_fixed.nii.gz,", Folder+"temporary_moving.nii.gz, 1, 4]",
                            "--convergence", "200x100x50", "--shrink-factors", "8x4x2", "--smoothing-sigmas", "0x0x0vox",
                            "--transform", "BSplineSyN[0.25,2]",
                            "--metric", "MeanSquares[", Folder+"temporary_fixed.nii.gz,", Folder+"temporary_moving.nii.gz, 1, 4]",
                            "--convergence", "200x100x50", "--shrink-factors", "8x4x2",
                            "--smoothing-sigmas", "0x0x0vox",
                            "--output", Folder+"warp_tp_tc",
                            "--interpolation", "BSpline[3]"])

        previous_current = nib.load(Folder+"warp_tp_tc1Warp.nii.gz")
        nib.save(previous_current, Folder+"warp_"+str(previous) + "_"+str(current)+"_thoracic.nii.gz")

        for m in range(0, len(Metrics)):
            metric = m
            preprocess_file(previous,
                            current,
                            )
            subprocess.call(['antsApplyTransforms', "--dimensionality",  "2",
                             "--input", Folder+"temporary_fixed.nii.gz",
                             "--output", Folder+"Applied_warp_"+str(Metrics[m])+"_"+str(Thoracic[previous])+"_on_"+str(Thoracic[current])+".nii.gz",
                             "--transform", Folder+"warp_"+str(previous) + "_"+str(current)+"_thoracic.nii.gz",
                             Folder+"warp_tp_tc0GenericAffine.mat",
                             "--reference-image", Folder+"temporary_moving.nii.gz"])

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
                            "--metric", "MeanSquares[", Folder+"temporary_fixed.nii.gz,", Folder+"temporary_moving.nii.gz, 1, 4]",
                            "--convergence", "200x100x50", "--shrink-factors", "8x4x2", "--smoothing-sigmas", "0x0x0vox",
                            "--transform", "BSplineSyN[0.25,2]",
                            "--metric", "MeanSquares[", Folder+"temporary_fixed.nii.gz,", Folder+"temporary_moving.nii.gz, 1, 4]",
                            "--convergence", "200x100x50", "--shrink-factors", "8x4x2",
                            "--smoothing-sigmas", "0x0x0vox",
                            "--output", Folder+"warp_tn_tc",
                            "--interpolation", "BSpline[3]"])
    
        print ii
    
        previous_current = nib.load(Folder+"warp_tn_tc1Warp.nii.gz")
        nib.save(previous_current, Folder+"warp_"+str(nex) + "_"+str(current)+"_thoracic.nii.gz")

        for m in range(0, len(Metrics)):
            metric = m
            preprocess_file(nex,
                            current,
                            )
            subprocess.call(['antsApplyTransforms', "--dimensionality",  "2",
                             "--input", Folder+"temporary_fixed.nii.gz",
                             "--output", Folder+"Applied_warp_"+str(Metrics[m])+"_"+str(Thoracic[nex])+"_on_"+str(Thoracic[current])+".nii.gz",
                             "--transform", Folder+"warp_"+str(nex) + "_"+str(current)+"_thoracic.nii.gz",
                             Folder+"warp_tn_tc0GenericAffine.mat",
                             "--reference-image", Folder+"temporary_fixed.nii.gz"])

#   Register METRIC_REF(i-x) --> METRIC_REF(i) <--- METRIC_REF(i+x) for Lumbar
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
                            "--metric", "MeanSquares[", Folder+"temporary_fixed.nii.gz,", Folder+"temporary_moving.nii.gz, 1, 4]",
                            "--convergence", "200x100x50", "--shrink-factors", "8x4x2", "--smoothing-sigmas", "0x0x0vox",
                            "--transform", "BSplineSyN[0.25,2]",
                            "--metric", "MeanSquares[", Folder+"temporary_fixed.nii.gz,", Folder+"temporary_moving.nii.gz, 1, 4]",
                            "--convergence", "200x100x50", "--shrink-factors", "8x4x2",
                            "--smoothing-sigmas", "0x0x0vox",
                            "--output", Folder+"warp_tp_tc",
                            "--interpolation", "BSpline[3]"])

        previous_current = nib.load(Folder+"warp_tp_tc1Warp.nii.gz")
        nib.save(previous_current, Folder+"warp_"+str(previous) + "_"+str(current)+"_lumbar.nii.gz")

        for m in range(0, len(Metrics)):
            metric = m
            preprocess_file(previous,
                            current,
                            )
            subprocess.call(['antsApplyTransforms', "--dimensionality",  "2",
                             "--input", Folder+"temporary_fixed.nii.gz",
                             "--output", Folder+"Applied_warp_"+str(Metrics[m])+"_"+str(Lumbar[previous])+"_on_"+str(Lumbar[current])+".nii.gz",
                             "--transform", Folder+"warp_"+str(previous) + "_"+str(current)+"_lumbar.nii.gz",
                             Folder+"warp_tp_tc0GenericAffine.mat",
                             "--reference-image", Folder+"temporary_fixed.nii.gz"])

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
                            "--metric", "MeanSquares[", Folder+"temporary_fixed.nii.gz,", Folder+"temporary_moving.nii.gz, 1, 4]",
                            "--convergence", "200x100x50", "--shrink-factors", "8x4x2", "--smoothing-sigmas", "0x0x0vox",
                            "--transform", "BSplineSyN[0.25,2]",
                            "--metric", "MeanSquares[", Folder+"temporary_fixed.nii.gz,", Folder+"temporary_moving.nii.gz, 1, 4]",
                            "--convergence", "200x100x50", "--shrink-factors", "8x4x2",
                            "--smoothing-sigmas", "0x0x0vox",
                            "--output", Folder+"warp_tn_tc",
                            "--interpolation", "BSpline[3]"])
        
        print iii
    
        previous_current = nib.load(Folder+"warp_tn_tc1Warp.nii.gz")
        nib.save(previous_current, Folder+"warp_"+str(nex) + "_"+str(current)+"_lumbar.nii.gz")

        for m in range(0, len(Metrics)):
            metric = m
            preprocess_file(nex,
                            current,
                            )
            subprocess.call(['antsApplyTransforms', "--dimensionality",  "2",
                             "--input", Folder+"temporary_fixed.nii.gz",
                             "--output", Folder+"Applied_warp_"+str(Metrics[m])+"_"+str(Lumbar[nex])+"_on_"+str(Lumbar[current])+".nii.gz",
                             "--transform", Folder+"warp_"+str(nex) + "_"+str(current)+"_lumbar.nii.gz",
                             Folder+"warp_tn_tc0GenericAffine.mat",
                             "--reference-image", Folder+"temporary_fixed.nii.gz"])


#   Register METRIC_REF(i-x) --> METRIC_REF(i) <--- METRIC_REF(i+x) for Sacral
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
                            "--metric", "MeanSquares[", Folder+"temporary_fixed.nii.gz,", Folder+"temporary_moving.nii.gz, 1, 4]",
                            "--convergence", "0x0", "--shrink-factors", "4x2", "--smoothing-sigmas", "0x0vox",
                            "--transform", "BSplineSyN[0.1,2]",
                            "--metric", "MeanSquares[", Folder+"temporary_fixed.nii.gz,", Folder+"temporary_moving.nii.gz, 1, 4]",
                            "--convergence", "0x0", "--shrink-factors", "4x2",
                            "--smoothing-sigmas", "0x0vox",
                            "--output", Folder+"warp_tp_tc",
                            "--interpolation", "BSpline[3]"])

        previous_current = nib.load(Folder+"warp_tp_tc1Warp.nii.gz")
        nib.save(previous_current, Folder+"warp_"+str(previous) + "_"+str(current)+"_sacral.nii.gz")

        for m in range(0, len(Metrics)):
            metric = m
            preprocess_file(previous,
                            current,
                            )
            subprocess.call(['antsApplyTransforms', "--dimensionality",  "2",
                             "--input", Folder+"temporary_fixed.nii.gz",
                             "--output", Folder+"Applied_warp_"+str(Metrics[m])+"_"+str(Sacral[previous])+"_on_"+str(Sacral[current])+".nii.gz",
                             "--transform", Folder+"warp_"+str(previous) + "_"+str(current)+"_sacral.nii.gz",
                             Folder+"warp_tp_tc0GenericAffine.mat",
                             "--reference-image", Folder+"temporary_fixed.nii.gz"])

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
                            "--metric", "MeanSquares[", Folder+"temporary_fixed.nii.gz,", Folder+"temporary_moving.nii.gz, 1, 4]",
                            "--convergence", "200x100x50", "--shrink-factors", "8x4x2", "--smoothing-sigmas", "0x0x0vox",
                            "--transform", "BSplineSyN[0.25,2]",
                            "--metric", "MeanSquares[", Folder+"temporary_fixed.nii.gz,", Folder+"temporary_moving.nii.gz, 1, 4]",
                            "--convergence", "200x100x50", "--shrink-factors", "8x4x2",
                            "--smoothing-sigmas", "0x0x0vox",
                            "--output", Folder+"warp_tn_tc",
                            "--interpolation", "BSpline[3]"])

        previous_current = nib.load(Folder+"warp_tn_tc1Warp.nii.gz")
        nib.save(previous_current, Folder+"warp_"+str(nex) + "_"+str(current)+"_sacral.nii.gz")

        for m in range(0, len(Metrics)):
            metric = m
            preprocess_file(nex,
                            current,
                            )
            subprocess.call(['antsApplyTransforms', "--dimensionality",  "2",
                             "--input", Folder+"temporary_fixed.nii.gz",
                             "--output", Folder+"Applied_warp_"+str(Metrics[m])+"_"+str(Sacral[nex])+"_on_"+str(Sacral[current])+".nii.gz",
                             "--transform", Folder+"warp_"+str(nex) + "_"+str(current)+"_sacral.nii.gz",
                             Folder+"warp_tn_tc0GenericAffine.mat",
                             "--reference-image", Folder+"temporary_fixed.nii.gz"])

# # Concatenate the volumes per section
# for i in len(Cervical_list):
#
#     Volume_3D = np.zeros((151, 151, 6))
#     nib.save(Volume_4D, "/Users/hanam/Documents/Tracts_testing_2/all_levels/ref_4D")




