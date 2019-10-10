# #!/usr/bin/env python

# This script loops across slices per level and registers slice i on the median slice for that level and.
# Then, it applies the transformations on the same slices for each metric
# outputs a 3d volume (x, y, i) for each metric.


import os
import shutil
import subprocess
import math
import numpy as np
import nibabel as nib

import params


# TODO: input individual rat samples instead of the average across rats.
# TODO: introduce a params.py file with hard-coded path. Alternatively, use flags (e.g. -f FOLDER).
# TODO: try doing the registration across adjacent slices and concatenate warping fields (more accurate registration)


def preprocess_file(file_moving, file_fixed, metric=None):
    """
    For each image (registered image and image being registered on), the files are loaded, the metric is chosen
    and then temporarily saved as a new image with the appropriate image header
    :param moving:
    :param fixed:
    :param metric: str: file prefix of the metric to extract
    :return: output filenames for moving and fixed
    """

    # Open each 3d file (x, y, metric)
    nii_moving = nib.load(file_moving)
    nii_fixed = nib.load(file_fixed)

    # Select the metric you want
    data_moving = nii_moving.get_data()
    data_moving = data_moving.transpose((1, 0, 2, 3))
    data_moving = data_moving.squeeze(axis=2)
    data_moving = data_moving[..., params.input_file_prefix.index(metric)]

    data_fixed = nii_fixed.get_data()
    data_fixed = data_fixed.transpose((1, 0, 2, 3))
    data_fixed = data_fixed.squeeze(axis=2)
    data_fixed = data_fixed[..., params.input_file_prefix.index(metric)]

    # Create a new file for each file, name it "{}_pre.nii.gz".format(p)
    nii_moving_new = nib.Nifti1Image(data_moving, nii_moving.affine, nii_moving.header)
    nii_fixed_new = nib.Nifti1Image(data_fixed, nii_fixed.affine, nii_fixed.header)

    file_tmp_moving = 'temporary_moving.nii'
    file_tmp_fixed = 'temporary_fixed.nii'

    nib.save(nii_moving_new, file_tmp_moving)
    nib.save(nii_fixed_new, file_tmp_fixed)

    return file_tmp_moving, file_tmp_fixed


def print_output(file_path):
    """
    Nice printout with colors
    :param file_path:
    :return: prints message that script is now outputting file with path corresponding to file_path
    """
    print("Creating file \033[0;34m" + file_path + "\033[0;0m...\n")


def register_moving_to_fixed(level_moving, level_fixed, copy_moving=False):
    """
    Register 2D moving image to 2D fixed image with ANTs, and apply transformations to all metrics
    :param level_moving: str: level name of moving image
    :param level_fixed:  str: level name of fixed image
    :param copy_moving: Bool: If True, does not estimate transformation. Only copy moving to fixed.
    :return:
    """
    if not copy_moving:
        # Preprocess images (reorient, etc.) to be compatible with ANTs
        file_tmp_moving, file_tmp_fixed = preprocess_file(params.file_prefix + level_moving + ext,
                                                          params.file_prefix + level_fixed + ext,
                                                          metric=params.input_file_prefix_reference)
        applied_warp = 'warp_{}_to_{}_'.format(level_moving, level_fixed)
        # #  Register METRIC_REF(i-x) --> METRIC_REF(i)
        # print_output(applied_warp + "1Warp.nii.gz")
        # print_output(applied_warp + "0GenericAffine.mat")
        cmd = ['antsRegistration', "--dimensionality", "2", "--transform", "Affine[0.5]",
               "--metric", "MeanSquares[", file_tmp_fixed, ",", file_tmp_moving, ", 1, 5]",
               "--convergence", "100x100", "--shrink-factors", "8x4", "--smoothing-sigmas", "4x2vox",
               "--transform", "BSplineSyN[0.5, 2, 0]",
               "--metric", "MeanSquares[", file_tmp_fixed, ",", file_tmp_moving, ", 1, 4]",
               "--convergence", "100x100x100x100", "--shrink-factors", "8x4x2x1",
               "--smoothing-sigmas", "4x2x1x0vox",
               "--output", "[", applied_warp, ",temporary_moving_reg.nii]", "--interpolation", "BSpline[3]"]
        print(' '.join(cmd))
        subprocess.call(cmd)

    # Apply transformation to all metrics
    for file in params.input_file_prefix:
        # Extract 2d slice for specific metric
        file_tmp_moving, file_tmp_fixed = preprocess_file(params.file_prefix + level_moving + ext,
                                                          params.file_prefix + level_fixed + ext,
                                                          metric=file)
        file_output = file + '_' + level_moving + '_to_' + level_fixed + ext
        print_output(file_output)
        # If reference slice, only copy it (no registration)
        if copy_moving:
            shutil.copy(file_tmp_fixed, file_output)
        # Else, apply warping field
        else:
            cmd = ['antsApplyTransforms',
                   "--dimensionality",
                   "2",
                   "--input",
                   file_tmp_moving,
                   "--output",
                   file_output,
                   "--transform",
                   applied_warp + "1Warp.nii.gz",
                   applied_warp + "0GenericAffine.mat",
                   "--reference-image",
                   file_tmp_fixed,
                   "--interpolation",
                   "BSpline[3]",
                   ]
            print(' '.join(cmd))
            subprocess.call(cmd)

def split_each_3d_volume_across_z_and_concatenate_metrics_along_t(list_files, level_array_list_flat):
    """
    Split each 3D volume across z, and concatenate metrics along the 4th dimension. The 3rd dimension is a singleton.
    :param list_files:
    :param level_array_list_flat: List of ordered levels to use for output file name
    :return: (nx, ny)
    """

    list_nii = []
    for file in list_files:
        list_nii.append(nib.Nifti1Image.load(file))

    nz = list_nii[0].get_data().shape[2]
    for i_z in range(nz):
        data_2d_metrics = \
            np.stack([list_nii[i].get_data()[:, :, i_z, np.newaxis] for i in range(len(list_files))], axis=3)
        nii_metric = nib.Nifti1Image(data_2d_metrics, list_nii[0].affine, list_nii[0].header)
        nib.save(nii_metric, params.file_prefix + '{}'.format(level_array_list_flat[i_z]))

    return nii_metric.shape[0:2]

# SCRIPT STARTS HERE
# ======================================================================================================================

# commandNameTemp = os.path.basename(__file__).strip(".py")

ext = '.nii'  # .nii or .nii.gz

os.makedirs(os.path.join(params.FOLDER, params.OUTPUT_FOLDER), exist_ok=True)
os.chdir(os.path.join(params.FOLDER, params.OUTPUT_FOLDER))

Cervical_list = []
Thoracic_list = []
Lumbar_list = []
Sacral_list = []

# Create flattened list of levels in the proper order
level_array_list = [params.regions['cervical'],
                    params.regions['thoracic'],
                    params.regions['lumbar'],
                    params.regions['sacral']]
level_array_list_flat = [item for sublist in level_array_list for item in sublist]

# Split each metric across z and concatenate along metric dimension
nx, ny = split_each_3d_volume_across_z_and_concatenate_metrics_along_t(
    [os.path.join(params.FOLDER, i+params.input_file_ext) for i in params.input_file_prefix],
    level_array_list_flat)

# levels = [Cervical_list, Thoracic_list, Lumbar_list, Sacral_list]
# i_z = 0
# for level_array_number, level_array in enumerate(level_array_list):
#     for folder_name in level_array:
#         # folder_path = os.path.join(Folder, folder_name)
#         # filename = os.path.join(folder_path, "Volume4D_sym_cleaned.nii.gz")
#         levels[level_array_number].append('AtlasRat_AllMetrics_z{}.nii'.format(i_z))
#         # TODO: have file names with suffix C2, C3, etc.
#         i_z += 1

# For each level, the registration will follow this logic:
# Let's assume 7 slices in a region. The central slice is 4.
# - slice 2 will be registered to slice 3
# - slice 1 will be registered to slice 3
# - slice 0 will be registered to slice 3
# for level_number, level in enumerate(levels):
for region, levels in params.regions.items():
    print("\033[1;35mNow processing region: " + region + "...\033[0;0m")
    for i in range(len(levels)):
        i_fixed_image = levels.index(params.reference_level[region])
        # Define central slice for this region, which will be the destination image for registration
        # For slices below the central slice
        if i in range(0, i_fixed_image):
            i_moving_image = i_fixed_image - i - 1
            register_moving_to_fixed(levels[i_moving_image],
                                     levels[i_fixed_image])
        # For slices above the central slice
        elif i > i_fixed_image:
            i_moving_image = i
            register_moving_to_fixed(levels[i_moving_image],
                                     levels[i_fixed_image])
        # If this is the central slice, just copy it
        elif i == i_fixed_image:
            register_moving_to_fixed(levels[i],
                                     levels[i_fixed_image],
                                     copy_moving=True)

# Concatenate data for each region, according to: (x, y, z, metric)
ofolder = 'concat_within_region'
os.makedirs(ofolder, exist_ok=True)
for region, levels in params.regions.items():
    print("\033[1;35mConcatenate region: " + region + "...\033[0;0m")
    data4d = np.zeros([nx, ny, len(levels), len(params.input_file_prefix)])
    for file_metrics in params.input_file_prefix:
        for level in levels:
            nii2d = nib.Nifti1Image.load(file_metrics + '_' + level + '_to_' + params.reference_level[region] + ext)
            data4d[:, :, levels.index(level), params.input_file_prefix.index(file_metrics)] = nii2d.get_data()
    nii4d = nib.Nifti1Image(data4d, nii2d.affine, nii2d.header)
    nib.save(nii4d, os.path.join(ofolder, params.file_prefix + region + ext))
