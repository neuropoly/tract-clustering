# #!/usr/bin/env python

# This script loops across slices per level and registers slice i on the median slice for that level and.
# Then, it applies the transformations on the same slices for each metric
# outputs a 3d volume (x, y, i) for each metric.


import os
import subprocess
import math
import numpy as np
import nibabel as nib
#import Parameters

#subprocess.call(Parameters)


# TODO: input individual rat samples instead of the average across rats.
# TODO: introduce a parameters.py file with hard-coded path. Alternatively, use flags (e.g. -f FOLDER).
# TODO: try doing the registration across adjacent slices and concatenate warping fields (more accurate registration)


def preprocess_file(moving, fixed):
    # For each image (registered image and image being registered on), the files are loaded, the metric is chosen
    # and then temporarily saved as a new image with the appropriate image header

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

    nib.save(tmp_moving, "temporary_moving.nii")
    nib.save(tmp_fixed, "temporary_fixed.nii")

    return tmp_moving, tmp_fixed


def print_output(file_path):
    '''
    :param file_path:
    :return: prints message that script is now outputting file with path corresponding to file_path
    '''
    print("Creating file \033[0;34m" + file_path + "\033[0;0m...\n")


def split_each_3d_volume_across_z_and_concatenate_metrics_along_t(list_files):
    """
    Split each 3D volume across z, and concatenate metrics along the 4th dimension. The 3rd dimension is a singleton.
    :param list_files:
    :return:
    """

    list_nii = []
    for file in list_files:
        list_nii.append(nib.Nifti1Image.load(file))

    nz = list_nii[0].get_data().shape[2]
    for i_z in range(nz):
        data_2d_metrics = \
            np.stack([list_nii[i].get_data()[:, :, i_z, np.newaxis] for i in range(len(list_files))], axis=3)
        nii_metric = nib.Nifti1Image(data_2d_metrics, list_nii[0].affine, list_nii[0].header)
        nib.save(nii_metric, 'AtlasRat_AllMetrics_z{}'.format(i_z))


commandNameTemp = os.path.basename(__file__).strip(".py")
# Parameters
# FOLDER: folder that contains all slices and all metrics
Folder = "/Users/hanam/Desktop/AtlasRat"
output_folder = 'results'

os.makedirs(os.path.join(Folder, output_folder), exist_ok=True)
os.chdir(os.path.join(Folder, output_folder))


list_files = [
    'AtlasRat_AD.nii.gz',
    'AtlasRat_AED.nii.gz',
    'AtlasRat_GR.nii.gz',
    'AtlasRat_MT.nii.gz',
    'AtlasRat_MVF.nii.gz',
]

# METRICS: list of metrics to register, e.g. [avf.nii.gz, ad.nii.gz]
Metrics = [
    "Axon_Density",
    "axon_equiv_diameter",
    # "AVF_corrected",
    "GR_corrected",
    "Myelin_thickness",
    "MVF_corrected"
]

# Read FOLDER that contains all slices and all metrics
# list_levels = next(os.walk(Folder))[1]

# Create array of levels to have levels in proper order
Cervical = ["C1",
            "C2",
            "C3",
            "C4",
            "C5",
            "C6",
            "C7",
            "C8"]

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
            "T13"]

Lumbar = ["L1",
          "L2",
          "L3",
          "L4",
          "L5",
          "L6"]

Sacral = ["S1",
          "S2",
          "S3",
          "S4"]


Cervical_list = []
Thoracic_list = []
Lumbar_list = []
Sacral_list = []

split_each_3d_volume_across_z_and_concatenate_metrics_along_t([os.path.join(Folder, i) for i in list_files])

level_array_list = [Cervical, Thoracic, Lumbar, Sacral]
levels = [Cervical_list, Thoracic_list, Lumbar_list, Sacral_list]
i_z = 0
for level_array_number, level_array in enumerate(level_array_list):
    for folder_name in level_array:
        # folder_path = os.path.join(Folder, folder_name)
        # filename = os.path.join(folder_path, "Volume4D_sym_cleaned.nii.gz")
        levels[level_array_number].append('AtlasRat_AllMetrics_z{}.nii'.format(i_z))
        # TODO: have file names with suffix C2, C3, etc.
        i_z += 1


#   Register METRIC_REF(i-x) --> METRIC_REF(i) <--- METRIC_REF(i+x) for Cervical
#     # outputs warp(i-1->1)


level_names = ["cervical", "thoracic", "lumbar", "sacral"]
# For each level, the registration will follow this logic:
# Let's assume 7 slices in a region. The central slice is 4.
# - slice 2 will be registered to slice 3
# - slice 1 will be registered to slice 3
# - slice 0 will be registered to slice 3
for level_number, level in enumerate(levels):
     print("\033[1;35mNow processing region: " + level_names[level_number] + "...\033[0;0m")
     for i in range(len(level)):
          metric = 0
          central_level = int(math.ceil(len(level) / 2))
          if i in range(1, central_level):
               moving_image = (central_level - 1) - i
               fixed_image = (central_level - 1)

               preprocess_file(moving_image, fixed_image)

               applied_warp = 'warp_{}_{}_to_{}'.format(level_names[level_number], moving_image, fixed_image)
               # #  Register METRIC_REF(i-x) --> METRIC_REF(i)
               # print_output(applied_warp + "1Warp.nii.gz")
               # print_output(applied_warp + "0GenericAffine.mat")
               cmd = ['antsRegistration', "--dimensionality", "2", "--transform", "Affine[0.5]",
                                    "--metric", "MeanSquares[", "temporary_fixed.nii,", "temporary_moving.nii, 1, 5]",
                                    "--convergence", "100x100", "--shrink-factors", "8x4", "--smoothing-sigmas", "4x2vox",
                                    "--transform", "BSplineSyN[0.5, 2, 0]",
                                    "--metric", "MeanSquares[", "temporary_fixed.nii,", "temporary_moving.nii, 1, 4]",
                                    "--convergence", "100x100x100x100", "--shrink-factors", "8x4x2x1",
                                    "--smoothing-sigmas", "4x2x1x0vox",
                                    "--output", "[", applied_warp, ",temporary_moving_reg.nii]", "--interpolation", "BSpline[3]"]
               print(' '.join(cmd))
               subprocess.call(cmd)

               # "--output", applied_warp, "temporary_moving_out" + str(moving_image) + "to" + str(
               #     fixed_image) + ".nii.gz", "aslkf", "--interpolation", "BSpline[3]"])

               # Apply transformation to all metrics
               for m in range(0, len(Metrics)):
                   metric = m
                   preprocess_file(moving_image, fixed_image,)
                   print_output("Applied_warp_" + str(Metrics[m]) + "_" + str(
                   level_array_list[level_number][moving_image]) + "_on_" + str(level_array_list[level_number][fixed_image]) + ".nii.gz")
                   cmd = ['antsApplyTransforms',
                          "--dimensionality",
                          "2",
                          "--input",
                          "temporary_moving.nii",
                          "--output",
                          "Applied_warp_" + str(Metrics[m]) + "_" + str(level_array_list[level_number][moving_image]) + "_on_" + str(level_array_list[level_number][fixed_image]) + ".nii.gz",
                          "--transform",
                          applied_warp + "1Warp.nii.gz",
                          applied_warp + "0GenericAffine.mat",
                          "--reference-image",
                          "temporary_fixed.nii",
                          "--interpolation",
                          "BSpline[3]",
                          ]
                   print(' '.join(cmd))
                   subprocess.call(cmd)


          if i > (central_level - 1):
                moving_image = i
                fixed_image = (central_level - 1)
                preprocess_file(moving_image, fixed_image,)
                #applied_warp = "warp_" + str(moving_image) + "_" + str(fixed_image) + "_" + level_names[level_number]
                print_output(applied_warp + "1Warp.nii.gz")
                print_output(applied_warp + "0GenericAffine.mat")
                # #  Register METRIC_REF(i +1) --> METRIC_REF(i)
                cmd = ['antsRegistration', "--dimensionality", "2", "--transform", "Affine[0.5]",
                                     "--metric", "MeanSquares[", "temporary_fixed.nii,",
                                     "temporary_moving.nii, 1, 5]",
                                     "--convergence", "100x100", "--shrink-factors", "8x4", "--smoothing-sigmas",
                                     "4x2vox",
                                     "--transform", "BSplineSyN[0.5,2]",
                                     "--metric", "MeanSquares[", "temporary_fixed.nii,",
                                     "temporary_moving.nii, 1, 4]",
                                     "--convergence", "100x100x100x100", "--shrink-factors", "8x4x2x1",
                                     "--smoothing-sigmas", "4x2x1x0vox",
                                     "--output", "[", applied_warp, ",temporary_moving_reg.nii]",
                                     "--interpolation", "BSpline[3]"]
                print(' '.join(cmd))
                subprocess.call(cmd)

                #subprocess.call(['antsRegistration', "--dimensionality", "2", "--transform", "Affine[0.5]",
                #                "--metric", "MeanSquares[",  "temporary_fixed.nii.gz,",
                #              "temporary_moving.nii.gz, 1, 4]",
                #              "--convergence", "0x0", "--shrink-factors", "8x4", "--smoothing-sigmas", "2x2vox",
                #             "--output", "warpinit_",
                #            "--interpolation", "BSpline[3]"])

                #subprocess.call(['antsRegistration', "--dimensionality", "2", "--transform", "Affine[0.5]",
                #                 "--metric", "MeanSquares[", "temporary_fixed.nii.gz,",
                #                 "temporary_moving.nii.gz, 1, 4]",
                #                 "--convergence", "100x100", "--shrink-factors", "8x4", "--smoothing-sigmas",
                #                 "2x2vox",
                #                 "--transform", "BSplineSyN[0.5,2]",
                #                 "--metric", "MeanSquares[", "temporary_fixed.nii.gz,",
                #                 "temporary_moving.nii.gz, 1, 4]",
                #                 "--convergence", "100x100x100x100", "--shrink-factors", "8x4x2x1",
                #                 "--smoothing-sigmas", "0x0x0x0vox",
                #                 "--output",
                #                 "warp_tn_tc_second_" + str(moving_image) + "_" + str(fixed_image) + "_cervical",
                #                 "--interpolation", "BSpline[3]"])

                #subprocess.call(['sct_concat_transfo', "-d", "temporary_fixed.nii.gz",
                #                 "-w", "warp_tn_tc_first_" + str(moving_image) + "_" + str(
                #        fixed_image) + "_cervical0GenericAffine.mat",
                #                 "warp_tp_tc_second_" + str(moving_image) + "_" + str(
                #                     fixed_image) + "_cervical1Warp.nii.gz",
                #                 "-o", "warp_tn_tc_combined_" + str(moving_image) + "_" + str(fixed_image) + ".nii.gz"])



                for m in range(0, len(Metrics)):
                    metric = m
                    preprocess_file(moving_image,
                                    fixed_image,
                                    )
                    cmd = ['antsApplyTransforms', "--dimensionality", "2",
                                        "--input", "temporary_moving.nii",
                                        "--output", "Applied_warp_" + str(Metrics[m]) + "_" + str(
                            level_array_list[level_number][moving_image]) + "_on_" + str(level_array_list[level_number][fixed_image]) + ".nii.gz",
                                        "--transform",
                                        applied_warp + "1Warp.nii.gz",
                                        applied_warp + "0GenericAffine.mat",
                                        "--reference-image", "temporary_fixed.nii"]
                    print(' '.join(cmd))
                    subprocess.call(cmd)

  # # Concatenate the volumes per section
    # for i in len(level):
    #
    #     Volume_3D = np.zeros((151, 151, 6))
    #     nib.save(Volume_4D, "/Users/hanam/Documents/Tracts_testing_2/all_levels/ref_4D")
