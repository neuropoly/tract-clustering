# This script loops across slices per level and registers slice i on the median slice for that level and.
# Then, it applies the transformations on the same slices for each metric
# outputs a 3d volume (x, y, i) for each metric.

#import seaborn as sns
import pickle


import os
import subprocess
import math
import nibabel as nib

from os import walk

# ================================================================================================================
def print_output(file_path):
    '''
    :param file_path:
    :return: prints message that script is now outputting file with path corresponding to file_path
    '''
    print("Creating file \033[0;34m" + file_path + "\033[0;0m...\n")


#=================================================================================================================

commandNameTemp = os.path.basename(__file__).strip(".py")
# Parameters
# FOLDER: folder that contains all slices and all metrics
Folder = "/Users/hanam/Documents/Tracts_testing_2/all_levels/"
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
                "S4"]

Cervical_list = []
Thoracic_list = []
Lumbar_list = []
Sacral_list = []


level_array_list = [Cervical, Thoracic, Lumbar, Sacral]
levels = [Cervical_list, Thoracic_list, Lumbar_list, Sacral_list]
for level_array_number, level_array in enumerate(level_array_list):
    for folder_name in level_array:
        folder_path = os.path.join(Folder, folder_name)
        filename = os.path.join(folder_path, "Volume4D_sym_cleaned.nii.gz")
        levels[level_array_number].append(filename)


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

    nib.save(tmp_moving, "temporary_moving.nii.gz")
    nib.save(tmp_fixed, "temporary_fixed.nii.gz")

    return tmp_moving, tmp_fixed


#   Register METRIC_REF(i-x) --> METRIC_REF(i) <--- METRIC_REF(i+x) for Cervical
# TODO: don<t duplicate levels: use same code for cervical, thoracic, etc. DONE
#     # outputs warp(i-1->1)

# TODO: chdir in working dir, remove all the absolute path (clarity), DONE

level_names = ["cervical", "thoracic", "lumbar", "sacral"]
for level_number, level in enumerate(levels):
     print("\033[1;35mNow processing level " + level_names[level_number] + "...\033[0;0m")
     for i in range(len(level)):
          metric = 0
          central_level = int(math.ceil(len(level) / 2))
          if i in range(1, central_level):
               moving_image = (central_level - 1) - i
               fixed_image = (central_level - 1)

               preprocess_file(moving_image, fixed_image)

               applied_warp = "warp_" + str(moving_image) + "_" + str(fixed_image) + "_" + level_names[level_number]
               # #  Register METRIC_REF(i-x) --> METRIC_REF(i)
               print_output(applied_warp + "1Warp.nii.gz")
               print_output(applied_warp + "0GenericAffine.mat")
               subprocess.call(['antsRegistration', "--dimensionality", "2", "--transform", "Affine[0.5]",
                                    "--metric", "MI[", "temporary_fixed.nii.gz,", "temporary_moving.nii.gz, 1, 5]",
                                    "--convergence", "100x50", "--shrink-factors", "4x2", "--smoothing-sigmas", "2x2vox",
                                    "--transform", "BSplineSyN[0.15,2, 0]",
                                    "--metric", "MeanSquares[", "temporary_fixed.nii.gz,", "temporary_moving.nii.gz, 1, 4]",
                                    "--convergence", "200x50x20", "--shrink-factors", "8x4x2",
                                    "--smoothing-sigmas", "0x0x0vox",
                                    "--output", applied_warp, "--interpolation", "BSpline[3]"])

               # "--output", applied_warp, "temporary_moving_out" + str(moving_image) + "to" + str(
               #     fixed_image) + ".nii.gz", "aslkf", "--interpolation", "BSpline[3]"])



               for m in range(0, len(Metrics)):
                   metric = m
                   preprocess_file(moving_image, fixed_image,)
                   print_output("Applied_warp_" + str(Metrics[m]) + "_" + str(
                   level_array_list[level_number][moving_image]) + "_on_" + str(level_array_list[level_number][fixed_image]) + ".nii.gz")
                   subprocess.call(['antsApplyTransforms', "--dimensionality", "2",
                                        "--input", "temporary_fixed.nii.gz",
                                        "--output", "Applied_warp_" + str(Metrics[m]) + "_" + str(
                   level_array_list[level_number][moving_image]) + "_on_" + str(level_array_list[level_number][fixed_image]) + ".nii.gz",
                                        "--transform",
                                    applied_warp + "1Warp.nii.gz", applied_warp + "0GenericAffine.mat",
                                        "--reference-image", "temporary_fixed.nii.gz", "--interpolation", "BSpline[3]"])

          if i > (central_level - 1):
                moving_image = i
                fixed_image = (central_level - 1)
                preprocess_file(moving_image, fixed_image,)
                print_output(applied_warp + "1Warp.nii.gz")
                print_output(applied_warp + "0GenericAffine.mat")
                # #  Register METRIC_REF(i +1) --> METRIC_REF(i)
                subprocess.call(['antsRegistration', "--dimensionality", "2", "--transform", "Affine[0.5]",
                                     "--metric", "MeanSquares[", "temporary_fixed.nii.gz,",
                                     "temporary_moving.nii.gz, 1, 5]",
                                     "--convergence", "100x100", "--shrink-factors", "8x4", "--smoothing-sigmas",
                                     "2x2vox",
                                     "--transform", "BSplineSyN[0.5,2]",
                                     "--metric", "MeanSquares[", "temporary_fixed.nii.gz,",
                                     "temporary_moving.nii.gz, 1, 4]",
                                     "--convergence", "100x100x100x100", "--shrink-factors", "8x4x2x1",
                                     "--smoothing-sigmas", "0x0x0x0vox",
                                     "--output", applied_warp,
                                     "--interpolation", "BSpline[3]"])

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
                    subprocess.call(['antsApplyTransforms', "--dimensionality", "2",
                                        "--input", "temporary_fixed.nii.gz",
                                        "--output", "Applied_warp_" + str(Metrics[m]) + "_" + str(
                            level_array_list[level_number][moving_image]) + "_on_" + str(level_array_list[level_number][fixed_image]) + ".nii.gz",
                                        "--transform",
                                        applied_warp + "1Warp.nii.gz",
                                        applied_warp + "0GenericAffine.mat",
                                        "--reference-image", "temporary_fixed.nii.gz"])

  # # Concatenate the volumes per section
    # for i in len(level):
    #
    #     Volume_3D = np.zeros((151, 151, 6))
    #     nib.save(Volume_4D, "/Users/hanam/Documents/Tracts_testing_2/all_levels/ref_4D")
