import shutil
import nibabel as nib
import os
import numpy as np
from matplotlib.pylab import *
from matplotlib import pyplot as plt
from matplotlib import colors
import params
from PIL import Image, ImageOps

ext = '.nii'

# Initialize logging
# logging.basicConfig(level=logging.DEBUG)

os.chdir(os.path.join(params.FOLDER, params.OUTPUT_FOLDER))
path_paxinos_export_folder = os.path.join(params.FOLDER, params.OUTPUT_FOLDER,'paxinos_slicewise')
path_clustering_slicewise_export_folder = os.path.join(params.FOLDER, params.OUTPUT_FOLDER,'crop_clustering_slicewise')
path_concat_within_region_export_folder = os.path.join(params.FOLDER, params.OUTPUT_FOLDER,'crop_concat_within_region')
path_folder_results_clustering = os.path.join(params.FOLDER, params.OUTPUT_FOLDER,params.OUTPUT_FOLDER_SLICEWISE)

def generate_paxinos_slicewise(path_paxinos_export_folder):
    print ('\n Running generate_paxinos_slicewise')
    if not os.path.exists(path_paxinos_export_folder):
        os.makedirs(path_paxinos_export_folder)
    # Define levels from params.
    levels = []
    for region in params.regions.keys():
        levels = levels + params.regions[region]

    for level in levels:
        nii = nib.load(params.file_prefix_all + level + ext)

        data = nii.get_fdata()
        paxinos3d = np.squeeze(data[:, :, 0, 6:13])

        fig = plt.figure(figsize=(5, 5))

        # Display Paxinos
        for i_label in range(paxinos3d.shape[2]):
            labels_rgb = np.zeros([paxinos3d.shape[0], paxinos3d.shape[1], 4])
            for ix in range(paxinos3d.shape[0]):
                for iy in range(paxinos3d.shape[1]):
                    ind_color = list(params.colors.keys())[i_label]
                    labels_rgb[ix, iy] = colors.to_rgba(params.colors[ind_color], paxinos3d[ix, iy, i_label])
            plt.imshow(np.fliplr(rot90(labels_rgb)), aspect="equal")
        plt.axis('off')
        fig.savefig(os.path.join(path_paxinos_export_folder,
                                'paxinos-full_level_{}.png'.format(level))) 
        fig.clear()
        plt.close()

    # Crop Paxinos images
    list_files = os.listdir(path_paxinos_export_folder)
    for file in list_files:
        if file.startswith('paxinos-full_level_'):
            file_path = os.path.join(path_paxinos_export_folder, file)
            img = Image.open(file_path)
            left = 145 
            top = 170
            right = left + 225
            bottom = top + 155
            img_res = img.crop((left, top, right, bottom)) 
            img_res.save(file_path)

def crop_clustering_slicewise(path_clustering_slicewise_export_folder,path_folder_results_clustering):
    print ('\n Running crop_clustering_slicewise')
    if not os.path.exists(path_clustering_slicewise_export_folder):
        os.makedirs(path_clustering_slicewise_export_folder)
    # Crop clustering slice-wise images
    list_files = os.listdir(path_folder_results_clustering)
    for file in list_files:
        if file.startswith('clustering_results_ncluster'):
            file_path_input = os.path.join(path_folder_results_clustering, file)
            file_path_output = os.path.join(path_clustering_slicewise_export_folder, file)
            img = Image.open(file_path_input)
            left = 45 
            top = 20
            right = left + 610
            bottom = top + 450
            img_res = img.crop((left, top, right, bottom)) 
            img_res.save(file_path_output)

def crop_concat_within_region(path_concat_within_region_export_folder):
    print ('\n Running crop_concat_within_region')
    if not os.path.exists(path_concat_within_region_export_folder):
        os.makedirs(path_concat_within_region_export_folder)
    # Crop concat_within_region images
    list_files = os.listdir(os.path.join(params.FOLDER, params.OUTPUT_FOLDER, params.folder_concat_region))
    print (list_files)
    for file in list_files:
        if file.startswith('clustering_results_avgz'):
            file_path_input = os.path.join(os.path.join(params.FOLDER, params.OUTPUT_FOLDER, params.folder_concat_region), file)
            file_path_output = os.path.join(path_concat_within_region_export_folder, file)
            img = Image.open(file_path_input)
            left = 15 
            top = 60
            right = left + 670
            bottom = top + 600
            img_res = img.crop((left, top, right, bottom)) 
            img_res.save(file_path_output)

generate_paxinos_slicewise(path_paxinos_export_folder)
crop_clustering_slicewise(path_clustering_slicewise_export_folder,path_folder_results_clustering)
crop_concat_within_region(path_concat_within_region_export_folder)