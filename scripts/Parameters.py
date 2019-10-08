#--------------------------------------------------------------------------
# Name: Parameters.py
# Date: 2019-10-02
# Description: Parameters for registration of tracts.
#
#--------------------------------------------------------------------------


Folder= "/Users/hanam/Desktop/AtlasRat"
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

Metrics = [
    "Axon_Density",
    "axon_equiv_diameter",
    # "AVF_corrected",
    "GR_corrected",
    "Myelin_thickness",
    "MVF_corrected"
]

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
