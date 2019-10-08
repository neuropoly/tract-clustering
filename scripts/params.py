# #!/usr/bin/env python
#
# Description: Parameters for registration of tracts.


FOLDER = '/Users/julien/Desktop/AtlasRat'  # Path to folder that contains all slices and all metrics
OUTPUT_FOLDER = 'results'

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
