# #!/usr/bin/env python
#
# Description: Parameters for registration of tracts.


# PARAMETERS THAT ARE SPECIFIC TO THE INPUT DATA
# ======================================================================================================================

FOLDER = '/Users/julien/Desktop/AtlasRat'  # Path to folder that contains all slices and all metrics
OUTPUT_FOLDER = 'results'

input_file_prefix = [
    'AtlasRat_AD',
    'AtlasRat_AED',
    'AtlasRat_GR',
    'AtlasRat_MT',
    'AtlasRat_MVF',
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


# PARAMETERS THAT ARE USED INTERNALLY BY THE CODE
# ======================================================================================================================

file_prefix = 'AtlasRat_AllMetrics_'
