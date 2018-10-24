# #!/usr/bin/env python
# 
# This script loops across slices and registers slice i-1 on slice i and 
# slice i+1 on slice i. Then, it applies the transformations on the neighbor 
# slices and outputs a 3d volume (x, y, 3) for each metric.

# Parameters
# FOLDER: folder that contains all slices and all metrics
# METRICS: list of metrics to register, e.g. [avf.nii.gz, ad.nii.gz]
# METRIC_REF: file name of metric to use as reference for registration

# Read FOLDER that contains all slices and all metrics

# Estimate transfo
# Loop across slices
#   read METRIC_REF i-1, i and i+1
#   register METRIC_REF(i-1) --> METRIC_REF(i)
#     # outputs warp(i-1->1)
#   register METRIC_REF(i+1) --> METRIC_REF(i)
#     # outputs warp(i+1->1)
# 
# Apply transfo across metrics
# Loop across metrics (m)
#   Loop across slices (i)
#     apply warp(i-1->1) on METRICS[m]
#       output: METRICS[m, i-1]_r
#     apply warp(i+1->1) on METRICS[m]
#       output: METRICS[m, i+1]_r
#     concatenate: METRICS[m, i-1]_r, METRICS[m, i]_r, METRICS[m, i+1]_r
#       output: METRICS3D[m, i] --> 3D (x, y, 3) metric

