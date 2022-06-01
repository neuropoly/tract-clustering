#!/usr/bin/env python

# Utility functions for the tract-clustering project


import os
import shutil
import subprocess
import math
import numpy as np
import nibabel as nib

import params
import logging

def generate_intensity_list(n):
    start = 0.2
    end = 1
    if n < 2:
        raise Exception("behaviour not defined for n<2")
    step = (end - start) / float(n - 1)
    intensity_list = [(round((start + x * step), 2)) for x in range(n)]
    return intensity_list


def get_best_matching_color_with_paxinos(im=None, imref=None):
    """
    Find the color index for im that corresponds to the tract of imref with the maximum overlap
    :param im: numpy.array: X, Y, TRACT
    :param imref: numpy.array: 3D image with tract to match: X, Y, TRACT
    :return: list: RGB color
    """
    sorted_score = []
    list_color = []
    max_score, max_index = [], []
    list_intensity = []
    # Loop across labels from clustering image, compute overlap and assign color
    for i_label in range(im.shape[2]):
        score = []
        # Compute similarity score between clustering label and each of the Paxinos label
        for i in range(imref.shape[2]):
            score.append(np.sum(np.multiply(im[..., i_label], imref[..., i])))
        max_index.append(np.argmax(score))
        max_score.append(np.max(score))
        logging.debug("Clustering label: #{} | Ref label: #{} | Color: {} | Scores: {}".format(i_label, np.argmax(score), list(params.colors.keys())[np.argmax(score)], score))
        # Find the clustering labels that correspond to this Paxinos label
        # list_color --> color for clustering labels
        list_color.append(list(params.colors.keys())[np.argmax(score)])

    # Loop across labels from clustering image and assign intensity based on computed overlap score
    for i_label in range(im.shape[2]):
        index_matched = list(np.where(np.array(max_index) == i_label)[0])
        # list_intensity --> intensity value for clustering labels
        if index_matched:
            values_list_intensity = []
            # Normalize the scores of the matched labels between 0.2 and 1
            if len(index_matched) > 1:
                values_list_intensity = generate_intensity_list(len(index_matched))
            # If there is only one score, set intensity value to 1
            else:
                values_list_intensity.append(1.0)
            for index_position in index_matched:
                list_intensity.insert(index_position,values_list_intensity[index_matched.index(index_position)])

    logging.debug("Selected colors: {}".format(list_color))
    logging.debug("Selected intensities: {}".format(list_intensity))

    return list_color, list_intensity
