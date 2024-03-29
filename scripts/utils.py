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


def generate_intensity_list(list_score):
    """
    Generate intensities between 0.2 and 1, scaled based on the overlap score.
    :param list_score:
    :return: list_intensity
    """
    min_intensity = 0.2
    # Normalize between 0 and 1
    list_intensity = [min_intensity + (i - min(list_score)) * (1 - min_intensity) / (max(list_score) - min(list_score))
                      for i in list_score]
    # Deal with precision issues
    list_intensity = [np.clip(i, min_intensity, 1) for i in list_intensity]
    return list_intensity


def get_best_matching_color_with_paxinos(im=None, imref=None):
    """
    Find the color index for im that corresponds to the tract of imref with the maximum overlap
    :param im: numpy.array: X, Y, TRACT
    :param imref: numpy.array: 3D image with tract to match: X, Y, TRACT
    :return: list_color: RGB color
    :return: list_intensity: List of intensity. Same size as list_color.
    """
    list_color = []
    max_score, max_index = [], []
    # Loop across labels from clustering image, compute overlap and assign color
    for i_label in range(im.shape[2]):
        score = []
        # Compute similarity score between clustering label and each of the Paxinos label
        for i in range(imref.shape[2]):
            score.append(np.sum(np.multiply(im[..., i_label], imref[..., i])))
        max_index.append(np.argmax(score))
        max_score.append(np.max(score))
        logging.debug("Clustering label: #{} | Ref label: #{} | Color: {} | Scores: {}".
                      format(i_label, np.argmax(score), list(params.colors.keys())[np.argmax(score)], score))
        # Find the clustering labels that correspond to this Paxinos label
        # list_color --> color for clustering labels
        list_color.append(list(params.colors.keys())[np.argmax(score)])

    # Loop across labels from clustering image (im) and if the label appears more than once (ie: same color for the
    # label), assign various intensity values to be able to distinguish them.
    list_intensity = [0] * len(list_color)
    for i_label in range(imref.shape[2]):
        index_matched = list(np.where(np.array(max_index) == i_label)[0])
        # list_intensity --> intensity value for clustering labels
        if index_matched:
            values_list_intensity = []
            # Normalize the scores of the matched labels between 0.2 and 1 using the overlap score
            if len(index_matched) > 1:
                values_list_intensity = generate_intensity_list([max_score[i] for i in index_matched])
            # If there is only one score, set intensity value to 1
            else:
                values_list_intensity.append(1.0)
            for i in range(len(index_matched)):
                list_intensity[index_matched[i]] = values_list_intensity[i]
            # for index_position in index_matched:
            #     list_intensity.insert(index_position, values_list_intensity[index_matched.index(index_position)])

    logging.debug("Selected colors: {}".format(list_color))
    logging.debug("Selected intensities: {}".format(list_intensity))

    return list_color, list_intensity
