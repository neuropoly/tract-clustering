#!/usr/bin/env python

# Utility functions for the tract-clustering project


import os
import shutil
import subprocess
import math
import numpy as np
import nibabel as nib

import params


def get_best_matching_color_with_paxinos(im=None, imref=None):
    """
    Find the color index for im that corresponds to the tract of imref with the maximum overlap
    :param im: 3D image: X, Y, TRACT
    :param imref: 3D image with tract to match: X, Y, TRACT
    :return: list: RGB color
    """
    sorted_score = []
    list_color = []
    # Match colors with reference image
    for i_label in range(imref.shape[2]):
        # compute MI between a given tract from the reference image and all tracts from the input image
        score = [np.sum(np.multiply(im[..., i], imref[..., i_label])) for i in range(im.shape[2])]
        #
        # mi_score = [mutual_info_score(im[..., i].reshape(np.multiply(im.shape[0], im.shape[1])),
        #                               imref[..., i_label].reshape(np.multiply(im.shape[0], im.shape[1])))
        #             for i in range(im.shape[2])
        #             ]
        logger.debug("Ref label #{}: Mutual information: {}".format(i_label, score))
        sorted_score.append(np.argmax(score))
        # list_color.append(list(params.colors.keys())[np.argmax(score)])

    # Fill with remainind colors
    for i in range(8, 8+im.shape[2]-imref.shape[2]):
        list_color.append(list(params.colors.keys())[i])

    logger.debug("Selected colors: {}".format(list_color))

    # debugging
    for i in range(8):
        matshow(imref[..., i], fignum=i+1, cmap=cm.gray), plt.colorbar(), show()

    return list_color