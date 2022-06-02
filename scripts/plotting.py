#!/usr/bin/env python

# Plotting functions

import logging
from matplotlib import pyplot as plt
from matplotlib import colors
import numpy as np
import os

from utils import get_best_matching_color_with_paxinos
import params


def plot_clustering(labels3d, paxinos3d, n_cluster, level, path_output):
    """
    Plot clustering results with the Paxinos/Watson atlas.
    :param labels3d:
    :param paxinos3d:
    :param n_cluster:
    :param level:
    :param path_output:
    :return:
    """
    logging.info("Generate figure...")
    fig = plt.figure(figsize=(6.5, 5))

    # Display Paxinos
    # TODO: generalize BASE_COLORS for more than 8 labels
    ax = fig.add_subplot(1, 2, 2)
    ax.set_facecolor((1, 1, 1))
    for i_label in range(paxinos3d.shape[2]):
        labels_rgb = np.zeros([paxinos3d.shape[0], paxinos3d.shape[1], 4])
        for ix in range(paxinos3d.shape[0]):
            for iy in range(paxinos3d.shape[1]):
                ind_color = list(params.colors.keys())[i_label]
                labels_rgb[ix, iy] = colors.to_rgba(params.colors[ind_color], paxinos3d[ix, iy, i_label])
        ax.imshow(np.fliplr(np.rot90(labels_rgb)), aspect="equal")
    plt.axis('off')
    # Find label color corresponding best to the Paxinos atlas
    list_color, list_intensity = get_best_matching_color_with_paxinos(im=labels3d, imref=paxinos3d)

    # Display clustering
    ax2 = fig.add_subplot(1, 2, 1)
    for i_label in range(n_cluster):
        labels_rgb = np.zeros([labels3d.shape[0], labels3d.shape[1], 4])
        for ix in range(labels3d.shape[0]):
            for iy in range(labels3d.shape[1]):
                logging.debug(f"level: {level}, i_label: {i_label}, ix: {ix}, iy: {iy}")
                labels_rgb[ix, iy] = colors.to_rgba(params.colors[list_color[i_label]], labels3d[ix, iy, i_label] * (
                    list_intensity[i_label]))
        ax2.imshow(np.rot90(labels_rgb), aspect="equal")
    plt.axis('off')
    plt.tight_layout()
    fig.subplots_adjust(hspace=0, wspace=0.01)
    fig.savefig(os.path.join(path_output, 'clustering_results_ncluster{}_{}.png'.format(n_cluster, level)),
                transparent=True)
    fig.clear()
    plt.close()
