#!/usr/bin/env python
#
# This program takes as input an anatomic image and the centerline or segmentation of its spinal cord (that you can get
# using sct_get_centerline.py or sct_segmentation_propagation) and returns the anatomic image where the spinal
# cord was straightened.
#
# ---------------------------------------------------------------------------------------
# Copyright (c) 2013 NeuroPoly, Polytechnique Montreal <www.neuro.polymtl.ca>
# Authors: Julien Cohen-Adad, Geoffrey Leveque, Julien Touati
# Modified: 2014-09-01
#
# License: see the LICENSE.TXT
# ======================================================================================================================
# check if needed Python libraries are already installed or not

import os
import time
import sys
from msct_parser import Parser
import sct_utils as sct
import numpy as np
from msct_image import Image
from sortedcontainers import SortedListWithKey
from operator import itemgetter

try:
    import cPickle as pickle
except:
    import pickle

# import keras necessary classes
import theano
theano.config.floatX = 'float32'

from keras.models import load_model


def extract_patches_from_image(image_file, patches_coordinates, patch_size=32, slice_of_interest=None, verbose=1):
    result = []
    for k in range(len(patches_coordinates)):
        if slice_of_interest is None:
            ind = [patches_coordinates[k][0], patches_coordinates[k][1], patches_coordinates[k][2]]
        else:
            ind = [patches_coordinates[k][0], patches_coordinates[k][1], slice_of_interest]

        # Transform voxel coordinates to physical coordinates to deal with different resolutions
        # 1. transform ind to physical coordinates
        ind_phys = image_file.transfo_pix2phys([ind])[0]
        # 2. create grid around ind  - , ind_phys[2]
        grid_physical = np.mgrid[ind_phys[0] - patch_size / 2:ind_phys[0] + patch_size / 2, ind_phys[1] - patch_size / 2:ind_phys[1] + patch_size / 2]
        # 3. transform grid to voxel coordinates
        coord_x = grid_physical[0, :, :].ravel()
        coord_y = grid_physical[1, :, :].ravel()
        coord_physical = [[coord_x[i], coord_y[i], ind_phys[2]] for i in range(len(coord_x))]
        grid_voxel = np.array(image_file.transfo_phys2continuouspix(coord_physical))
        np.set_printoptions(threshold=np.inf)
        # 4. interpolate image on the grid, deal with edges
        patch = np.reshape(image_file.get_values(np.array([grid_voxel[:, 0], grid_voxel[:, 1], grid_voxel[:, 2]]), interpolation_mode=1), (patch_size, patch_size))

        if verbose == 2:
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots()
            cax = ax.imshow(patch, cmap='gray')
            cbar = fig.colorbar(cax, ticks=[0, 255])
            plt.show()

        if patch.shape[0] == patch_size and patch.shape[1] == patch_size:
            result.append(np.expand_dims(patch, axis=0))
    if len(result) != 0:
        return np.concatenate(result, axis=0)
    else:
        return None


def display_patches(patches, nb_of_subplots=None):
    import matplotlib.pyplot as plt

    if nb_of_subplots is None:
        nb_of_subplots = [4, 4]  # [X, Y]

    nb_of_subplots_per_fig = nb_of_subplots[0] * nb_of_subplots[1]

    nb_of_patches = patches.shape[0]
    if nb_of_patches == 0:
        return

    nb_of_figures_to_display = int(nb_of_patches / nb_of_subplots_per_fig)
    if nb_of_figures_to_display == 0:
        nb_of_figures_to_display = 1
    elif nb_of_patches % nb_of_figures_to_display != 0:
        nb_of_figures_to_display += 1

    for i in range(nb_of_figures_to_display):
        fig = plt.figure('Patches #' + str(i * nb_of_subplots_per_fig) + ' to #' + str((i + 1) * nb_of_subplots_per_fig - 1))
        patches_to_display = patches[i*nb_of_subplots_per_fig:(i+1)*nb_of_subplots_per_fig, :, :]

        for j in range(patches_to_display.shape[0]):
            ax = plt.subplot(nb_of_subplots[0], nb_of_subplots[1], j+1)
            ax.imshow(patches_to_display[j, :, :], vmin=0, vmax=255, cmap='gray')
            ax.set_axis_off()
            ax.set_aspect('equal')

        plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1)
        plt.show()


def predict(fname_input, fname_model, initial_resolution, fname_output=''):
    time_prediction = time.time()

    patch_size = 32
    model = load_model(fname_model)
    im_data = Image(fname_input)

    # intensity normalization
    im_data.data = 255.0 * (im_data.data - np.percentile(im_data.data, 0)) / np.abs(np.percentile(im_data.data, 0) - np.percentile(im_data.data, 100))

    nx, ny, nz, nt, px, py, pz, pt = im_data.dim

    # first round of patch prediction
    initial_coordinates_x = range(patch_size/2, nx - patch_size/2, initial_resolution[0])
    initial_coordinates_y = range(patch_size/2, ny - patch_size/2, initial_resolution[1])
    X, Y = np.meshgrid(initial_coordinates_x, initial_coordinates_y)
    X, Y = X.ravel(), Y.ravel()
    initial_coordinates = [[X[i], Y[i]] for i in range(len(X))]

    nb_voxels_explored = 0
    coord_positive = SortedListWithKey(key=itemgetter(0, 1, 2))
    coord_positive_saved = []

    print 'Initial prediction'
    for slice_number in range(0, nz, initial_resolution[2]):
        patches = extract_patches_from_image(im_data, initial_coordinates, patch_size=patch_size, slice_of_interest=slice_number, verbose=0)
        patches = np.asarray(patches, dtype=int)
        patches = patches.reshape(patches.shape[0], 1, patches.shape[1], patches.shape[2])
        y_pred = model.predict(patches, batch_size=32, verbose=0)
        classes_predictions = np.where(y_pred[:, 1] > 0.5)[0].tolist()

        """
        patches_positive = np.squeeze(patches[np.where(y_pred == 1)[0]], axis=(1,))
        display_patches(patches_positive, nb_of_subplots=[4, 4])
        """

        coord_positive.update([[initial_coordinates[coord][0], initial_coordinates[coord][1], slice_number] for coord in classes_predictions])
        coord_positive_saved.extend([[initial_coordinates[coord][0], initial_coordinates[coord][1], slice_number, y_pred[coord, 1]] for coord in classes_predictions])
        nb_voxels_explored += len(patches)
    last_coord_positive = coord_positive

    # data preparation
    list_offset = [[xv, yv, zv] for xv in [-1, 0, 1] for yv in [-1, 0, 1] for zv in [-3, -2, -1, 0, 1, 2, 3] if [xv, yv, zv] != [0, 0, 0]]

    # informative data
    iteration = 1

    print '\nIterative prediction based on mathematical morphology'
    while len(last_coord_positive) != 0:
        print '\nIteration #' + str(iteration) + '. Number of positive voxel to explore:' + str(len(last_coord_positive))
        current_coordinates = SortedListWithKey(key=itemgetter(0, 1, 2))
        for coord in last_coord_positive:
            for offset in list_offset:
                new_coord = [coord[0] + offset[0], coord[1] + offset[1], coord[2] + offset[2]]
                if 0 <= new_coord[0] < im_data.data.shape[0] and 0 <= new_coord[1] < im_data.data.shape[1] and 0 <= new_coord[2] < im_data.data.shape[2]:
                    if current_coordinates.count(new_coord) == 0:
                        if coord_positive.count(new_coord) == 0:
                            current_coordinates.add(new_coord)

        print 'Patch extraction (N=' + str(len(current_coordinates)) + ')'
        patches = extract_patches_from_image(im_data, current_coordinates, patch_size=patch_size, verbose=0)
        #display_patches(patches, nb_of_subplots=[10, 10])
        if patches is not None:
            patches = np.asarray(patches, dtype=int)
            patches = patches.reshape(patches.shape[0], 1, patches.shape[1], patches.shape[2])
            y_pred = model.predict(patches, batch_size=32, verbose=1)
            classes_predictions = np.where(y_pred[:, 1] > 0.5)[0].tolist()

            """
            patches_positive = np.squeeze(patches[np.where(y_pred == 1)[0]], axis=(1,))
            display_patches(patches_positive, nb_of_subplots=[10, 10])
            """

            last_coord_positive = [[current_coordinates[coord][0], current_coordinates[coord][1], current_coordinates[coord][2]] for coord in classes_predictions]
            coord_positive.update(last_coord_positive)
            coord_positive_saved.extend([[current_coordinates[coord][0], current_coordinates[coord][1], current_coordinates[coord][2], y_pred[coord, 1]] for coord in classes_predictions])
            nb_voxels_explored += len(patches)
        else:
            last_coord_positive = []

        iteration += 1

    nb_voxels_image = nx * ny * nz
    print '\nNumber of voxels explored = ' + str(nb_voxels_explored) + '/' + str(nb_voxels_image) + ' (' + str(round(100.0 * nb_voxels_explored / nb_voxels_image, 2)) + '%)'

    # write results
    print '\nWriting results'
    input_image = im_data.copy()
    input_image.data *= 0
    for coord in coord_positive_saved:
        print coord
        input_image.data[coord[0], coord[1], coord[2]] = coord[3]
    path_input, file_input, ext_input = sct.extract_fname(fname_input)
    fname_output_temp = path_input + "tmp." + file_input + "_cord_prediction" + ext_input
    input_image.setFileName(fname_output_temp)
    input_image.save()

    from scipy.ndimage import binary_opening, binary_closing, label
    input_image.data = binary_closing(input_image.data, structure=np.ones((2, 2, 1))).astype(np.int)
    input_image.data = binary_opening(input_image.data, structure=np.ones((2, 2, 1))).astype(np.int)
    blobs, number_of_blobs = label(input_image.data)
    size = np.bincount(blobs.ravel())
    biggest_label = size[1:].argmax() + 1
    input_image.data *= 0
    input_image.data[blobs == biggest_label] = 1

    from sct_straighten_spinalcord import smooth_centerline
    x_centerline_fit, y_centerline_fit, z_centerline, x_centerline_deriv, y_centerline_deriv, z_centerline_deriv = smooth_centerline(input_image, algo_fitting='nurbs', verbose=1, nurbs_pts_number=3000, all_slices=False, phys_coordinates=False, remove_outliers=False)

    input_image.data *= 0
    for i in range(0, len(z_centerline), 1):
        input_image.data[int(round(x_centerline_fit[i])), int(round(y_centerline_fit[i])), int(round(z_centerline[i]))] = 1

    if fname_output == '':
        path_input, file_input, ext_input = sct.extract_fname(fname_input)
        fname_output = file_input + "_cord_prediction" + ext_input
    input_image.setFileName(fname_output)
    input_image.save()

    time_prediction = time.time() - time_prediction
    sct.printv('Time to predict cord location: ' + str(np.round(time_prediction)) + ' seconds', 1)


def get_parser():
    # Initialize parser
    parser = Parser(__file__)

    # Mandatory arguments
    parser.usage.set_description("This program takes as input an anatomic image and outputs a binary image with the spinal cord centerline/segmentation.")
    parser.add_option(name="-i",
                      type_value="image_nifti",
                      description="input image.",
                      mandatory=True,
                      example="t2.nii.gz")

    parser.add_option(name="-c",
                      type_value="multiple_choice",
                      description="type of image contrast, t2: cord dark / CSF bright ; t1: cord bright / CSF dark.",
                      mandatory=False,
                      example=['t1', 't2'])

    parser.add_option(name="-o",
                      type_value="file_output",
                      description="centerline/segmentation output filename.",
                      mandatory=False,
                      default_value='',
                      example="data_straight.nii.gz")

    parser.add_option(name="-r",
                      type_value="multiple_choice",
                      description="remove temporary files.",
                      mandatory=False,
                      example=['0', '1'],
                      default_value='1')

    parser.add_option(name="-v",
                      type_value="multiple_choice",
                      description="Verbose. 0: nothing, 1: basic, 2: extended.",
                      mandatory=False,
                      example=['0', '1', '2'],
                      default_value='1')

    parser.add_option(name='-qc',
                      type_value='multiple_choice',
                      description='Output images for quality control.',
                      mandatory=False,
                      example=['0', '1'],
                      default_value='0')

    return parser


if __name__ == "__main__":
    parser = get_parser()
    arguments = parser.parse(sys.argv[1:])

    #fname_model = '/Users/benjamindeleener/data/machine_learning/results_detection/2016-08-16_nopad_HCandCSM/model_cnn_it240000.h5'
    #fname_model = '/Users/benjamindeleener/data/machine_learning/results_detection/2016-08-17_HCandCSM_4conv/model_cnn_it530000.h5'
    fname_model = '/Users/benjamindeleener/data/machine_learning/results_detection/2016-09-19_large_dataset/experiment2/model_cnn_it240000.h5'
    #fname_model = '/Users/benjamindeleener/data/machine_learning/results_detection/2016-09-19_large_dataset/experiment2/model_cnn_it1260000.h5'
    initial_resolution = [5, 5, 10]

    fname_input = arguments['-i']

    fname_output = ''
    if '-o' in arguments:
        fname_output = arguments['-o']

    predict(fname_input, fname_model, initial_resolution, fname_output=fname_output)
