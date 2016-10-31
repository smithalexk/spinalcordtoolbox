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

# # import keras necessary classes
# import theano
# theano.config.floatX = 'float32'

# from keras.models import load_model

from sklearn.externals import joblib
from skimage.feature import hog
from sklearn.svm import SVC
from sklearn.base import BaseEstimator
import matplotlib.pyplot as plt
from scipy.sparse.csgraph import shortest_path
from sct_straighten_spinalcord import smooth_centerline
from scipy.spatial import distance
from math import sqrt
import bz2
from scipy import ndimage

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
        patch = np.reshape(image_file.get_values(np.array([grid_voxel[:, 0], grid_voxel[:, 1], grid_voxel[:, 2]]), interpolation_mode=1, boundaries_mode='reflect'), (patch_size, patch_size))

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


def display_patches(patches, nb_of_subplots=None, nb_of_figures_to_display=0):

    if nb_of_subplots is None:
        nb_of_subplots = [4, 4]  # [X, Y]

    nb_of_subplots_per_fig = nb_of_subplots[0] * nb_of_subplots[1]

    nb_of_patches = patches.shape[0]
    if nb_of_patches == 0:
        return

    nb_of_figures_to_display_max = int(nb_of_patches / nb_of_subplots_per_fig)
    if nb_of_figures_to_display <= 0 or nb_of_figures_to_display > nb_of_figures_to_display_max:
        if nb_of_figures_to_display_max == 0:
            nb_of_figures_to_display = 1
        elif nb_of_patches % nb_of_figures_to_display_max != 0:
            nb_of_figures_to_display = nb_of_figures_to_display_max + 1

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

class Classifier_svm(BaseEstimator):
    def __init__(self, params={}):

        self.clf = SVC()
        self.params = params
 
    def train(self, X, y):
        self.clf.fit(X, y)
 
    def predict(self, X):
        return self.clf.predict_proba(X)

    def save(self, fname_out):
        joblib.dump(self.clf, fname_out + '.pkl')

    def load(self, fname_in):
        clf = joblib.load(fname_in + '.pkl')

        self.clf = clf

        self.params = clf.get_params()
        print self.clf.get_params()

    def set_params(self, params):
        self.clf.set_params(**params)
        print params
        self.params = params

def extract_hog_feature(patch_list, param=None):

    if param is None:
        param = {'orientations': 8, 'pixels_per_cell': [6, 6], 'cells_per_block': [3,3],
                'visualize': False, 'transform_sqrt': True}

    X_test = []
    for patch in patch_list:
        hog_feature = np.array(hog(image = patch, orientations=param['orientations'],
                pixels_per_cell=param['pixels_per_cell'], cells_per_block=param['cells_per_block'],
                transform_sqrt=param['transform_sqrt'], visualise=param['visualize']))
        X_test.append(hog_feature)

    X_test = np.array(X_test)

    return X_test

def extract_patch_feature(im):

    return im


def plot_2D_detections(im_data, slice_number, initial_coordinates, classes_predictions):

    slice_cur = im_data.data[:,:,slice_number]

    fig1 = plt.figure()
    ax1 = plt.subplot(1, 1, 1)
    ax1.imshow(slice_cur,cmap=plt.get_cmap('gray'))

    for coord_mesh in range(len(initial_coordinates)):
        plt.scatter(initial_coordinates[coord_mesh][1], initial_coordinates[coord_mesh][0], c='Red')

    for coord in classes_predictions:
        plt.scatter(initial_coordinates[coord][1], initial_coordinates[coord][0], c='Green')
    
    ax1.set_axis_off()
    plt.show()


def predict_along_centerline(im_data, model, patch_size, coord_positive_saved, feature_fct, threshold, idx_2_process, list_offset, path_output, verbose=0):

    coord_positive_saved = [tuple([int(ii) for ii in sublist]) for sublist in coord_positive_saved 
                                                                    if sublist[0] < im_data.data.shape[0]
                                                                        and sublist[1] < im_data.data.shape[1]
                                                                        and sublist[2] < im_data.data.shape[2]]
    coord_positive_saved
    coord_positive = SortedListWithKey(key=itemgetter(0, 1, 2))
    coord_positive.update(coord_positive_saved)
    last_coord_positive = coord_positive

    # informative data
    iteration = 1

    print '\n... Iterative prediction based on mathematical morphology'
    while len(last_coord_positive) != 0:

        print '\n... ... Iteration #' + str(iteration) + '. # of positive voxel to explore: ' + str(len(last_coord_positive))
        current_coordinates = SortedListWithKey(key=itemgetter(0, 1, 2))
        for coord in last_coord_positive:
            for offset in list_offset:
                new_coord = [coord[0] + offset[0], coord[1] + offset[1], coord[2] + offset[2]]
                if new_coord[2] in idx_2_process:
                    if 0 <= new_coord[0] < im_data.data.shape[0] and 0 <= new_coord[1] < im_data.data.shape[1] and 0 <= new_coord[2] < im_data.data.shape[2]:
                        if current_coordinates.count(new_coord) == 0:
                            if coord_positive.count(new_coord) == 0:
                                current_coordinates.add(new_coord)

        print '... ... Patch extraction (N=' + str(len(current_coordinates)) + ')'

        patches = extract_patches_from_image(im_data, current_coordinates, patch_size=patch_size, verbose=0)
        
        if patches is not None:
            patches = np.asarray(patches, dtype=int)
            patches = patches.reshape(patches.shape[0], patches.shape[1], patches.shape[2])

            X_test = feature_fct(patches)
            y_pred = model.predict(X_test)

            y_pred = np.array(y_pred)
            classes_predictions = np.where(y_pred[:, 1] > threshold)[0].tolist()
            print '... ... ' + str(len(classes_predictions)) + ' patches detected as Pos'

            if iteration % 2 == 0 and verbose != 0:
                patches_positive = np.squeeze(patches[np.where(y_pred[:, 1] > threshold)[0]])
                display_patches(patches_positive, nb_of_subplots=[10, 10], nb_of_figures_to_display=1)

            last_coord_positive = [[current_coordinates[coord][0], current_coordinates[coord][1], current_coordinates[coord][2]] for coord in classes_predictions]
            coord_positive.update(last_coord_positive)
            coord_positive_saved.extend([[current_coordinates[coord][0], current_coordinates[coord][1], current_coordinates[coord][2], 1] for coord in classes_predictions])
        else:
            last_coord_positive = []

        z_pos_pred = np.array([coord[2] for coord in coord_positive_saved])
        number_no_detection = im_data.dim[2] - len(np.unique(z_pos_pred))
        if number_no_detection == 0:
            last_coord_positive = []

        iteration += 1

    # write results
    input_image = im_data.copy()
    input_image.data *= 0
    for coord in coord_positive_saved:
        input_image.data[coord[0], coord[1], coord[2]] = 1
    # path_input, file_input, ext_input = sct.extract_fname(fname_input)
    # fname_seg_tmp = path_output + file_input + "_cord_prediction_seg_tmp" + ext_input
    fname_seg_tmp = path_output + "_seg_tmp.nii.gz"
    input_image.setFileName(fname_seg_tmp)
    input_image.save()

    z_pos_pred = np.array([coord[2] for coord in coord_positive_saved])
    number_no_detection = im_data.dim[2] - len(np.unique(z_pos_pred))

    return coord_positive_saved, fname_seg_tmp, number_no_detection

def prediction_init(im_data, model, initial_resolution, list_offset, threshold, feature_fct, patch_size, path_output, verbose=0):

    time_prediction = time.time()

    nx, ny, nz, nt, px, py, pz, pt = im_data.dim
    print 'Image Dimensions: ' + str([nx, ny, nz]) + '\n'
    nb_voxels_image = nx * ny * nz

    # first round of patch prediction
    initial_coordinates_x = range(0, nx, initial_resolution[0])
    initial_coordinates_y = range(0, ny, initial_resolution[1])
    X, Y = np.meshgrid(initial_coordinates_x, initial_coordinates_y)
    X, Y = X.ravel(), Y.ravel()
    initial_coordinates = [[X[i], Y[i]] for i in range(len(X))]

    nb_voxels_explored = 0
    coord_positive = SortedListWithKey(key=itemgetter(0, 1, 2))
    coord_positive_saved = []

    print '... Initial prediction:\n'
    tot_pos_pred = 0
    for slice_number in range(0, nz, initial_resolution[2]):
        print '... ... slice #' + str(slice_number) + '/' + str(nz)
        patches = extract_patches_from_image(im_data, initial_coordinates, patch_size=patch_size, slice_of_interest=slice_number, verbose=0)
        patches = np.asarray(patches, dtype=int)
        patches = patches.reshape(patches.shape[0], patches.shape[1], patches.shape[2])

        X_test = feature_fct(patches)
        y_pred = model.predict(X_test)

        y_pred = np.array(y_pred)
        classes_predictions = np.where(y_pred[:, 1] > threshold)[0].tolist()
        print '... ... # of pos prediction: ' + str(len(classes_predictions)) + '\n'
        tot_pos_pred += len(classes_predictions)

        if slice_number % 4 == 0 and verbose > 0:
            plot_2D_detections(im_data, slice_number, initial_coordinates, classes_predictions)
        
        coord_positive.update([[initial_coordinates[coord][0], initial_coordinates[coord][1], slice_number] for coord in classes_predictions])
        coord_positive_saved.extend([[initial_coordinates[coord][0], initial_coordinates[coord][1], slice_number, y_pred[coord, 1]] for coord in classes_predictions])
        
        nb_voxels_explored += len(patches)
    
    print '\n... # of voxels explored = ' + str(nb_voxels_explored) + '/' + str(nb_voxels_image) + ' (' + str(round(100.0 * nb_voxels_explored / nb_voxels_image, 2)) + '%)'
    last_coord_positive = coord_positive

    coord_positive_saved, fname_seg_tmp, number_no_detection = predict_along_centerline(im_data, model, patch_size, 
                                                                                            coord_positive_saved, feature_fct, 
                                                                                            threshold, range(0, nz), list_offset, 
                                                                                            path_output, verbose)

    print '\n... Time to predict cord location: ' + str(np.round(time.time() - time_prediction)) + ' seconds'
    print '... # of slice without pos: ' + str(number_no_detection) + '/' + str(nz) + ' (' + str(round(float(100.0 * (number_no_detection)) / nz, 2)) + '%)\n'

    return fname_seg_tmp

def shortest_path_graph(fname_seg, path_output):

    # TODO:     Nettoyer code

    im = Image(fname_seg)
    im_data = im.data
    im_data[im_data>0.0] = 1.0

    image_graph = {}
    cmpt = 0
    for z in range(im.dim[2]):
        labeled, nb_label = ndimage.label(im_data[:,:,z] == 1.0)
        if nb_label > 0:
            cmpt_obj = 0
            coordi = []
            while cmpt_obj < 3 and cmpt_obj < nb_label:
                size = np.bincount(labeled.ravel())
                biggest_label = size[1:].argmax() + 1
                clump_mask = labeled == biggest_label
                coordi.append(tuple([int(c) for c in ndimage.measurements.center_of_mass(clump_mask)]))
                labeled[labeled == biggest_label] = 0
                cmpt_obj += 1
            image_graph[z] = {'coords': coordi}
        else:
            image_graph[z] = {'coords': []}

    last_bound = -1
    graph_bound = []
    cmpt = 0
    for z in range(im.dim[2]):
        if len(image_graph[z]['coords']) == 0:
            if last_bound != z-1:
                graph_bound.append(range(last_bound+1,z))
            last_bound = z
            cmpt += 1

    if last_bound+1 != im.dim[2]:
        graph_bound.append(range(last_bound+1,im.dim[2]))

    idx_slice_not_detected = list(set(range(im.dim[2]))-set([xx for sublist in graph_bound for xx in sublist]))

    def centeroidnp(arr):
        length = arr.shape[0]
        sum_x = np.sum(arr[:, 0])
        sum_y = np.sum(arr[:, 1])
        return sum_x/length, sum_y/length

    node_list = []
    cmpt_bloc = 0
    centerline_coord = {}
    for bloc in graph_bound:
        cmpt_bloc += 1

        bloc_matrix = []
        for i in bloc[:-1]:
            bloc_matrix.append(distance.cdist(image_graph[i]['coords'], image_graph[i+1]['coords'], 'euclidean'))

        len_bloc = [len(image_graph[z]['coords']) for z in bloc]
        tot_candidates_bloc = sum(len_bloc)

        matrix_list = []
        lenght_done = 0
        for i in range(len(bloc[:-1])):
            matrix_i = np.ones((len_bloc[i], tot_candidates_bloc))*1000.0
            matrix_i[:,lenght_done + len_bloc[i] : lenght_done + len_bloc[i] + len_bloc[i+1]] = bloc_matrix[i]
            lenght_done += len_bloc[i]
            matrix_list.append(matrix_i)

        matrix_list.append(np.ones((len_bloc[len(bloc)-1], tot_candidates_bloc))*1000.0)

        matrix = np.concatenate(matrix_list)
        matrix[range(matrix.shape[0]), range(matrix.shape[1])] = 0

        matrix_dijkstra = shortest_path(matrix)
        cOm_0 = centeroidnp(np.array([coord for coord in  image_graph[bloc[0]]['coords']]))
        list_cOm_candidates = distance.cdist(image_graph[bloc[0]]['coords'], [cOm_0 for i in range(len(image_graph[bloc[0]]['coords']))], 'euclidean').tolist()
        list_cOm_candidates = [l[0] for l in list_cOm_candidates]
        src = list_cOm_candidates.index(min(list_cOm_candidates))
        
        node_list_cur = []
        node_list_cur.append(src)
        lenght_done = 0
        for i in range(len(bloc[:-1])):
            test = matrix_dijkstra[lenght_done + src,lenght_done + len_bloc[i] : lenght_done + len_bloc[i] + len_bloc[i+1]]

            src = test.argmin()
            node_list_cur.append(src)
            lenght_done += len_bloc[i]
        node_list.append(node_list_cur)

        cmpt = 0
        for i in bloc:
            centerline_coord[i] = [image_graph[i]['coords'][node_list_cur[cmpt]][0], image_graph[i]['coords'][node_list_cur[cmpt]][1], i]
            cmpt += 1
    
    # write results
    print '\n... ... Writing results'
    input_image = im.copy()
    input_image.data *= 0
    for z in range(im.dim[2]):
        if z in centerline_coord:
            input_image.data[centerline_coord[z][0], centerline_coord[z][1], z] = 1
    fname_output_tmp = path_output + "_reg_tmp.nii.gz"
    input_image.setFileName(fname_output_tmp)
    input_image.save()

    return idx_slice_not_detected, fname_output_tmp

def post_processing(im_data, fname_seg_tmp, model, offset, max_iter, threshold, feature_fct, patch_size, path_output, fname_output, verbose=0):
    
    time_prediction = time.time()

    nx, ny, nz, nt, px, py, pz, pt = im_data.dim
    print 'Image Dimensions: ' + str([nx, ny, nz]) + '\n'
    nb_voxels_image = nx * ny * nz

    im_seg = Image(fname_seg_tmp)
    im_seg_data = im_seg.data
    sum_seg = [np.sum(im_seg_data[:,:,z]) for z in range(im_seg.dim[2])]
    number_no_detection = sum(1 for cmpt in sum_seg if cmpt == 0)
    iter_cmpt = 1
    while max_iter > iter_cmpt:

        print '\n... Iteration #' + str(iter_cmpt)
        print '... Dijkstra regularization'
        idx2process, fname_centerline_tmp = shortest_path_graph(fname_seg_tmp, path_output)

        print '... NURBS interpolation'
        x_centerline, y_centerline, z_centerline = smooth_centerline(fname_centerline_tmp, algo_fitting='nurbs', nurbs_pts_number=3000)[:3]

        print '... # of slices without pos before prediction #' + str(iter_cmpt) + ': ' + str(len(idx2process)) + '/' + str(nz) + ' (' + str(round(float(100.0 * len(idx2process)) / nz, 2)) + '%)\n'

        coord_positive_saved = zip(x_centerline, y_centerline, z_centerline)
        list_offset = [[xv, yv, zv] for xv in range(-offset[0]-iter_cmpt,offset[0]+iter_cmpt) 
                                    for yv in range(-offset[1]-iter_cmpt,offset[1]+iter_cmpt) 
                                    for zv in range(-offset[2]-2*iter_cmpt,offset[2]+2*iter_cmpt) if [xv, yv, zv] != [0, 0, 0]]

        print '... Classifier prediction'
        coord_positive_saved, fname_seg_tmp, number_no_detection = predict_along_centerline(im_data, model, patch_size, coord_positive_saved, feature_fct, threshold, idx2process, list_offset, path_output, verbose)

        print '... # of slices without pos after prediction #' + str(iter_cmpt) + ': ' + str(number_no_detection) + '/' + str(nz) + ' (' + str(round(float(100.0 * (number_no_detection)) / nz, 2)) + '%)\n'

        iter_cmpt += 1
        if not number_no_detection:
            iter_cmpt == max_iter

    # write results
    print '\nWriting results'
    input_image = im_data.copy()
    input_image.data *= 0
    for coord in coord_positive_saved:
        input_image.data[coord[0], coord[1], coord[2]] = 1
    # path_input, file_input, ext_input = sct.extract_fname(fname_input)
    fname_output = fname_output
    input_image.setFileName(fname_output)
    input_image.save()


def compute_error(fname_centerline, fname_gold_standard):

    im_pred = Image(fname_centerline)
    im_true = Image(fname_gold_standard)

    nx, ny, nz, nt, px, py, pz, pt = im_true.dim

    count_slice, count_no_detection = 0, 0
    mse_dist = []
    for z in range(im_true.dim[2]):
        if np.sum(im_pred.data[:,:,z]):
            if np.sum(im_true.data[:,:,z]):
                x_true, y_true = [np.where(im_true.data[:,:,z] > 0)[i][0] for i in range(len(np.where(im_true.data[:,:,z] > 0)))]
                x_pred, y_pred = [np.where(im_pred.data[:,:,z] > 0)[i][0] for i in range(len(np.where(im_pred.data[:,:,z] > 0)))]
                
                dist = ((x_true-x_pred)*px)**2 + ((y_true-y_pred)*py)**2
                mse_dist.append(dist)
               
                count_slice += 1
        else:
            count_no_detection += 1
    
    print '\n# of slices not detected: ' + str(count_no_detection) + '/' + str(nz) + ' (' + str(round(100.0 * (count_no_detection) / nz, 2)) + '%)'
    print 'Accuracy of centerline detection (MSE) = ' + str(np.round(sqrt(sum(mse_dist)/float(count_slice)), 2)) + ' mm'
    print 'Max move between prediction and groundtruth = ' + str(np.round(max(mse_dist),2)) + ' mm'

    return sqrt(sum(mse_dist)/float(count_slice)), max(mse_dist)
def plot_centerline(fname_input, fname_centerline, fname_gold_standard, folder_output):

    im = Image(fname_input)
    im_pred = Image(fname_centerline)
    im_true = Image(fname_gold_standard)

    nx, ny, nz, nt, px, py, pz, pt = im_true.dim

    # Middle S-I slice
    slice_middle_S_I = im_true.data[:,:,int(nz)/2]
    x_middle_S_I = np.where(slice_middle_S_I==1)[0][0]

    slice_display = im.data[x_middle_S_I,:,:]

    zz, y_true, y_pred = [], [], []
    for z in range(nz):
        if np.sum(im_pred.data[:,:,z]) and np.sum(im_true.data[:,:,z]):
            y_pred.append(np.where(im_pred.data[:,:,z] > 0)[1][0])
            y_true.append(np.where(im_true.data[:,:,z] > 0)[1][0])
            zz.append(z)

    fig = plt.figure(figsize=(15,15))
    ax = plt.subplot(1, 1, 1)

    ax.imshow(np.rot90(np.fliplr(slice_display)),cmap=plt.get_cmap('gray'))
    plt.plot(y_true, zz, c='gold', linewidth=2)
    plt.plot(y_pred, zz, c='skyblue', linewidth=2)

    ax.set_axis_off()
    plt.savefig(folder_output + 'centerline_res.png')

    from skimage.transform import rotate
    from skimage import io
    im_IS = io.imread(folder_output + 'centerline_res.png')
    im_SI = rotate(im_IS, 180)
    io.imsave(folder_output + 'centerline_res.png', im_SI)

def get_parser():
    # Initialize parser
    parser = Parser(__file__)

    # Mandatory arguments
    parser.usage.set_description("This program takes as input an anatomic image and outputs a binary image with the spinal cord centerline/segmentation.")
    parser.add_option(name="-i",
                      type_value="image_nifti",
                      description="Input image",
                      mandatory=True)

    parser.add_option(name="-c",
                      type_value="multiple_choice",
                      description="type of image contrast, t2: cord dark / CSF bright ; t1: cord bright / CSF dark.",
                      mandatory=True,
                      example=['t1', 't2'])

    parser.add_option(name="-o",
                      type_value="file_output",
                      description="Output image",
                      mandatory=True)

    parser.add_option(name="-imodel",
                      type_value="str",
                      description="Model path",
                      mandatory=True,
                      example="/Users/chgroc/data/spine_detection/results2D/model_t2_linear_000/LinearSVM_train.pkl")

    parser.add_option(name="-threshold",
                      type_value="float",
                      description="Threshold value",
                      mandatory=True,
                      example=0.5)

    parser.add_option(name="-param",
                      type_value="str",
                      description="Path to Grid search parameters dict",
                      mandatory=True)

    parser.add_option(name="-eval",
                      type_value="int",
                      description="Choice to do (1) or not do (0) the validation step: compute error + plot 2D centerline",
                      mandatory=False,
                      default_value=0,
                      example=0)

    parser.add_option(name="-r",
                      type_value="int",
                      description="Remove temporary files",
                      mandatory=False,
                      default_value=1,
                      example=0)

    parser.add_option(name="-v",
                      type_value="int",
                      description="Verbose",
                      mandatory=False,
                      default_value=0,
                      example=0)
    return parser


if __name__ == "__main__":
    parser = get_parser()
    arguments = parser.parse(sys.argv[1:])

    # Classifier model
    fname_model = arguments['-imodel'].split('.')[0]
    print fname_model.split('/')[-1]
    if 'SVM' in fname_model.split('/')[-1]:
        feature_fct = extract_hog_feature
        model = Classifier_svm()
    else:
        feature_fct = extract_patch_feature
        print 'Benjamin: comment initialises tu ton CNN?'
    print '\nLoading model...'
    model.load(fname_model)
    print '...\n'
    threshold = arguments['-threshold']
    print '\n TODO: Donner un dictionnaire avec le chemin vers le modele et la classe correspondante + Thresh'

    path_params = arguments['-param']
    with open(path_params) as outfile:    
        params = pickle.load(outfile)
        outfile.close()
    #Patch and grid size
    patch_size = params['patch_size']
    initial_resolution = params['initial_resolution']
    initial_list_offset = params['initial_list_offset']
    offset = params['offset']
    max_iter = params['max_iter']

    if '-v' in arguments:
        verbose = int(arguments['-v'])

    if '-r' in arguments:
        bool_remove = int(arguments['-r'])

    if '-eval' in arguments:
        bool_eval = int(arguments['-eval'])

    # Input Image
    fname_input = arguments['-i']
    folder_input, subject_name = os.path.split(fname_input)
    subject_name = subject_name.split('.nii.gz')[0]
    im_data = Image(fname_input)
    print '\nInput Image: ' + fname_input

    # Output Folder
    fname_output = arguments['-o']
    folder_output, subject_name_out = os.path.split(fname_output)
    folder_output += '/'
    prefixe_output = subject_name.split('.nii.gz')[0]
    print '\nOutput Folder: ' + folder_output

    tick = time.time()

    im_data.data = 255.0 * (im_data.data - np.percentile(im_data.data, 0)) / np.abs(np.percentile(im_data.data, 0) - np.percentile(im_data.data, 100))

    print '\nRun Initial patch-based prediction'
    fname_seg = prediction_init(im_data, model, initial_resolution, initial_list_offset, 
                                 threshold, feature_fct, patch_size, folder_output + prefixe_output, verbose)
    
    print '\nRun Post Processing'
    post_processing(im_data, fname_seg, model, offset, max_iter,
                        threshold, feature_fct, patch_size, folder_output + prefixe_output, fname_output, verbose)

    print '\nProcessing time: ' + str(round(time.time() - tick,2)) + 's'

    if bool_eval:
        print '\nCompute Error'
        fname_gold_standard = folder_output + prefixe_output + '_centerline.nii.gz'
        print fname_gold_standard
        fname_mask_viewer = folder_output + prefixe_output + '_mask_viewer.nii.gz'
        print fname_mask_viewer
        if not os.path.isfile(fname_gold_standard):
            if not os.path.isfile(fname_mask_viewer):
                sct.run('sct_propseg -i ' + fname_input + ' -c ' + arguments['-c'] + ' -ofolder '+  folder_output + ' -centerline-binary -init-centerline viewer')
            else:
                sct.run('sct_propseg -i ' + fname_input + ' -c ' + arguments['-c'] + ' -ofolder '+  folder_output + ' -centerline-binary -init-centerline ' + fname_mask_viewer)
        mean_err, max_move = compute_error(fname_output, fname_gold_standard)
        plot_centerline(fname_input, fname_output, fname_gold_standard, folder_output + prefixe_output)

        print 'Input Image: ' + fname_input
        print 'Centerline Manual: ' + fname_gold_standard
        print 'Centerline Predicted: ' + fname_output

    if bool_remove:
        os.remove(folder_output + prefixe_output + '_reg_tmp.nii.gz')
        os.remove(folder_output + prefixe_output + '_seg_tmp.nii.gz')
        if 'eval' in arguments:
            os.remove(fname_gold_standard)
            os.remove(fname_mask_viewer)
            os.remove(folder_output + prefixe_output + '_seg.nii.gz')
