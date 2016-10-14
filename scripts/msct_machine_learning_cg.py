#!/usr/bin/env python
#########################################################################################
#
# This module contains some functions and classes for patch-based machine learning
#
# ---------------------------------------------------------------------------------------
# Copyright (c) 2016 Polytechnique Montreal <www.neuro.polymtl.ca>
# Authors: Benjamin De Leener
# Modified: 2016-10-04
#
# About the license: see the file LICENSE.TXT
#########################################################################################

import os
import sct_utils as sct
from msct_image import Image
import numpy as np
import itertools
import json
import pickle
from progressbar import Bar, ETA, Percentage, ProgressBar, Timer
from skimage.feature import hog
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials, space_eval
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, precision_score, recall_score, accuracy_score
from sklearn.base import BaseEstimator
from sklearn.preprocessing import StandardScaler
from sklearn import svm
from sklearn.externals import joblib
from os import listdir
from os.path import isfile, join
import time
import random 


import bz2
import cPickle as pickle
import multiprocessing as mp
import copy_reg
import types


def _pickle_method(method):
    """
    Author: Steven Bethard (author of argparse)
    http://bytes.com/topic/python/answers/552476-why-cant-you-pickle-instancemethods
    """
    func_name = method.im_func.__name__
    obj = method.im_self
    cls = method.im_class
    cls_name = ''
    if func_name.startswith('__') and not func_name.endswith('__'):
        cls_name = cls.__name__.lstrip('_')
    if cls_name:
        func_name = '_' + cls_name + func_name
    return _unpickle_method, (func_name, obj, cls)


def _unpickle_method(func_name, obj, cls):
    """
    Author: Steven Bethard
    http://bytes.com/topic/python/answers/552476-why-cant-you-pickle-instancemethods
    """
    for cls in cls.mro():
        try:
            func = cls.__dict__[func_name]
        except KeyError:
            pass
        else:
            break
    return func.__get__(obj, cls)

copy_reg.pickle(types.MethodType, _pickle_method, _unpickle_method)



def center_of_patch_equal_one(data):
    patch_size_x, patch_size_y = data['patches_gold'].shape[2], data['patches_gold'].shape[3]
    return np.squeeze(data['patches_gold'][:, 0, int(patch_size_x / 2), int(patch_size_y / 2)])


def extract_patches_from_image(path_dataset, fname_raw_images, fname_gold_images, patches_coordinates, patch_info, verbose=1):
    # input: list_raw_images
    # input: list_gold_images
    # output: list of patches. One patch is a pile of patches from (first) raw images and (second) gold images. Order are respected.

    # TODO: apply rotation of the image to take patches in planes event when doing the extraction in physical space

    patch_size = patch_info['patch_size']  # [int, int]
    patch_pixdim = patch_info['patch_pixdim']  # {'axial': [float, float], 'sagittal': [float, float], 'frontal': [float, float]}

    raw_images = [Image(path_dataset + fname) for fname in fname_raw_images]
    gold_images = [Image(path_dataset + fname) for fname in fname_gold_images]

    for k in range(len(patches_coordinates)):

        ind = [patches_coordinates[k][0], patches_coordinates[k][1], patches_coordinates[k][2]]
        patches_raw, patches_gold = [], []

        if 'axial' in patch_pixdim:
            range_x = np.linspace(ind[0] - (patch_size[0] / 2.0) * patch_pixdim['axial'][0], ind[0] + (patch_size[0] / 2.0) * patch_pixdim['axial'][0], patch_size[0])
            range_y = np.linspace(ind[1] - (patch_size[1] / 2.0) * patch_pixdim['axial'][1], ind[1] + (patch_size[1] / 2.0) * patch_pixdim['axial'][1], patch_size[1])
            coord_x, coord_y = np.meshgrid(range_x, range_y)
            coord_x = coord_x.ravel()
            coord_y = coord_y.ravel()
            coord_physical = [[coord_x[i], coord_y[i], ind[2]] for i in range(len(coord_x))]

            for raw_image in raw_images:
                grid_voxel = np.array(raw_image.transfo_phys2continuouspix(coord_physical))
                patch = np.reshape(raw_image.get_values(np.array([grid_voxel[:, 0], grid_voxel[:, 1], grid_voxel[:, 2]]),
                                                        interpolation_mode=1), (patch_size[0], patch_size[1]))
                patches_raw.append(np.expand_dims(patch, axis=0))

            for gold_image in gold_images:
                grid_voxel = np.array(gold_image.transfo_phys2continuouspix(coord_physical))
                patch = np.reshape(gold_image.get_values(np.array([grid_voxel[:, 0], grid_voxel[:, 1], grid_voxel[:, 2]]),
                                                         interpolation_mode=0), (patch_size[0], patch_size[1]))
                patches_gold.append(np.expand_dims(patch, axis=0))

        if 'sagittal' in patch_pixdim:
            range_x = np.linspace(ind[0] - (patch_size[0] / 2.0) * patch_pixdim['sagittal'][0], ind[0] + (patch_size[0] / 2.0) * patch_pixdim['sagittal'][0], patch_size[0])
            range_y = np.linspace(ind[1] - (patch_size[1] / 2.0) * patch_pixdim['sagittal'][1], ind[1] + (patch_size[1] / 2.0) * patch_pixdim['sagittal'][1], patch_size[1])
            coord_x, coord_y = np.meshgrid(range_x, range_y)
            coord_x = coord_x.ravel()
            coord_y = coord_y.ravel()
            coord_physical = [[ind[0], coord_x[i], coord_y[i]] for i in range(len(coord_x))]

            for raw_image in raw_images:
                grid_voxel = np.array(raw_image.transfo_phys2continuouspix(coord_physical))
                patch = np.reshape(raw_image.get_values(np.array([grid_voxel[:, 0], grid_voxel[:, 1], grid_voxel[:, 2]]),
                                                        interpolation_mode=1), (patch_size[0], patch_size[1]))
                patches_raw.append(np.expand_dims(patch, axis=0))

            for gold_image in gold_images:
                grid_voxel = np.array(gold_image.transfo_phys2continuouspix(coord_physical))
                patch = np.reshape(gold_image.get_values(np.array([grid_voxel[:, 0], grid_voxel[:, 1], grid_voxel[:, 2]]),
                                                         interpolation_mode=0), (patch_size[0], patch_size[1]))
                patches_gold.append(np.expand_dims(patch, axis=0))

        if 'frontal' in patch_pixdim:
            range_x = np.linspace(ind[0] - (patch_size[0] / 2.0) * patch_pixdim['frontal'][0], ind[0] + (patch_size[0] / 2.0) * patch_pixdim['frontal'][0], patch_size[0])
            range_y = np.linspace(ind[1] - (patch_size[1] / 2.0) * patch_pixdim['frontal'][1], ind[1] + (patch_size[1] / 2.0) * patch_pixdim['frontal'][1], patch_size[1])
            coord_x, coord_y = np.meshgrid(range_x, range_y)
            coord_x = coord_x.ravel()
            coord_y = coord_y.ravel()
            coord_physical = [[coord_x[i], ind[1], coord_y[i]] for i in range(len(coord_x))]

            for raw_image in raw_images:
                grid_voxel = np.array(raw_image.transfo_phys2continuouspix(coord_physical))
                patch = np.reshape(raw_image.get_values(np.array([grid_voxel[:, 0], grid_voxel[:, 1], grid_voxel[:, 2]]),
                                                        interpolation_mode=1), (patch_size[0], patch_size[1]))
                patches_raw.append(np.expand_dims(patch, axis=0))

            for gold_image in gold_images:
                grid_voxel = np.array(gold_image.transfo_phys2continuouspix(coord_physical))
                patch = np.reshape(gold_image.get_values(np.array([grid_voxel[:, 0], grid_voxel[:, 1], grid_voxel[:, 2]]),
                                                         interpolation_mode=0), (patch_size[0], patch_size[1]))
                patches_gold.append(np.expand_dims(patch, axis=0))

        patches_raw = np.concatenate(patches_raw, axis=0)
        patches_gold = np.concatenate(patches_gold, axis=0)

        yield {'patches_raw': patches_raw, 'patches_gold': patches_gold}


def get_minibatch(patch_iter, size):
    """Extract a minibatch of examples, return a tuple X_text, y.

    Note: size is before excluding invalid docs with no topics assigned.

    """
    data = [(patch['patches_raw'], patch['patches_gold']) for patch in itertools.islice(patch_iter, size)]

    if not len(data):
        return {'patches_raw': np.asarray([], dtype=np.float), 'patches_gold': np.asarray([], dtype=np.float)}

    patches_raw, patches_gold = zip(*data)
    patches_raw, patches_gold = np.asarray(patches_raw, dtype=np.float), np.asarray(patches_gold, dtype=np.float)

    return {'patches_raw': patches_raw, 'patches_gold': patches_gold}



def progress(stats):
    """Report progress information, return a string."""
    duration = time.time() - stats['t0']
    s = str(stats['n_train']) + " train samples (" + str(stats['n_train_pos']) + " positive)\n"
    s += str(stats['n_test']) + " test samples (" + str(stats['n_test_pos']) + " positive)\n"
    s += "accuracy: " + str(stats['accuracy']) + "\n"
    s += "precision: " + str(stats['precision']) + "\n"
    s += "recall: " + str(stats['recall']) + "\n"
    s += "roc: " + str(stats['roc']) + "\n"
    s += "in " + str(duration) + "s (" + str(stats['n_train'] / duration) + " samples/sec)"
    return s


def progress_train(stats):
    """Report progress information, return a string."""
    s = str(stats['n_train']) + " train samples (" + str(stats['n_train_pos']) + " positive)\n"

    if stats['total_hyperopt_time']:
        s += "Hyper param optimization in: " + str(stats['total_hyperopt_time']) + "s\n"

    s += "Training in: " + str(stats['total_fit_time']) + "s"
    
    return s

class FileManager():
    def __init__(self, dataset_path, fct_explore_dataset, patch_extraction_parameters, fct_groundtruth_patch):
        self.dataset_path = sct.slash_at_the_end(dataset_path, slash=1)
        # This function should take the path to the dataset as input and outputs the list of files (wrt dataset path) that compose the dataset (image + groundtruth)
        self.fct_explore_dataset = fct_explore_dataset

        self.patch_extraction_parameters = patch_extraction_parameters
        # ratio_dataset represents the ratio between the training, testing and validation datasets.
        # default is: 60% training, 20% testing, 20% validation
        if 'ratio_dataset' in self.patch_extraction_parameters:
            self.ratio_dataset = self.patch_extraction_parameters['ratio_dataset']
        else:
            self.ratio_dataset = [0.6, 0.2, 0.2]
        # patch size is the number of pixels that are in a patch in each dimensions. Patches are only 2D
        # warning: patch size must correspond to the ClassificationModel
        # Example: [32, 32] means a patch with 32x32 pixels
        if 'patch_size' in self.patch_extraction_parameters:
            self.patch_size = self.patch_extraction_parameters['patch_size']
        else:
            self.patch_size = None
        # patch_pixdim represents the resolution of the patch
        if 'patch_pixdim' in self.patch_extraction_parameters:
            self.patch_pixdim = self.patch_extraction_parameters['patch_pixdim']
        else:
            self.patch_pixdim = None
        # extract_all_positive is a boolean variable. If True, the system extracts all positive patches from the dataset
        if 'extract_all_positive' in self.patch_extraction_parameters:
            self.extract_all_positive = self.patch_extraction_parameters['extract_all_positive']
        else:
            self.extract_all_positive = False
        # extract_all_negative is a boolean variable. If True, the system extracts all positive patches from the dataset
        if 'extract_all_negative' in self.patch_extraction_parameters:
            self.extract_all_negative = self.patch_extraction_parameters['extract_all_negative']
        else:
            self.extract_all_negative = False
        # ratio_patches_voxels is the ratio of patches to extract in all the possible patches in the images. Typically = 10%
        if 'ratio_patches_voxels' in self.patch_extraction_parameters:
            self.ratio_patches_voxels = self.patch_extraction_parameters['ratio_patches_voxels']
        else:
            self.ratio_patches_voxels = 0.1
        if 'batch_size' in self.patch_extraction_parameters:
            self.batch_size = self.patch_extraction_parameters['batch_size']
        else:
            self.batch_size = 1

        # patch_info is the structure that will be transmitted for patches extraction
        self.patch_info = {'patch_size': self.patch_size, 'patch_pixdim': self.patch_pixdim}

        # this function will be called on each patch to know its class/label
        self.fct_groundtruth_patch = fct_groundtruth_patch

        self.list_files = np.array(self.fct_explore_dataset(self.dataset_path))
        self.number_of_images = len(self.list_files)

        self.training_dataset, self.testing_dataset = [], []

        # list_classes is a dictionary that contains all the classes that are present in the dataset
        # this list is filled up iteratively while exploring the dataset
        # the key is the label of the class and the element is the number of element of each class
        self.list_classes = {}

        # class_weights is a dictionary containing the ratio of each class and the most represented class
        # len(class_weights) = len(list_classes)
        self.class_weights = {}

    def decompose_dataset(self, path_output):
        array_indexes = range(self.number_of_images)
        np.random.shuffle(array_indexes)

        self.training_dataset = self.list_files[np.ix_(array_indexes[:int(self.ratio_dataset[0] * self.number_of_images)])]
        self.testing_dataset = self.list_files[np.ix_(array_indexes[int(self.ratio_dataset[0] * self.number_of_images):int(self.ratio_dataset[0] * self.number_of_images)
                                                                                            +int(self.ratio_dataset[1] * self.number_of_images)])]

        results = {
            'training': {'raw_images': [data[0].tolist() for data in self.training_dataset], 'gold_images': [data[1].tolist() for data in self.training_dataset]},
            'testing': {'raw_images': [data[0].tolist() for data in self.testing_dataset], 'gold_images': [data[1].tolist() for data in self.testing_dataset]},
            'dataset_path': self.dataset_path
        }
        with open(path_output + 'datasets.json', 'w') as outfile:
            json.dump(results, outfile)

        return self.training_dataset, self.testing_dataset

    def iter_minibatches(self, patch_iter, minibatch_size):
        """Generator of minibatches."""
        data = get_minibatch(patch_iter, minibatch_size)
        while len(data['patches_raw']):
            yield data
            data = get_minibatch(patch_iter, minibatch_size)

    def compute_patches_coordinates(self, image):
        if self.extract_all_negative or self.extract_all_positive:
            print 'Extract all negative/positive patches: feature not yet ready...'

        image_dim = image.dim

        x, y, z = np.mgrid[0:image_dim[0], 0:image_dim[1], 0:image_dim[2]]
        indexes = np.array(zip(x.ravel(), y.ravel(), z.ravel()))
        physical_coordinates = np.asarray(image.transfo_pix2phys(indexes))

        random_batch = np.random.choice(physical_coordinates.shape[0], int(round(physical_coordinates.shape[0] * self.ratio_patches_voxels)))

        return physical_coordinates[random_batch]

    def explore(self):
        # training dataset
        global_results_patches = {'patch_info': self.patch_info}

        # TRAINING DATASET
        results_training = {}
        classes_training = {}
        for i, fnames in enumerate(self.training_dataset):
            fname_raw_images = self.training_dataset[i][0]
            fname_gold_images = self.training_dataset[i][1]
            reference_image = Image(self.dataset_path + fname_raw_images[0])  # first raw image is selected as reference

            patches_coordinates = self.compute_patches_coordinates(reference_image)
            print 'Number of patches in ' + fname_raw_images[0] + ' = ' + str(patches_coordinates.shape[0])

            stream_data = extract_patches_from_image(path_dataset=self.dataset_path,
                                                     fname_raw_images=fname_raw_images,
                                                     fname_gold_images=fname_gold_images,
                                                     patches_coordinates=patches_coordinates,
                                                     patch_info=self.patch_info,
                                                     verbose=1)

            minibatch_iterator_test = self.iter_minibatches(stream_data, self.batch_size)
            labels = []
            pbar = ProgressBar(widgets=[
                Timer(),
                ' ', Percentage(),
                ' ', Bar(),
                ' ', ETA()], max_value=patches_coordinates.shape[0])
            pbar.start()
            number_done = 0
            for data in minibatch_iterator_test:
                if np.ndim(data['patches_gold']) == 4:
                    labels.extend(center_of_patch_equal_one(data))
                    number_done += data['patches_gold'].shape[0]
                    pbar.update(number_done)
            pbar.finish()

            classes_in_image, counts = np.unique(labels, return_counts=True)
            for j, cl in enumerate(classes_in_image):
                if str(cl) not in classes_training:
                    classes_training[str(cl)] = [counts[j], 0.0]
                else:
                    classes_training[str(cl)][0] += counts[j]

            results_training[str(i)] = [[patches_coordinates[j, :].tolist(), labels[j]] for j in range(len(labels))]

        global_results_patches['training'] = results_training

        count_max_class, max_class = 0, ''
        for cl in classes_training:
            if classes_training[cl][0] > count_max_class:
                max_class = cl
        for cl in classes_training:
            classes_training[cl][1] = classes_training[cl][0] / float(classes_training[max_class][0])


        # TESTING DATASET
        results_testing = {}
        classes_testing = {}
        for i, fnames in enumerate(self.testing_dataset):
            fname_raw_images = self.testing_dataset[i][0]
            fname_gold_images = self.testing_dataset[i][1]
            reference_image = Image(self.dataset_path + fname_raw_images[0])  # first raw image is selected as reference

            patches_coordinates = self.compute_patches_coordinates(reference_image)
            print 'Number of patches in ' + fname_raw_images[0] + ' = ' + str(patches_coordinates.shape[0])

            stream_data = extract_patches_from_image(path_dataset=self.dataset_path,
                                                     fname_raw_images=fname_raw_images,
                                                     fname_gold_images=fname_gold_images,
                                                     patches_coordinates=patches_coordinates,
                                                     patch_info=self.patch_info,
                                                     verbose=1)

            minibatch_iterator_test = self.iter_minibatches(stream_data, self.batch_size)
            labels = []
            pbar = ProgressBar(widgets=[
                Timer(),
                ' ', Percentage(),
                ' ', Bar(),
                ' ', ETA()], max_value=patches_coordinates.shape[0])
            pbar.start()
            number_done = 0
            for data in minibatch_iterator_test:
                if np.ndim(data['patches_gold']) == 4:
                    labels.extend(center_of_patch_equal_one(data))
                    number_done += data['patches_gold'].shape[0]
                    pbar.update(number_done)
            pbar.finish()

            classes_in_image, counts = np.unique(labels, return_counts=True)
            for j, cl in enumerate(classes_in_image):
                if str(cl) not in classes_testing:
                    classes_testing[str(cl)] = [counts[j], 0.0]
                else:
                    classes_testing[str(cl)][0] += counts[j]

            results_testing[str(i)] = [[patches_coordinates[j, :].tolist(), labels[j]] for j in range(len(labels))]

        global_results_patches['testing'] = results_testing

        count_max_class, max_class = 0, ''
        for cl in classes_testing:
            if classes_testing[cl][0] > count_max_class:
                max_class = cl
        for cl in classes_testing:
            classes_testing[cl][1] = classes_testing[cl][0] / float(classes_testing[max_class][0])

        global_results_patches['statistics'] = {'classes_training': classes_training, 'classes_testing': classes_testing}

        with open(path_output + 'patches.json', 'w') as outfile:
            json.dump(global_results_patches, outfile)


class Trainer():
    def __init__(self, data_filemanager_path, datasets_dict_fname, patches_dict_prefixe, patches_pos_dict_prefixe, classifier_model, fct_feature_extraction, param_training, results_path, model_path):

        with bz2.BZ2File(data_filemanager_path + datasets_dict_fname, 'rb') as f:
            datasets_dict = pickle.load(f)
            f.close()

        self.data_filemanager_path = sct.slash_at_the_end(data_filemanager_path, slash=1)
        self.datasets_dict_fname = datasets_dict_fname

        self.dataset_path = sct.slash_at_the_end(str(datasets_dict['dataset_path']), slash=1)
        print self.dataset_path
        print ' '
        print 'Attention path modifie'
        self.dataset_path = sct.slash_at_the_end('/Users/chgroc/data/spine_detection/data/test_very_small/', slash=1)
        print self.dataset_path
        print ' '

        self.dataset_stats = datasets_dict['statistics']
        self.patch_info = datasets_dict['patch_info']

        self.training_dataset = datasets_dict['training']
        self.fname_training_raw_images = datasets_dict['training']['raw_images']
        self.fname_training_gold_images = datasets_dict['training']['gold_images']
        self.patches_dict_prefixe = patches_dict_prefixe
        if patches_pos_dict_prefixe is not None:
            self.patches_pos_dict_prefixe = patches_pos_dict_prefixe

        self.testing_dataset = datasets_dict['testing']
        self.fname_testing_raw_images = datasets_dict['testing']['raw_images']
        self.fname_testing_gold_images = datasets_dict['testing']['gold_images']

        self.model_name = classifier_model['model_name']
        self.model = classifier_model['model']

        if 'model_hyperparam' in classifier_model:
            self.model_hyperparam = classifier_model['model_hyperparam']
        else:
            self.model_hyperparam = None

        self.fct_feature_extraction = fct_feature_extraction

        self.param_training = param_training
        self.param_hyperopt = self.param_training['hyperopt']

        if 'ratio_patch_per_img' in self.param_training:
            self.ratio_patch_per_img = self.param_training['ratio_patch_per_img']
        else:
            self.ratio_patch_per_img = 1.0

        self.results_path = sct.slash_at_the_end(results_path, slash=1)
        self.model_path = sct.slash_at_the_end(model_path, slash=1)

        self.stats = {'n_train': 0, 'n_train_pos': 0,
                        'n_test': 0, 'n_test_pos': 0,
                        'accuracy': 0.0, 'precision': 0.0, 'recall': 0.0, 'roc': 0.0,
                        'accuracy_history': [(0, 0)], 'precision_history': [(0, 0)], 'recall_history': [(0, 0)],
                        'roc_history': [(0, 0)],
                        't0': time.time(),
                        'runtime_history': [(0, 0)], 'total_fit_time': 0.0, 'total_hyperopt_time': 0.0}

    def prepare_patches(self, fname_raw_images):
        ###############################################################################################################
        #
        # Output:       Coordinates Dict{'index in datasets.pbz2': [coord[float, float, float], ...], ...}
        #               Labels Dict{'index in datasets.pbz2': [label(int), ...], ...]}
        #
        ###############################################################################################################

        # Dict initialization
        coord_prepared, label_prepared = {}, {}

        coord_label_patches_file, coord_label_patches_pos_file = [], []
        
        # Iteration for all INPUT fname subjects
        for i, fname in enumerate(fname_raw_images):

            # By default the first: fname[0]
            fname_patch = self.patches_dict_prefixe + fname[0].split('.',1)[0] + '.' + self.datasets_dict_fname.split('.',1)[1]
            with bz2.BZ2File(self.data_filemanager_path + fname_patch, 'rb') as f:
                coord_label_patches = pickle.load(f)
                f.close()

            fname_patch_pos = self.patches_pos_dict_prefixe + fname[0].split('.',1)[0] + '.' + self.datasets_dict_fname.split('.',1)[1]
            if os.path.exists(self.data_filemanager_path + fname_patch_pos):
                with bz2.BZ2File(self.data_filemanager_path + fname_patch_pos, 'rb') as f:
                    coord_label_patches_pos = pickle.load(f)
                    f.close()
            else:
                coord_label_patches_pos = None

            # List of coord and List of label initialization
            coord_prepared_tmp, label_prepared_tmp = [], []

            # If a label class balance is expected
            # Only for training dataset
            if coord_label_patches_pos is not None:
                nb_patches_pos_tot = len(coord_label_patches_pos)
                # For CNN, nb_patches_pos_to_extract = nb_patches_pos_tot
                # Same ratio to extract from (random) patches and from pos patches
                nb_patches_pos_to_extract = int(self.ratio_patch_per_img * nb_patches_pos_tot)

                # Iteration for all nb_patches_pos_to_extract pos patches
                for i_patch_pos in range(nb_patches_pos_to_extract):
                    coord_prepared_tmp.append(coord_label_patches_pos[i_patch_pos][0])
                    label_prepared_tmp.append(coord_label_patches_pos[i_patch_pos][1])
            else:
                print 'If a label class balance is expected: Please provide Coordinates of positive patches'

            nb_patches_tot = len(coord_label_patches)
            nb_patches_to_extract = int(self.ratio_patch_per_img * nb_patches_tot)

            # Iteration for all nb_patches_to_extract patches
            for i_patch in range(nb_patches_to_extract):
                coord_prepared_tmp.append(coord_label_patches[i_patch][0])
                label_prepared_tmp.append(coord_label_patches[i_patch][1])

            # Shuffle to prevent the case where all pos patches are gather in one minibatch
            index_shuf = range(len(coord_prepared_tmp))
            np.random.shuffle(index_shuf)

            coord_prepared[str(i)] = [coord_prepared_tmp[idx] for idx in index_shuf]
            label_prepared[str(i)] = [label_prepared_tmp[idx] for idx in index_shuf]

        return coord_prepared, label_prepared


    def extract_patch_feature_label_from_image(self, path_dataset, fname_raw_images, patches_coordinates, patches_labels):
        ###############################################################################################################
        #
        # Output:       Dict {'patches_raw': (32,32), 'patches_feature': (32,32) or feature vector, 'patches_label': int}
        #
        # TODO:         Histogram Normalization
        #
        ###############################################################################################################

        patch_size = self.patch_info['patch_size']  # [int, int]
        patch_pixdim = self.patch_info['patch_pixdim']  # {'axial': [float, float], 'sagittal': [float, float], 'frontal': [float, float]}

        raw_images = [Image(path_dataset + fname) for fname in fname_raw_images]

        # Intensity Normalization
        for img in raw_images:
            img.data = 255.0 * (img.data - np.percentile(img.data, 0)) / np.abs(np.percentile(img.data, 0) - np.percentile(img.data, 100))

        for k in range(len(patches_coordinates)):

            ind = [patches_coordinates[k][0], patches_coordinates[k][1], patches_coordinates[k][2]]
            label = int(patches_labels[k])

            patches_raw, patches_feature = [], []

            if 'axial' in patch_pixdim:
                range_x = np.linspace(ind[0] - (patch_size[0] / 2.0) * patch_pixdim['axial'][0], ind[0] + (patch_size[0] / 2.0) * patch_pixdim['axial'][0], patch_size[0])
                range_y = np.linspace(ind[1] - (patch_size[1] / 2.0) * patch_pixdim['axial'][1], ind[1] + (patch_size[1] / 2.0) * patch_pixdim['axial'][1], patch_size[1])
                coord_x, coord_y = np.meshgrid(range_x, range_y)
                coord_x = coord_x.ravel()
                coord_y = coord_y.ravel()
                coord_physical = [[coord_x[i], coord_y[i], ind[2]] for i in range(len(coord_x))]

                for raw_image in raw_images:
                    grid_voxel = np.array(raw_image.transfo_phys2continuouspix(coord_physical))
                    patch = np.reshape(raw_image.get_values(np.array([grid_voxel[:, 0], grid_voxel[:, 1], grid_voxel[:, 2]]),
                                                            interpolation_mode=1), (patch_size[0], patch_size[1]))
                    patches_raw.append(np.expand_dims(patch, axis=0))
                    # Feature Extraction
                    patches_feature.append(self.fct_feature_extraction(patch))

            if 'sagittal' in patch_pixdim:
                range_x = np.linspace(ind[0] - (patch_size[0] / 2.0) * patch_pixdim['sagittal'][0], ind[0] + (patch_size[0] / 2.0) * patch_pixdim['sagittal'][0], patch_size[0])
                range_y = np.linspace(ind[1] - (patch_size[1] / 2.0) * patch_pixdim['sagittal'][1], ind[1] + (patch_size[1] / 2.0) * patch_pixdim['sagittal'][1], patch_size[1])
                coord_x, coord_y = np.meshgrid(range_x, range_y)
                coord_x = coord_x.ravel()
                coord_y = coord_y.ravel()
                coord_physical = [[ind[0], coord_x[i], coord_y[i]] for i in range(len(coord_x))]

                for raw_image in raw_images:
                    grid_voxel = np.array(raw_image.transfo_phys2continuouspix(coord_physical))
                    patch = np.reshape(raw_image.get_values(np.array([grid_voxel[:, 0], grid_voxel[:, 1], grid_voxel[:, 2]]),
                                                            interpolation_mode=1), (patch_size[0], patch_size[1]))
                    patches_raw.append(np.expand_dims(patch, axis=0))
                    # Feature Extraction
                    patches_feature.append(self.fct_feature_extraction(patch))

            if 'frontal' in patch_pixdim:
                range_x = np.linspace(ind[0] - (patch_size[0] / 2.0) * patch_pixdim['frontal'][0], ind[0] + (patch_size[0] / 2.0) * patch_pixdim['frontal'][0], patch_size[0])
                range_y = np.linspace(ind[1] - (patch_size[1] / 2.0) * patch_pixdim['frontal'][1], ind[1] + (patch_size[1] / 2.0) * patch_pixdim['frontal'][1], patch_size[1])
                coord_x, coord_y = np.meshgrid(range_x, range_y)
                coord_x = coord_x.ravel()
                coord_y = coord_y.ravel()
                coord_physical = [[coord_x[i], ind[1], coord_y[i]] for i in range(len(coord_x))]

                for raw_image in raw_images:
                    grid_voxel = np.array(raw_image.transfo_phys2continuouspix(coord_physical))
                    patch = np.reshape(raw_image.get_values(np.array([grid_voxel[:, 0], grid_voxel[:, 1], grid_voxel[:, 2]]),
                                                            interpolation_mode=1), (patch_size[0], patch_size[1]))
                    patches_raw.append(np.expand_dims(patch, axis=0))
                    # Feature Extraction
                    patches_feature.append(self.fct_feature_extraction(patch))

            patches_raw = np.concatenate(patches_raw, axis=0)
            patches_feature = np.concatenate(patches_feature, axis=0)

            yield {'patches_raw': patches_raw, 'patches_feature': patches_feature, 'patches_label': label}


    def get_minibatch_patch_feature_label(self, patch_iter, size):
    ###############################################################################################################
    #
    # Output:       Dict {'patches_raw': (size, nb_channel, 32, 32), 'patches_feature': (size, 32, 32) or (size, feature vector size), 'patches_label': (size,)}
    #
    # TODO:         patches_features, patches_label: multichannel case
    # 
    ###############################################################################################################

        data = [(patch['patches_raw'], patch['patches_feature'], patch['patches_label']) for patch in itertools.islice(patch_iter, size)]

        if not len(data):
            return {'patches_raw': np.asarray([], dtype=np.float), 'patches_feature': np.asarray([], dtype=np.float), 'patches_label': np.asarray([], dtype=int)}

        patches_raw, patches_feature, patches_label = zip(*data)
        patches_raw, patches_feature, patches_label = np.asarray(patches_raw, dtype=np.float), np.asarray(patches_feature, dtype=np.float), np.asarray(patches_label, dtype=int)

        return {'patches_raw': patches_raw, 'patches_feature': patches_feature, 'patches_label': patches_label}

    def iter_minibatches_trainer(self, coord_dict_prepared, label_dict_prepared, minibatch_size, fname_dataset):
    ###############################################################################################################
    #
    # TODO:         Should be checked
    #               patches_features, patches_label: multichannel case
    # 
    ###############################################################################################################

        temp_minibatch = {'patches_raw': np.empty( shape=(0, 0, 0, 0) ), 'patches_feature': np.empty( shape=(0, 0) ), 'patches_label': np.empty( shape=(0,) )} # len=0 

        for index_fname_image in coord_dict_prepared:

            fname_raw_cur = map(str, fname_dataset['raw_images'][int(index_fname_image)])
            fname_gold_cur = map(str, fname_dataset['gold_images'][int(index_fname_image)])
            
            stream_data = self.extract_patch_feature_label_from_image(path_dataset=self.dataset_path,
                                                                        fname_raw_images=fname_raw_cur,
                                                                        patches_coordinates=coord_dict_prepared[str(index_fname_image)],
                                                                        patches_labels=label_dict_prepared[str(index_fname_image)])

            minibatch = self.get_minibatch_patch_feature_label(stream_data, minibatch_size - temp_minibatch['patches_raw'].shape[0])

            if minibatch['patches_raw'].shape[0] == minibatch_size:
                yield minibatch
            else:
                if temp_minibatch['patches_raw'].shape[0] == 0:
                    temp_minibatch = minibatch # len!=0
                else:
                    patches_raw_temp = np.concatenate([temp_minibatch['patches_raw'], minibatch['patches_raw']], axis=0)
                    patches_feature_temp = np.concatenate([temp_minibatch['patches_feature'], minibatch['patches_feature']], axis=0)
                    patches_gold_temp = np.concatenate([temp_minibatch['patches_label'], minibatch['patches_label']], axis=0)
                    minibatch = {'patches_raw': patches_raw_temp, 'patches_feature': patches_feature_temp, 'patches_label': patches_gold_temp} # concat
                    if minibatch['patches_raw'].shape[0] == minibatch_size:
                        temp_minibatch = {'patches_raw': np.empty( shape=(0, 0, 0, 0) ), 'patches_feature': np.empty( shape=(0, 0) ), 'patches_label': np.empty( shape=(0,) )}
                        yield minibatch
                    else:
                        temp_minibatch = minibatch

    def hyperparam_optimization(self, coord_prepared_train, label_prepared_train, ratio_test):
    ###############################################################################################################
    #
    # TODO:         hyperopt_train_test: To be adapted to CNN architecture: https://github.com/fchollet/keras/issues/1591
    #               hp.uniform and hp.choice: To be discussed with Benjamin
    # 
    ###############################################################################################################

        # hyperparam dict must be provided
        if self.model_hyperparam is not None:
            nb_subj_test = int(len(coord_prepared_train) * ratio_test) # Nb subjects used for hyperopt testing
            nb_patch_all_train_subj = [len(coord_prepared_train[str(i)]) for i in coord_prepared_train] # List nb patches available for each subj in all training dataset
            
            if 'minibatch_size_test' in self.param_training:
                minibatch_size_test = self.param_training['minibatch_size_test']
            else:
                test_minibatch_size = sum(nb_patch_all_train_subj[:nb_subj_test]) # Define minibatch size used for hyperopt testing
            
            if 'minibatch_size_train' in self.param_training:
                minibatch_size_train = self.param_training['minibatch_size_train']
            else:
                train_minibatch_size = sum(nb_patch_all_train_subj[nb_subj_test:]) # Define minibatch size used for hyperopt training

            # Split coord_prepared_train and label_prepared_train for hyperopt training and testing
            cmpt = 0
            coord_prepared_train_hyperopt, coord_prepared_test_hyperopt = {}, {}
            label_prepared_train_hyperopt, label_prepared_test_hyperopt = {}, {}
            for i in coord_prepared_train:
                if cmpt < nb_subj_test:
                    coord_prepared_test_hyperopt[str(i)] = coord_prepared_train[str(i)]
                    label_prepared_test_hyperopt[str(i)] = label_prepared_train[str(i)]
                else:
                    coord_prepared_train_hyperopt[str(i)] = coord_prepared_train[str(i)]
                    label_prepared_train_hyperopt[str(i)] = label_prepared_train[str(i)]
                cmpt += 1

            # Create minibatch iterators for hyperopt training and testing
            minibatch_iterator_test = self.iter_minibatches_trainer(coord_prepared_test_hyperopt, label_prepared_test_hyperopt, 
                                                                test_minibatch_size, self.training_dataset)
            minibatch_iterator_train = self.iter_minibatches_trainer(coord_prepared_train_hyperopt, label_prepared_train_hyperopt, 
                                                                train_minibatch_size, self.training_dataset)

            # Same testing dataset for all hyperopt algo
            test_samples = minibatch_iterator_test.next()
            X_test = np.array(test_samples['patches_feature'])
            y_true = np.array(test_samples['patches_label'])

            # Create hyperopt dict compatible with hyperopt Lib
            model_hyperparam_hyperopt = {}                             
            for param in self.model_hyperparam:
                param_cur = self.model_hyperparam[param]
                if type(param_cur) is list and len(param_cur) == 2:
                    model_hyperparam_hyperopt[param] = hp.uniform(param, param_cur[0], param_cur[1])
                else:
                    model_hyperparam_hyperopt[param] = hp.choice(param, param_cur)

            # Objective function
            def hyperopt_train_test(params):

                    ### Benjamin: ajouter ici self.param_hyperopt['nb_epoch']
                    self.model.set_params(params) # Update model hyperparam with params provided by hyperopt library algo

                    tick = time.time()
                    self.model.train(X_train, y_train)
                    toc = time.time()-tick

                    ### Benjamin: ajouter ici self.param_hyperopt['eval_factor']
                    y_pred = self.model.predict(X_test)
                    score = self.param_hyperopt['fct'](y_true, y_pred) # Score to maximize

                    return {'loss': -score, 'status': STATUS_OK, 'eval_time': toc}

            cmpt = 0
            for data in minibatch_iterator_train:
                X_train = np.array(data['patches_feature'])
                y_train = np.array(data['patches_label'])

                # Trials object: results report
                trials = Trials()
                # Documentation: https://github.com/hyperopt/hyperopt/wiki/FMin
                best = fmin(hyperopt_train_test, model_hyperparam_hyperopt, algo=self.param_hyperopt['algo'], max_evals=self.param_hyperopt['nb_eval'], trials=trials)

                # Save results
                pickle.dump(trials.trials, open(self.results_path + 'trials_' + str(cmpt).zfill(3) + '.pkl', "wb"))
                cmpt += 1
        
        else:
            print ' '
            print 'Please provide a hyper parameter dict (called \'model_hyperparam\') in your classifier_model dict'
            print ' '


    def set_hyperopt_train(self, coord_train, label_train, path_best_trial=''):
    ###############################################################################################################
    #
    # IF path_best_trial='':
    #       - Open all trials_*.pkl generated by hyperparam_optimization
    #       - Find better score: save related trials_*.pkl file as trials_best.pkl
    # ELSE:
    #       - load path_best_trial
    # 
    # Set optimized params to self.model and save it as self.model_name + '_opt'
    #
    # Train the model 
    # 
    ###############################################################################################################

        if path_best_trial == '':
            fname_trials = [f for f in listdir(self.results_path) if isfile(join(self.results_path, f)) and f.startswith('trials_')]

            trials_score_list = []
            trials_eval_time_list = []
            for f in fname_trials:
                with open(self.results_path + f) as outfile:    
                    trial = pickle.load(outfile)
                    outfile.close()
                loss_list = [trial[i]['result']['loss'] for i in range(len(trial))]
                eval_time_list = [trial[i]['result']['eval_time'] for i in range(len(trial))]
                trials_score_list.append(min(loss_list))
                trials_eval_time_list.append(sum(eval_time_list))

            self.stats['total_hyperopt_time'] = sum(trials_eval_time_list)

            idx_best_trial = trials_score_list.index(min(trials_score_list))
            with open(self.results_path + fname_trials[idx_best_trial]) as outfile:    
                best_trial = pickle.load(outfile)
                pickle.dump(best_trial, open(self.results_path + 'best_trial.pkl', "wb"))
                outfile.close()
        else:
            with open(path_best_trial) as outfile:    
                best_trial = pickle.load(outfile)
                outfile.close()

        loss_list = [best_trial[i]['result']['loss'] for i in range(len(best_trial))]
        idx_best_params = loss_list.index(min(loss_list))
        best_params = best_trial[idx_best_params]['misc']['vals']

        if self.model_hyperparam is not None:

            model_hyperparam_opt = {}
            for k in self.model_hyperparam.keys():
                if isinstance(best_params[k][0], int):
                    model_hyperparam_opt[k] = self.model_hyperparam[k][best_params[k][0]]
                else:
                    model_hyperparam_opt[k] = float(best_params[k][0])

            self.model.set_params(model_hyperparam_opt)
            
            if 'minibatch_size_train' in self.param_training:
                minibatch_size_train = self.param_training['minibatch_size_train']
            else:
                minibatch_size_train = sum([len(coord_train[str(i)]) for i in coord_train])

            minibatch_iterator_train = self.iter_minibatches_trainer(coord_train, label_train, 
                                                                minibatch_size_train, self.training_dataset)
            tick = time.time()
            for data_train in minibatch_iterator_train:
                X_train = data_train['patches_feature']
                y_train = data_train['patches_label']
                
                self.model.train(X_train, y_train)
                
                self.stats['total_fit_time'] += time.time() - tick
                self.stats['n_train'] += X_train.shape[0]
                self.stats['n_train_pos'] += sum(y_train)

            self.train_model_path = self.model_path + self.model_name + '_opt'
            self.model.save(self.train_model_path)

            print progress_train(self.stats)

        else:
            print ' '
            print 'Please provide a hyper parameter dict (called \'model_hyperparam\') in your classifier_model dict'
            print ' '

    def run_prediction(self, coord_test, label_test):
    ###############################################################################################################
    #
    # TODO:     Check CNN compatibility
    #           Commented lines: used in msct_keras_classification
    # 
    ###############################################################################################################

        # 
        if 'minibatch_size_test' in self.param_training:
            minibatch_size_test = self.param_training['minibatch_size_test']
        else:
            minibatch_size_test = sum([len(coord_test[str(i)]) for i in coord_test])

        minibatch_iterator_test = self.iter_minibatches_trainer(coord_test, label_test, 
                                                            minibatch_size_test, self.testing_dataset)

        self.stats['prediction_time'] = 0
        y_pred, y_test = [], []

        for data_test in minibatch_iterator_test:
            X_test = data_test['patches_feature']
            y_test_cur = data_test['patches_label']

            tick = time.time()
            # y_pred_cur = self.model.predict(X_test, batch_size=32)
            y_pred_cur = self.model.predict(X_test)
            self.stats['prediction_time'] += time.time() - tick
            # y_pred.extend(np.argmax(y_pred_cur, axis=1).tolist())
            y_pred.extend(y_pred_cur)
            y_test.extend(y_test_cur)
            self.stats['n_test'] += X_test.shape[0]
            self.stats['n_test_pos'] += sum(y_test_cur)

        y_test = np.array(y_test)
        y_pred = np.array(y_pred)
        self.stats['accuracy'] = accuracy_score(y_test, y_pred)
        self.stats['precision'] = precision_score(y_test, y_pred)
        self.stats['recall'] = recall_score(y_test, y_pred)
        self.stats['roc'] = roc_auc_score(y_test, y_pred)

        acc_history = (self.stats['accuracy'], self.stats['n_train'])
        self.stats['accuracy_history'].append(acc_history)

        precision_history = (self.stats['precision'], self.stats['n_train'])
        self.stats['precision_history'].append(precision_history)

        recall_history = (self.stats['recall'], self.stats['n_train'])
        self.stats['recall_history'].append(recall_history)

        roc_history = (self.stats['roc'], self.stats['n_train'])
        self.stats['roc_history'].append(roc_history)

        run_history = (self.stats['accuracy'])
        self.stats['runtime_history'].append(run_history)

        print progress(self.stats)

        pickle.dump(self.stats, open(self.results_path + self.model_name + '_pred.pkl', "wb"))