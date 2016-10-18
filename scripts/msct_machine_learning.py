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
import bz2
import cPickle as pickle
from progressbar import Bar, ETA, Percentage, ProgressBar, Timer
import multiprocessing as mp

import copy_reg
import types

# import pickle
from skimage.feature import hog
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials, space_eval
from sklearn.metrics import roc_auc_score, precision_score, recall_score, accuracy_score, roc_curve
from sklearn.externals import joblib
from os import listdir
from os.path import isfile, join
import time
import random 
import math
import matplotlib.pyplot as plt


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
    s = ''

    if 'n_train' in stats:
        s += str(stats['n_train']) + " train samples (" + str(stats['n_train_pos']) + " positive)\n"
        s += 'Training time: ' + str(stats['total_fit_time']) + ' s\n'

    if 'n_test' in stats:
        s += str(stats['n_test']) + " test samples (" + str(stats['n_test_pos']) + " positive)\n"
        s += "accuracy: " + str(stats['accuracy']) + "\n"
        s += "precision: " + str(stats['precision']) + "\n"
        s += "recall: " + str(stats['recall']) + "\n"
        s += "roc: " + str(stats['roc']) + "\n"
        s += 'Prediction time: ' + str(stats['total_predict_time']) + ' s\n'

    return s

class FileManager(object):
    def __init__(self, dataset_path, fct_explore_dataset, patch_extraction_parameters, fct_groundtruth_patch, path_output):
        self.dataset_path = sct.slash_at_the_end(dataset_path, slash=1)
        # This function should take the path to the dataset as input and outputs the list of files (wrt dataset path) that compose the dataset (image + groundtruth)
        self.fct_explore_dataset = fct_explore_dataset

        self.path_output = path_output

        self.patch_extraction_parameters = patch_extraction_parameters
        # ratio_dataset represents the ratio between the training, testing and validation datasets.
        # default is: 60% training, 20% testing, 20% validation
        if 'ratio_dataset' in self.patch_extraction_parameters:
            self.ratio_dataset = self.patch_extraction_parameters['ratio_dataset']
        else:
            self.ratio_dataset = [0.8, 0.2]
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

        self.cpu_number = mp.cpu_count()

    def iter_minibatches(self, patch_iter, minibatch_size):
        """Generator of minibatches."""
        data = get_minibatch(patch_iter, minibatch_size)
        while len(data['patches_raw']):
            yield data
            data = get_minibatch(patch_iter, minibatch_size)

    def decompose_dataset(self):
        array_indexes = range(self.number_of_images)
        np.random.shuffle(array_indexes)

        self.training_dataset = self.list_files[np.ix_(array_indexes[:int(self.ratio_dataset[0] * self.number_of_images)])]
        self.testing_dataset = self.list_files[np.ix_(array_indexes[int(self.ratio_dataset[0] * self.number_of_images):])]

        return self.training_dataset, self.testing_dataset

    def compute_patches_coordinates(self, image):
        image_dim = image.dim

        x, y, z = np.mgrid[0:image_dim[0], 0:image_dim[1], 0:image_dim[2]]
        indexes = np.array(zip(x.ravel(), y.ravel(), z.ravel()))
        physical_coordinates = np.asarray(image.transfo_pix2phys(indexes))

        random_batch = np.random.choice(physical_coordinates.shape[0], int(round(physical_coordinates.shape[0] * self.ratio_patches_voxels)))

        return physical_coordinates[random_batch]

    def worker_explore(self, arguments_worker):
        try:
            i = arguments_worker[0]
            dataset = arguments_worker[1]

            fname_raw_images = dataset[i][0]
            fname_gold_images = dataset[i][1]
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
                    labels.extend(self.fct_groundtruth_patch(data))
                    number_done += data['patches_gold'].shape[0]
                    pbar.update(number_done)
            pbar.finish()

            # processing results
            results = [[patches_coordinates[j, :].tolist(), labels[j]] for j in range(len(labels))]

            # write results in file
            path_fname, file_fname, ext_fname = sct.extract_fname(self.dataset_path + fname_raw_images[0])
            with bz2.BZ2File(self.path_output + 'patches_coordinates_' + file_fname + '.pbz2', 'w') as f:
                pickle.dump(results, f)

            del patches_coordinates

            return [i, labels]

        except KeyboardInterrupt:
            return

        except Exception as e:
            print 'Error on line {}'.format(sys.exc_info()[-1].tb_lineno)
            raise e

    def explore(self):
        if self.extract_all_positive:
            for i in range(len(self.training_dataset)):
                fname_gold_image = self.training_dataset[i][1][0]  # first gold image is the reference
                im_gold = Image(self.dataset_path + fname_gold_image)
                coordinates_positive = np.where(im_gold.data == 1)
                coordinates_positive = np.asarray([[coordinates_positive[0][j], coordinates_positive[1][j], coordinates_positive[2][j]] for j in range(len(coordinates_positive[0]))])
                coordinates_positive = np.asarray(im_gold.transfo_pix2phys(coordinates_positive))
                label_positive = np.ones(coordinates_positive.shape[0])

                results_positive = [[coordinates_positive[j, :].tolist(), label_positive[j]] for j in range(len(label_positive))]

                # write results in file
                path_fname, file_fname, ext_fname = sct.extract_fname(self.dataset_path + self.training_dataset[i][0][0])
                with bz2.BZ2File(self.path_output + 'patches_coordinates_positives_' + file_fname + '.pbz2', 'w') as f:
                    pickle.dump(results_positive, f)

        # TRAINING DATASET
        classes_training = {}

        pool = mp.Pool(processes=self.cpu_number)
        results = pool.map(self.worker_explore, itertools.izip(range(len(self.training_dataset)), itertools.repeat(self.training_dataset)))

        pool.close()
        try:
            pool.join()  # waiting for all the jobs to be done
            for result in results:
                labels = result[1]

                classes_in_image, counts = np.unique(labels, return_counts=True)
                for j, cl in enumerate(classes_in_image):
                    if str(cl) not in classes_training:
                        classes_training[str(cl)] = [counts[j], 0.0]
                    else:
                        classes_training[str(cl)][0] += counts[j]

        except KeyboardInterrupt:
            print "\nWarning: Caught KeyboardInterrupt, terminating workers"
            pool.terminate()
            sys.exit(2)
        except Exception as e:
            print "Error in FileManager on line {}".format(sys.exc_info()[-1].tb_lineno)
            print e
            sys.exit(2)

        count_max_class, max_class = 0, ''
        for cl in classes_training:
            if classes_training[cl][0] > count_max_class:
                max_class = cl
        for cl in classes_training:
            classes_training[cl][1] = classes_training[cl][0] / float(classes_training[max_class][0])

        # TESTING DATASET
        classes_testing = {}

        pool = mp.Pool(processes=self.cpu_number)
        results = pool.map(self.worker_explore, itertools.izip(range(len(self.testing_dataset)), itertools.repeat(self.testing_dataset)))

        pool.close()
        try:
            pool.join()  # waiting for all the jobs to be done
            for result in results:
                labels = result[1]

                classes_in_image, counts = np.unique(labels, return_counts=True)
                for j, cl in enumerate(classes_in_image):
                    if str(cl) not in classes_testing:
                        classes_testing[str(cl)] = [counts[j], 0.0]
                    else:
                        classes_testing[str(cl)][0] += counts[j]

        except KeyboardInterrupt:
            print "\nWarning: Caught KeyboardInterrupt, terminating workers"
            pool.terminate()
            sys.exit(2)
        except Exception as e:
            print "Error in FileManager on line {}".format(sys.exc_info()[-1].tb_lineno)
            print e
            sys.exit(2)

        count_max_class, max_class = 0, ''
        for cl in classes_testing:
            if classes_testing[cl][0] > count_max_class:
                max_class = cl
        for cl in classes_testing:
            classes_testing[cl][1] = classes_testing[cl][0] / float(classes_testing[max_class][0])

        results = {
            'training': {'raw_images': [data[0].tolist() for data in self.training_dataset],
                         'gold_images': [data[1].tolist() for data in self.training_dataset]},
            'testing': {'raw_images': [data[0].tolist() for data in self.testing_dataset],
                        'gold_images': [data[1].tolist() for data in self.testing_dataset]},
            'dataset_path': self.dataset_path,
            'statistics': {'classes_training': classes_training, 'classes_testing': classes_testing},
            'patch_info': self.patch_info
        }

        with bz2.BZ2File(self.path_output + 'datasets.pbz2', 'w') as f:
            pickle.dump(results, f)


class Trainer():
    def __init__(self, data_filemanager_path, datasets_dict_fname, patches_dict_prefixe, patches_pos_dict_prefixe, classifier_model, fct_feature_extraction, param_training, results_path, model_path):

        with bz2.BZ2File(data_filemanager_path + datasets_dict_fname, 'rb') as f:
            datasets_dict = pickle.load(f)
            f.close()

        self.data_filemanager_path = sct.slash_at_the_end(data_filemanager_path, slash=1)
        self.datasets_dict_fname = datasets_dict_fname

        self.dataset_path = sct.slash_at_the_end(str(datasets_dict['dataset_path']), slash=1)


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

        if 'data_path_local' in self.param_training:
            self.dataset_path = sct.slash_at_the_end(self.param_training['data_path_local'], slash=1)

        self.results_path = sct.slash_at_the_end(results_path, slash=1)
        self.model_path = sct.slash_at_the_end(model_path, slash=1)

    def prepare_patches(self, fname_raw_images, ratio_patch_per_img=[1.0, 1.0]):
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

            print fname

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
                nb_patches_pos_to_extract = int(ratio_patch_per_img[1] * nb_patches_pos_tot)

                # Iteration for all nb_patches_pos_to_extract pos patches
                for i_patch_pos in range(nb_patches_pos_to_extract):
                    coord_prepared_tmp.append(coord_label_patches_pos[i_patch_pos][0])
                    label_prepared_tmp.append(coord_label_patches_pos[i_patch_pos][1])
            else:
                print '\n' + fname[0] + '...'
                print '... if a label class balance is expected: Please provide Coordinates of positive patches\n'

            nb_patches_tot = len(coord_label_patches)
            nb_patches_to_extract = int(ratio_patch_per_img[0] * nb_patches_tot)

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

    def hyperparam_optimization(self, coord_prepared_train, label_prepared_train):
    ###############################################################################################################
    #
    # TODO:         hyperopt_train_test: To be adapted to CNN architecture: https://github.com/fchollet/keras/issues/1591
    # 
    ###############################################################################################################

        # hyperparam dict must be provided
        if self.model_hyperparam is not None:
            nb_subj_test = int(len(coord_prepared_train) * self.param_hyperopt['ratio_eval']) # Nb subjects used for hyperopt testing
            nb_patch_all_train_subj = [len(coord_prepared_train[str(i)]) for i in coord_prepared_train] # List nb patches available for each subj in all training dataset
            
            if self.param_training['minibatch_size_train'] is not None:
                train_minibatch_size = self.param_training['minibatch_size_train']
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
            
            # Create hyperopt dict compatible with hyperopt Lib
            model_hyperparam_hyperopt = {}                             
            for param in self.model_hyperparam:
                param_cur = self.model_hyperparam[param]
                if type(param_cur) is not list and type(param_cur) is not tuple:
                    model_hyperparam_hyperopt[param] = param_cur
                elif type(param_cur) is list and len(param_cur) == 2:
                    model_hyperparam_hyperopt[param] = hp.uniform(param, param_cur[0], param_cur[1])
                else:
                    model_hyperparam_hyperopt[param] = hp.choice(param, param_cur)

            print 'Hyperparam Dict to test:'
            print model_hyperparam_hyperopt
            print ' '

            # Objective function
            def hyperopt_train_test(params):

                # Create minibatch iterators for hyperopt training and testing
                minibatch_iterator_train = self.iter_minibatches_trainer(coord_prepared_train_hyperopt, label_prepared_train_hyperopt, 
                                                                train_minibatch_size, self.training_dataset)

                self.model.set_params(params) # Update model hyperparam with params provided by hyperopt library algo

                stats = {'n_train': 0, 'n_train_pos': 0,
                        'n_test': 0, 'n_test_pos': 0,
                        'accuracy': 0.0, 'precision': 0.0, 'recall': 0.0, 'roc': 0.0,
                        'accuracy_history': [(0, 0)], 'precision_history': [(0, 0)], 'recall_history': [(0, 0)], 'roc_history': [(0, 0)],
                        't0': time.time(), 'total_fit_time': 0.0, 'total_predict_time': 0.0}
                
                cmpt = 0
                for n_epoch in range(self.param_training['number_of_epochs']):
                    for data in minibatch_iterator_train:
                        X_train = np.array(data['patches_feature'])
                        y_train = np.array(data['patches_label'])

                        stats['n_train'] += X_train.shape[0]
                        stats['n_train_pos'] += sum(y_train)

                        self.model.train(X_train, y_train)

                        # evaluation
                        if cmpt % self.param_hyperopt['eval_factor'] == 0 and cmpt != 0:
                            self.run_prediction(coord_prepared_test_hyperopt, label_prepared_test_hyperopt, [stats['n_train'], trials.tids[-1]], stats)
                            self.model.save(self.model_path + self.model_name + '_' + str(stats['n_train']).zfill(12) + '_' + str(trials.tids[-1]).zfill(6))
                        
                        cmpt += 1

                stats['total_fit_time'] = time.time() - stats['t0']
                
                y_true, y_pred = self.run_prediction(coord_prepared_test_hyperopt, label_prepared_test_hyperopt, [trials.tids[-1], stats['n_train']], stats)
                self.model.save(self.model_path + self.model_name + '_' + str(trials.tids[-1]).zfill(6) + '_' + str(stats['n_train']).zfill(12))

                score = self.param_hyperopt['fct'](y_true, y_pred) # Score to maximize

                return {'loss': -score, 'status': STATUS_OK, 'eval_time': stats['total_fit_time']}

            print '\nStarting Hyperopt...'
            print '... with ' + str(len(coord_prepared_train)) + ' images for training'
            print '... and ' + str(nb_subj_test) + ' images for hyper param evaluation\n'
            # Trials object: results report
            trials = Trials()
            # Documentation: https://github.com/hyperopt/hyperopt/wiki/FMin
            best = fmin(hyperopt_train_test, model_hyperparam_hyperopt, algo=self.param_hyperopt['algo'], max_evals=self.param_hyperopt['nb_eval'], trials=trials)

            # Save results
            pickle.dump(trials.trials, open(self.results_path + self.model_name + '_trials.pkl', "wb"))
            print '\n... End of Hyperopt!\n'
        
        else:
            print ' '
            print 'Please provide a hyper parameter dict (called \'model_hyperparam\') in your classifier_model dict'
            print ' '

    def set_hyperopt_train(self, coord_train, label_train):
    ###############################################################################################################
    #
    # Find better score: save related trials_*.pkl file as trials_best.pkl
    # 
    # Set optimized params to self.model and save it as self.model_name + '_opt'
    #
    # Train the model 
    # 
    ###############################################################################################################

        fname_trial = self.results_path + self.model_name + '_trials.pkl'

        with open(fname_trial) as outfile:    
            trial = pickle.load(outfile)
            outfile.close()

        loss_list = [trial[i]['result']['loss'] for i in range(len(trial))]
        eval_time_list = [trial[i]['result']['eval_time'] for i in range(len(trial))]
        trials_best_score = min(loss_list)
        total_hyperopt_time = sum(eval_time_list)
        idx_best_params = loss_list.index(min(loss_list))
        best_params = trial[idx_best_params]['misc']['vals']
        model_hyperparam_opt = {}
        for k in self.model_hyperparam.keys():
            if len(self.model_hyperparam[k]) == 1:
                model_hyperparam_opt[k] = self.model_hyperparam[k][0]
            else:
                if isinstance(best_params[k][0], int):
                    model_hyperparam_opt[k] = self.model_hyperparam[k][best_params[k][0]]
                else:
                    model_hyperparam_opt[k] = float(best_params[k][0])

        print 'Hyperopt best score: ' + str(round(-trials_best_score,3))
        print 'Total hyperopt time: ' + str(round(total_hyperopt_time,3)) + 's'
        print 'Best hyper params: ' + str(model_hyperparam_opt)
        
        self.model.set_params(model_hyperparam_opt)
            
        if self.param_training['minibatch_size_train'] is not None:
            train_minibatch_size = self.param_training['minibatch_size_train']
        else:
            train_minibatch_size = sum([len(coord_train[str(i)]) for i in coord_train]) # Define minibatch size used for training

        # Create minibatch iterators for training
        minibatch_iterator_train = self.iter_minibatches_trainer(coord_train, label_train, 
                                                                    train_minibatch_size, self.training_dataset)

        stats = {'n_train': 0, 'n_train_pos': 0,
                't0': time.time(), 'total_fit_time': 0.0}
        
        print '\nStarting Training...'
        print '... with ' + str(len(coord_train)) + ' images for training\n'
        
        for n_epoch in range(self.param_training['number_of_epochs']):
            for data in minibatch_iterator_train:
                X_train = np.array(data['patches_feature'])
                y_train = np.array(data['patches_label'])

                stats['n_train'] += X_train.shape[0]
                stats['n_train_pos'] += sum(y_train)

                self.model.train(X_train, y_train)

        stats['total_fit_time'] = time.time() - stats['t0']

        print progress(stats)

        self.model.save(self.model_path + self.model_name + '_train')
        pickle.dump(stats, open(self.results_path + self.model_name + '_train.pkl', "wb"))

        print '...End of Training!\n'


    def predict(self, coord_test, label_test):

        self.model.load(self.model_path + self.model_name + '_train')
        self.run_prediction(coord_test, label_test, fname_out='', stats=None)


    def run_prediction(self, coord_test, label_test, fname_out='', stats=None):
    ###############################################################################################################
    #
    # TODO:     Check CNN compatibility
    #           Commented lines: used in msct_keras_classification
    # 
    ###############################################################################################################

        if self.param_training['minibatch_size_test'] is not None:
            minibatch_size_test = self.param_training['minibatch_size_test']
        else:
            minibatch_size_test = sum([len(coord_test[str(i)]) for i in coord_test])

        if self.param_training['minibatch_size_train'] is not None: # For HyperOpt
            minibatch_size_train = self.param_training['minibatch_size_train']
        else:
            minibatch_size_train = sum([len(coord_test[str(i)]) for i in coord_test])

        if stats is None:
            # Used for Prediction on Testing dataset
            minibatch_iterator_test = self.iter_minibatches_trainer(coord_test, label_test, 
                                                            minibatch_size_test, self.testing_dataset)
            stats = {'n_test': 0, 'n_test_pos': 0,
                    'accuracy': 0.0, 'precision': 0.0, 'recall': 0.0, 'roc': 0.0,
                    'total_predict_time': 0.0}

            fname_out_progress = self.results_path + self.model_name + '_test.pkl'

            print '\nStarting Testing...'
            print '... with ' + str(len(coord_test)) + ' images for testing\n'

        else:
            # Used for Hyperopt
            minibatch_iterator_test = self.iter_minibatches_trainer(coord_test, label_test, 
                                                            minibatch_size_train, self.training_dataset)
            fname_out_progress = self.results_path + self.model_name + '_eval_' + str(fname_out[0]).zfill(6) + '_' + str(fname_out[1]).zfill(12) + '.pkl'

        
        y_pred, y_test = [], []

        for data_test in minibatch_iterator_test:
            X_test = data_test['patches_feature']
            y_test_cur = data_test['patches_label']

            tick = time.time()
            y_pred_cur = self.model.predict(X_test)
            stats['total_predict_time'] += time.time() - tick
            y_pred.extend(y_pred_cur)
            y_test.extend(y_test_cur)
            stats['n_test'] += X_test.shape[0]
            stats['n_test_pos'] += sum(y_test_cur)

        y_test = np.array(y_test)
        y_pred = np.array(y_pred)

        fpr, tpr, thresholds = roc_curve(y_test, y_pred[:,1], pos_label=1)

        roc_coord = np.column_stack((fpr, tpr))
        roc_dist = [math.hypot(roc_coord[i,0] - 0.0, roc_coord[i,1] - 1.0) for coord in range(roc_coord.shape[0])]
        thresholds = list(thresholds)
        best_treshold = thresholds[roc_dist.index(min(roc_dist))]
        y_pred = [1 if y_pred[j,1] >= best_treshold else 0 for j in range(y_pred.shape[0])]

        print sum(y_pred)

        stats['accuracy'] = accuracy_score(y_test, y_pred)
        stats['precision'] = precision_score(y_test, y_pred)
        stats['recall'] = recall_score(y_test, y_pred)
        stats['roc'] = roc_auc_score(y_test, y_pred)


        plt.figure()
        plt.plot(fpr, tpr, color='darkorange',
                 lw=2, label='ROC curve (area = %0.2f)' % roc_auc_score(y_test, y_pred))
        plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver operating characteristic example')
        plt.legend(loc="lower right")
        plt.show()
        

        if 'accuracy_history' in stats:
            acc_history = (stats['accuracy'], stats['n_train'])
            stats['accuracy_history'].append(acc_history)

            precision_history = (stats['precision'], stats['n_train'])
            stats['precision_history'].append(precision_history)

            recall_history = (stats['recall'], stats['n_train'])
            stats['recall_history'].append(recall_history)

            roc_history = (stats['roc'], stats['n_train'])
            stats['roc_history'].append(roc_history)

        print progress(stats)

        pickle.dump(stats, open(fname_out_progress, "wb"))

        return y_test, y_pred