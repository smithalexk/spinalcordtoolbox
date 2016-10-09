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


class FileManager(object):
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

        self.cpu_number = mp.cpu_count()

    def iter_minibatches(self, patch_iter, minibatch_size):
        """Generator of minibatches."""
        data = get_minibatch(patch_iter, minibatch_size)
        while len(data['patches_raw']):
            yield data
            data = get_minibatch(patch_iter, minibatch_size)

    def decompose_dataset(self, path_output):
        array_indexes = range(self.number_of_images)
        np.random.shuffle(array_indexes)

        self.training_dataset = self.list_files[np.ix_(array_indexes[:int(self.ratio_dataset[0] * self.number_of_images)])]
        self.testing_dataset = self.list_files[np.ix_(array_indexes[int(self.ratio_dataset[0] * self.number_of_images):])]

        results = {
            'training': {'raw_images': [data[0].tolist() for data in self.training_dataset], 'gold_images': [data[1].tolist() for data in self.training_dataset]},
            'testing': {'raw_images': [data[0].tolist() for data in self.testing_dataset], 'gold_images': [data[1].tolist() for data in self.testing_dataset]},
            'dataset_path': self.dataset_path
        }

        with bz2.BZ2File(path_output + 'datasets.pbz2', 'w') as f:
            pickle.dump(results, f)

        return self.training_dataset, self.testing_dataset

    def compute_patches_coordinates(self, image):
        if self.extract_all_negative or self.extract_all_positive:
            print 'Extract all negative/positive patches: feature not yet ready...'

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
                    labels.extend(center_of_patch_equal_one(data))
                    number_done += data['patches_gold'].shape[0]
                    pbar.update(number_done)
            pbar.finish()

            return [i, patches_coordinates, labels]

        except KeyboardInterrupt:
            return

        except Exception as e:
            print 'Error on line {}'.format(sys.exc_info()[-1].tb_lineno)
            raise e

    def explore(self):
        # training dataset
        global_results_patches = {'patch_info': self.patch_info}

        # TRAINING DATASET
        results_training = {}
        classes_training = {}

        pool = mp.Pool(processes=self.cpu_number)
        results = pool.map(self.worker_explore, itertools.izip(range(len(self.training_dataset)), itertools.repeat(self.training_dataset)))

        pool.close()
        try:
            pool.join()  # waiting for all the jobs to be done
            for result in results:
                i = result[0]
                patches_coordinates = result[1]
                labels = result[2]
                results_training[str(i)] = [[patches_coordinates[j, :].tolist(), labels[j]] for j in range(len(labels))]

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

        pool = mp.Pool(processes=self.cpu_number)
        results = pool.map(self.worker_explore, itertools.izip(range(len(self.testing_dataset)), itertools.repeat(self.testing_dataset)))

        pool.close()
        try:
            pool.join()  # waiting for all the jobs to be done
            for result in results:
                i = result[0]
                patches_coordinates = result[1]
                labels = result[2]
                results_testing[str(i)] = [[patches_coordinates[j, :].tolist(), labels[j]] for j in range(len(labels))]

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

        global_results_patches['testing'] = results_testing

        count_max_class, max_class = 0, ''
        for cl in classes_testing:
            if classes_testing[cl][0] > count_max_class:
                max_class = cl
        for cl in classes_testing:
            classes_testing[cl][1] = classes_testing[cl][0] / float(classes_testing[max_class][0])

        global_results_patches['statistics'] = {'classes_training': classes_training, 'classes_testing': classes_testing}

        with bz2.BZ2File(path_output + 'patches.pbz2', 'w') as f:
            pickle.dump(global_results_patches, f)


#########################################
# USE CASE
#########################################
def extract_list_file_from_path(path_data):
    ignore_list = ['.DS_Store']
    sct.printv('Extracting ' + path_data)
    cr = '\r'

    list_data = []
    for root, dirs, files in os.walk(path_data):
        for fname_im in files:
            if fname_im in ignore_list:
                continue
            if 'seg' in fname_im or 'gmseg' in fname_im:
                continue
            f_seg = None
            for fname_seg in files:
                if fname_im[:-7] in fname_seg:
                    f_seg = fname_seg
            list_data.append([[fname_im], [f_seg]])

    return list_data


def center_of_patch_equal_one(data):
    patch_size_x, patch_size_y = data['patches_gold'].shape[2], data['patches_gold'].shape[3]
    return np.squeeze(data['patches_gold'][:, 0, int(patch_size_x / 2), int(patch_size_y / 2)])



class Model(object):
    def __init__(self, fname):
        self.fname = fname

    def load(self):
        pass

    def save(self, fname_out):
        pass

    def train(self):
        pass

    def predict(self):
        return


from keras.models import Sequential
from keras.layers.core import Dense, Activation, Dropout, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.optimizers import SGD, Adadelta
from keras.utils import np_utils

class KerasConvNet(Sequential):
    def __init__(self):
        self.add(Convolution2D(32, 3, 3, border_mode='valid', input_shape=(1, patch_size, patch_size)))
        self.add(Activation('relu'))
        self.add(Convolution2D(32, 3, 3))
        self.add(Activation('relu'))
        self.add(MaxPooling2D(pool_size=(2, 2)))
        self.add(Dropout(0.25))

        self.add(Convolution2D(64, 3, 3, border_mode='valid'))
        self.add(Activation('relu'))
        self.add(Convolution2D(64, 3, 3))
        self.add(Activation('relu'))
        self.add(MaxPooling2D(pool_size=(2, 2)))
        self.add(Dropout(0.25))

        self.add(Flatten())
        self.add(Dense(256))
        self.add(Activation('relu'))
        self.add(Dropout(0.5))

        self.add(Dense(2))
        self.add(Activation('softmax'))

        ada = Adadelta(lr=0.1, rho=0.95, epsilon=1e-08)
        self.compile(loss='categorical_crossentropy', optimizer=ada)

    def save(self, fname):
        pass

    def train(self):
        pass

    def predict(self):
        pass




my_file_manager = FileManager(dataset_path='/Users/benjamindeleener/data/data_augmentation/test_very_small/',
                              fct_explore_dataset=extract_list_file_from_path,
                              patch_extraction_parameters={'ratio_dataset': [0.8, 0.2],
                                                           'ratio_patches_voxels': 0.1,
                                                           'patch_size': [32, 32],
                                                           'patch_pixdim': {'axial': [1.0, 1.0]},
                                                           'extract_all_positive': False,
                                                           'extract_all_negative': False,
                                                           'batch_size': 500},
                              fct_groundtruth_patch=None)

path_output = '/Users/benjamindeleener/data/data_augmentation/'

training_dataset, testing_dataset = my_file_manager.decompose_dataset(path_output)
my_file_manager.explore()

