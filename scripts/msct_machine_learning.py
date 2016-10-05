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

        # this function will be called on each patch to know its class/label
        self.fct_groundtruth_patch = fct_groundtruth_patch

        self.list_files = np.array(self.fct_explore_dataset(self.dataset_path))
        self.number_of_images = len(self.list_files)

        self.training_dataset, self.testing_dataset, self.validation_dataset = [], [], []

        # list_classes is a dictionary that contains all the classes that are present in the dataset
        # this list is filled up iteratively while exploring the dataset
        # the key is the label of the class and the element is the number of element of each class
        self.list_classes = {}

        # class_weights is a dictionary containing the ratio of each class and the most represented class
        # len(class_weights) = len(list_classes)
        self.class_weights = {}

    def decompose_dataset(self):
        array_indexes = range(self.number_of_images)
        np.random.shuffle(array_indexes)

        self.training_dataset = self.list_files[np.ix_(array_indexes[:int(self.ratio_dataset[0]*self.number_of_images)])]
        self.testing_dataset = self.list_files[np.ix_(array_indexes[int(self.ratio_dataset[0] * self.number_of_images):int((self.ratio_dataset[0] + self.ratio_dataset[1]) * self.number_of_images)])]
        self.validation_dataset = self.list_files[np.ix_(array_indexes[int((self.ratio_dataset[0] + self.ratio_dataset[1]) * self.number_of_images):])]

        return self.training_dataset, self.testing_dataset, self.validation_dataset

    def compute_patches_coordinates(self, image_dim, patch_size, patch_pixdim):
        return 


    def extract_patches_from_image(self, image_file, patches_coordinates, patch_size=32, slice_of_interest=None, verbose=1):
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
            grid_physical = np.mgrid[ind_phys[0] - patch_size / 2:ind_phys[0] + patch_size / 2,
                            ind_phys[1] - patch_size / 2:ind_phys[1] + patch_size / 2]
            # 3. transform grid to voxel coordinates
            coord_x = grid_physical[0, :, :].ravel()
            coord_y = grid_physical[1, :, :].ravel()
            coord_physical = [[coord_x[i], coord_y[i], ind_phys[2]] for i in range(len(coord_x))]
            grid_voxel = np.array(image_file.transfo_phys2continuouspix(coord_physical))
            np.set_printoptions(threshold=np.inf)
            # 4. interpolate image on the grid, deal with edges
            patch = np.reshape(image_file.get_values(np.array([grid_voxel[:, 0], grid_voxel[:, 1], grid_voxel[:, 2]]),
                                                     interpolation_mode=1), (patch_size, patch_size))

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

    def explore(self):
        # training dataset
        for i, fnames in enumerate(self.training_dataset):
            fname_image = self.training_dataset[i][0]
            fname_seg = self.training_dataset[i][1]
            im_image = Image(fname_image)
            nx, ny, nz, nt, px, py, pz, pt = im_image.dim

            data_im, data_seg = extract_slices_from_image(list_data[i][0], list_data[i][1])

            for data in :


        return



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
            list_data.append([root + fname_im, root + f_seg])

    return list_data


my_file_manager = FileManager(dataset_path='/Users/benjamindeleener/data/data_augmentation/large_nobrain_nopad/',
                              fct_explore_dataset=extract_list_file_from_path,
                              patch_extraction_parameters={'ratio_dataset': [0.6, 0.2, 0.2],
                                                           'patch_size': [32, 32],
                                                           'patch_pixdim': [1.0, 1.0],
                                                           'extract_all_positive': True,
                                                           'extract_all_negative': False},
                              fct_groundtruth_patch=None)

training_dataset, testing_dataset, validation_dataset = my_file_manager.decompose_dataset()
print training_dataset
print testing_dataset
print validation_dataset

