#!/usr/bin/env python
#########################################################################################
#
# This module is used to run msct_machine_learning.py
# run THEANO_FLAGS=mode=FAST_RUN,device=gpu1,floatX=float32 python train_patch_based_cnn.py
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
import numpy as np
import json
import pickle
from skimage.feature import hog
from hyperopt import tpe
from sklearn.metrics import roc_auc_score, precision_score, recall_score, accuracy_score
from sklearn.base import BaseEstimator
from sklearn.preprocessing import StandardScaler
from sklearn import svm

########## Change path
import sys
sys.path.insert(0, '/Users/chgroc/spinalcordtoolbox/scripts')
from msct_machine_learning import Trainer, FileManager

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
                if 'seg' in fname_seg or 'gmseg' in fname_seg:
                    if fname_im[:-7] in fname_seg:
                        f_seg = fname_seg
            list_data.append([[fname_im], [f_seg]])

    return list_data

def center_of_patch_equal_one(data):
    patch_size_x, patch_size_y = data['patches_gold'].shape[2], data['patches_gold'].shape[3]
    return np.squeeze(data['patches_gold'][:, 0, int(patch_size_x / 2), int(patch_size_y / 2)])

#path_data = '/Users/neuropoly/data/data_t1/'
path_data = '/Users/neuropoly/data/data_t2s/'
#path_data = '/Users/benjamindeleener/data/data_augmentation/data_t2s/'
#path_data = '/Volumes/data_processing/bdeleener/machinelearning/data_t2s/'

#path_output = '/Users/neuropoly/data/filemanager_t1/'
path_output = '/Users/neuropoly/data/filemanager_t2s/'
#path_output = '/Users/benjamindeleener/data/data_augmentation/filemanager_t2s/'
#path_output = '/Volumes/data_processing/bdeleener/machine_learning/filemanager_t2s/'


my_file_manager = FileManager(dataset_path=path_data,
                              fct_explore_dataset=extract_list_file_from_path,
                              patch_extraction_parameters={'ratio_dataset': [0.9, 0.1],
                                                           'ratio_patches_voxels': 0.1,
                                                           'patch_size': [32, 32],
                                                           'patch_pixdim': {'axial': [1.0, 1.0]},
                                                           'extract_all_positive': True,
                                                           'extract_all_negative': False,
                                                           'batch_size': 200},
                              fct_groundtruth_patch=center_of_patch_equal_one,
                              path_output=path_output)

training_dataset, testing_dataset = my_file_manager.decompose_dataset()
my_file_manager.explore()
