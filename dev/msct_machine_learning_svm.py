#!/usr/bin/env python
#########################################################################################
#
# This module is used to run msct_machine_learning.py
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
from sklearn import svm
from sklearn.externals import joblib

########## Change path
import sys
sys.path.insert(0, '/Users/chgroc/spinalcordtoolbox/scripts')
from msct_machine_learning import Trainer, FileManager

class Model(object):
    def __init__(self, params):
        self.params = params

    def load(self, fname_in):
        pass

    def save(self, fname_out):
        pass

    def train(self, X, y):
        pass

    def predict(self, X):
        return

    def set_params(self, params):
        pass

class Classifier_svm(BaseEstimator):
    def __init__(self, params={'kernel': 'rbf', 'C': 1.0}):

        self.clf = svm.SVC()
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
        print self.params

    def set_params(self, params):
        self.clf.set_params(**params)
        self.params = params

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


def extract_hog_feature(im, param=None):

    if param is None:
        param = {'orientations': 8, 'pixels_per_cell': [6, 6], 'cells_per_block': [3,3],
                'visualize': False, 'transform_sqrt': True}

    hog_feature = np.array(hog(image = im, orientations=param['orientations'],
                pixels_per_cell=param['pixels_per_cell'], cells_per_block=param['cells_per_block'],
                transform_sqrt=param['transform_sqrt'], visualise=param['visualize']))

    return hog_feature

def extract_patch_feature(im, param=None):

    return im

def center_of_patch_equal_one(data):
    patch_size_x, patch_size_y = data['patches_gold'].shape[2], data['patches_gold'].shape[3]
    return np.squeeze(data['patches_gold'][:, 0, int(patch_size_x / 2), int(patch_size_y / 2)])


# my_file_manager = FileManager(dataset_path=path_input,
#                               fct_explore_dataset=extract_list_file_from_path,
#                               patch_extraction_parameters={'ratio_dataset': [0.8, 0.2],
#                                                            'ratio_patches_voxels': 0.1,
#                                                            'patch_size': [32, 32],
#                                                            'patch_pixdim': {'axial': [1.0, 1.0]},
#                                                            'extract_all_positive': True,
#                                                            'extract_all_negative': False,
#                                                            'batch_size': 500},
#                               fct_groundtruth_patch=center_of_patch_equal_one,
#                               path_output=path_output)

# training_dataset, testing_dataset = my_file_manager.decompose_dataset()
# my_file_manager.explore()

results_path = '/Users/chgroc/data/spine_detection/results/'
model_path = '/Users/chgroc/data/spine_detection/model/'
# data_path = '/Users/chgroc/data/spine_detection/data/'
data_filemanager_path = '/Volumes/data_processing/bdeleener/machine_learning/filemanager_large_nobrain_nopad/'

svm_model = {'model_name': 'SVM', 'model': Classifier_svm(svm.SVC),
            'model_hyperparam':{'C': [1, 1000],
                                'kernel': ('sigmoid', 'poly', 'rbf'),
                                'gamma': [0, 20],
                                'class_weight': (None, 'balanced')}}

linear_svm_model = {'model_name': 'LinearSVM', 'model': Classifier_svm(svm.SVC),
                    'model_hyperparam':{'C': [1, 1000],
                                        'kernel': 'linear',
                                        'probability': True,
                                        'class_weight': ('balanced')}}

param_training = {'data_path_local': '/Volumes/data_processing/bdeleener/machine_learning/large_nobrain_nopad/',
                    'number_of_epochs': 1, 'patch_size': [32, 32],
                    'minibatch_size_train': None, # number for CNN, None for SVM
                    'minibatch_size_test': 1000,
                    'hyperopt': {'algo':tpe.suggest, 'nb_eval':20, 'fct': recall_score, 'eval_factor': 1, 'ratio_eval':0.5}}

my_trainer = Trainer(data_filemanager_path = data_filemanager_path,
                    datasets_dict_fname = 'datasets.pbz2',
                    patches_dict_prefixe = 'patches_coordinates_', 
                    patches_pos_dict_prefixe = 'patches_coordinates_positives_', 
                    classifier_model=linear_svm_model,
                    fct_feature_extraction=extract_hog_feature, 
                    param_training=param_training, 
                    results_path=results_path, model_path=model_path)

coord_prepared_train, label_prepared_train = my_trainer.prepare_patches(my_trainer.fname_training_raw_images, [0.001, 0.001])
# coord_prepared_test, label_prepared_test = my_trainer.prepare_patches(my_trainer.fname_testing_raw_images, [1.0, 1.0])

my_trainer.hyperparam_optimization(coord_prepared_train, label_prepared_train)
my_trainer.set_hyperopt_train(coord_prepared_train, label_prepared_train)
# my_trainer.predict(coord_prepared_test, label_prepared_test)