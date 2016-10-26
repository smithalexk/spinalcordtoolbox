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

# TODO: Rajouter center_of_patch_equal_one dans user case
#       Mettre a jour File Manager + fct au dessus
#       Faire 2 fichiers dans dev: CNN et SVM

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


from keras.models import Sequential
from keras.layers.core import Dense, Activation, Dropout, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.optimizers import SGD, Adadelta
from keras.utils import np_utils

class KerasConvNet(Sequential):
    def __init__(self, params):
        super(KerasConvNet, self).__init__()

        if 'patch_size' in params:
            self.patch_size = params['patch_size']  # must be a list of two elements
        else:
            self.patch_size = [32, 32]

        if 'number_of_channels' in params:
            self.number_of_channels = params['number_of_channels']
        else:
            self.number_of_channels = 1

        if 'number_of_classes' in params:
            self.number_of_classes = params['number_of_classes']
        else:
            self.number_of_classes = 2

        if 'batch_size' in params:
            self.batch_size = params['batch_size']
        else:
            self.batch_size = 256

        if 'weight_class' in params:
            self.weight_class = params['weight_class']
        else:
            self.weight_class = [1.0, 1.0]

        # must be a list of list corresponding to the number of layers, depth and features.
        # For example: [[32, 32], [64, 64]] means that there are two depth,
        # two layers per depth and 32 and 64 features for each layer/depth, respectively
        if 'depth_layers_features' in params:
            self.number_of_layer_per_depth = params['number_of_features']
        else:
            self.number_of_layer_per_depth = [[32, 32], [64, 64]]

        if 'number_of_feature_dense' in params:
            self.number_of_feature_dense = params['number_of_feature_dense']
        else:
            self.number_of_feature_dense = 256

        if 'activation_function' in params:
            self.activation_function = params['activation_function']
        else:
            self.activation_function = 'relu'

        if 'loss' in params:
            self.loss = params['loss']
        else:
            self.loss = 'categorical_crossentropy'

        #self.create_model()

    def create_model(self):
        for d in range(len(self.number_of_layer_per_depth)):
            for l in range(len(self.number_of_layer_per_depth[d])):
                if d == 0 and l == 0:
                    self.add(Convolution2D(self.number_of_layer_per_depth[d][l], 3, 3, border_mode='valid', input_shape=(self.number_of_channels, self.patch_size[0], self.patch_size[1]), name='input_layer'))
                elif d != 0 and l == 0:
                    self.add(Convolution2D(self.number_of_layer_per_depth[d][l], 3, 3, border_mode='valid', name='conv_'+str(d)+'_'+str(l)))
                else:
                    self.add(Convolution2D(self.number_of_layer_per_depth[d][l], 3, 3, name='conv_'+str(d)+'_'+str(l)))
                self.add(Activation(self.activation_function, name='activation_'+str(d)+'_'+str(l)))
            self.add(MaxPooling2D(pool_size=(2, 2), name='max-pooling_'+str(d)))
            self.add(Dropout(0.25, name='dropout_'+str(d)))

        self.add(Flatten(name='flatten'))
        self.add(Dense(self.number_of_feature_dense, name='dense_before_final'))
        self.add(Activation(self.activation_function, name='activation_final'))
        self.add(Dropout(0.5, name='dropout_final'))

        self.add(Dense(self.number_of_classes, name='dense_final'))
        self.add(Activation('softmax', name='softmax_activation'))

        ada = Adadelta(lr=1.0, rho=0.95, epsilon=1e-08)
        self.compile(loss=self.loss, optimizer=ada)

    def save(self, fname_out):
        super(KerasConvNet, self).save(fname_out + '_model.h5')
        self.save_weights(fname_out + '_weights.h5')

    def load(self, fname_in):
        self.load_weights(fname_in + '.h5')

    def train(self, X, y):
        if X.ndim == 3:
            X = np.expand_dims(X, axis=1)
        y = np_utils.to_categorical(y, nb_classes=self.number_of_classes)
        self.train_on_batch(X, y, class_weight=self.weight_class)

    def predict(self, X):
        if X.ndim == 3:
            X = np.expand_dims(X, axis=1)
        y_pred = super(KerasConvNet, self).predict(X, batch_size=self.batch_size)
        return y_pred

    def set_params(self, params):
        if 'depth_layers_features' in params:
            self.number_of_layer_per_depth = params['number_of_features']
        if 'number_of_feature_dense' in params:
            self.number_of_feature_dense = params['number_of_feature_dense']
        if 'activation_function' in params:
            self.activation_function = params['activation_function']
        if 'loss' in params:
            self.loss = params['loss']
        """self.layers = []
        self.outputs = []
        self.inbound_nodes = []
        self.outbound_nodes = []
        self.built = False
        self._flattened_layers = None"""
        self.create_model()

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


def extract_patch_feature(im, param=None):
    return im


def center_of_patch_equal_one(data):
    patch_size_x, patch_size_y = data['patches_gold'].shape[2], data['patches_gold'].shape[3]
    return np.squeeze(data['patches_gold'][:, 0, int(patch_size_x / 2), int(patch_size_y / 2)])

"""
my_file_manager = FileManager(dataset_path='/Volumes/folder_shared-1/benjamin/machine_learning/patch_based/large_nobrain_nopad/',
                              fct_explore_dataset=extract_list_file_from_path,
                              patch_extraction_parameters={'ratio_dataset': [0.9, 0.1],
                                                           'ratio_patches_voxels': 0.1,
                                                           'patch_size': [32, 32],
                                                           'patch_pixdim': {'axial': [1.0, 1.0]},
                                                           'extract_all_positive': False,
                                                           'extract_all_negative': False,
                                                           'batch_size': 200},
                              fct_groundtruth_patch=center_of_patch_equal_one)
"""

# my_file_manager.extract_all_positive = True
# training_dataset, testing_dataset = my_file_manager.decompose_dataset(model_path)
# my_file_manager.explore()

results_path = '/home/neuropoly/data/result_new_pipeline_large/'
model_path = '/home/neuropoly/data/model_new_pipeline_large/'
# data_path = '/Users/chgroc/data/spine_detection/data/'
data_filemanager_path = '/home/neuropoly/data/filemanager_large_nobrain_nopad/'

params_cnn = {'patch_size': [32, 32],
              'number_of_channels': 1,
              'batch_size': 128,
              'number_of_features': [[32, 32], [64, 64]],
              'loss': 'categorical_crossentropy'
             }

cnn_model = {'model_name': 'CNN', 'model': KerasConvNet(params_cnn),
             'model_hyperparam': {'class_weight': ('balanced')}}

methode_normalization_1={'methode_normalization_name':'histogram', 'param':{'cutoffp': (1, 99), 'landmarkp': [10, 20, 30, 40, 50, 60, 70, 80, 90], 'range': [0, 255]}}
methode_normalization_2={'methode_normalization_name':'percentile', 'param':{'range': [0, 255]}}

param_training = {'data_path_local': '/home/neuropoly/data/large_nobrain_nopad/',
                  'number_of_epochs': 10, 'patch_size': [32, 32], 'ratio_patch_per_img': 1.0,
                  'minibatch_size_train': 256, 'minibatch_size_test': 256,  # number for CNN, None for SVM
                  'hyperopt': {'algo': tpe.suggest,  # Grid Search algorithm
                               'nb_eval': 10,  # Nb max of param test
                               'fct': roc_auc_score,  # Objective function
                               'eval_factor': 10000,  # Evaluation rate
                               'ratio_dataset_eval': 0.25,
                               # Ratio of training dataset dedicated to hyperParam validation
                               'ratio_img_eval': 0.25,  # Ratio of patch per validation image
                               'ratio_img_train': 1.0}}

### Attention .json et .pbz2 : modif a faire dans Trainer.__init__
my_trainer = Trainer(data_filemanager_path=data_filemanager_path,
                     datasets_dict_fname='datasets.pbz2',
                     patches_dict_prefixe='patches_coordinates_',
                     patches_pos_dict_prefixe='patches_coordinates_positives_',
                     classifier_model=cnn_model,
                     fct_feature_extraction=extract_patch_feature,
                     param_training=param_training,
                     results_path=results_path,
                     model_path=model_path)

#coord_prepared_train, label_prepared_train = my_trainer.prepare_patches(my_trainer.fname_training_raw_images, 1.0)
coord_prepared_test, label_prepared_test = my_trainer.prepare_patches(my_trainer.fname_testing_raw_images, 1.0)

#my_trainer.hyperparam_optimization(coord_prepared_train, label_prepared_train)
# my_trainer.set_hyperopt_train(coord_prepared_train, label_prepared_train)
my_trainer.model.load(model_path + XXX)
my_trainer.run_prediction(coord_prepared_test, label_prepared_test, fname_out='', stats=None)