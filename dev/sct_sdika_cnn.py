import bz2
import sct_utils as sct
import pickle
import os
import nibabel as nib
import shutil

from msct_image import Image
import numpy as np
from math import sqrt

from collections import Counter
import random

import json
import bz2

from sortedcontainers import SortedListWithKey
from operator import itemgetter

import time
import argparse

import seaborn as sns
import matplotlib.pyplot as plt


TODO_STRING = """\n
                - sct_image RPI sur les 25 images
                - sct_convert en .img
                - envoyer le folder .img chez ferguson
                - faire tourner spine-detect-cnn
                - pull les resultats sur evans
                - compute les metrics de validation
                - idem sur le *_cnn_2
                - comparer les resultats: choisir le meilleur pour la suite
                - faire tourner cnn sur tout le dataset
                \n
            """


def create_folders_local(folder2create_lst):

    for folder2create in folder2create_lst:
        if not os.path.exists(folder2create):
            os.makedirs(folder2create)

def find_img_testing(path_large, contrast):

    center_lst, pathology_lst, path_img, path_seg = [], [], [], []
    for subj_fold in os.listdir(path_large):
        path_subj_fold = path_large + subj_fold + '/'
        if os.path.isdir(path_subj_fold):
            contrast_fold_lst = [contrast_fold for contrast_fold in os.listdir(path_subj_fold) if os.path.isdir(path_subj_fold+contrast_fold+'/')]
            contrast_fold_lst_oI = [contrast_fold for contrast_fold in contrast_fold_lst 
                                                                            if contrast_fold==contrast or contrast_fold.startswith(contrast+'_')]
            if len(contrast_fold_lst_oI):
                with open(path_subj_fold+'dataset_description.json') as data_file:    
                    data_description = json.load(data_file)
                    data_file.close()

                if len(contrast_fold_lst_oI)>1:
                    ax_candidates = [tt for tt in contrast_fold_lst_oI if 'ax' in tt]
                    if len(ax_candidates):
                        contrast_fold_oI = ax_candidates[0]
                    else:
                        contrast_fold_oI = contrast_fold_lst_oI[0]                                               
                else:
                    contrast_fold_oI = contrast_fold_lst_oI[0]

                path_contrast_fold = path_subj_fold+contrast_fold_oI+'/'
                if os.path.exists(path_contrast_fold+'segmentation_description.json'):
                    with open(path_contrast_fold+'segmentation_description.json') as data_file:    
                        data_seg_description = json.load(data_file)
                        data_file.close()
                    if len(data_seg_description['cord']):
                        path_img_cur = path_contrast_fold+contrast_fold_oI+'.nii.gz'
                        path_seg_cur = path_contrast_fold+contrast_fold_oI+'_seg_manual.nii.gz'
                        if os.path.exists(path_img_cur) and os.path.exists(path_seg_cur):
                            path_img.append(path_img_cur)
                            path_seg.append(path_seg_cur)
                            center_lst.append(data_description['Center'])
                            pathology_lst.append(data_description['Pathology'])
                        else:
                            print '\nIMG or SEG falta: ' + path_contrast_fold
                            print path_img_cur

    center_lst = list(set(center_lst))
    pathology_lst = [patho for patho in pathology_lst if patho != "" and patho != "HC"]
    pathology_dct = {x:pathology_lst.count(x) for x in pathology_lst}

    print '# of Centers: ' + str(len(center_lst))
    print 'Centers: ' + ', '.join(center_lst)
    print pathology_dct
    print '# of Subjects: ' + str(len(path_img))

    return path_img, path_seg

def transform_nii_img(img_lst, path_out):

    path_img2convert = []
    for img_path in img_lst:
        path_cur = img_path
        path_cur_out = path_out + '_'.join(img_path.split('/')[5:7]) + '.nii.gz'
        if not os.path.isfile(path_cur_out):
            shutil.copyfile(path_cur, path_cur_out)
            os.system('sct_image -i ' + path_cur_out + ' -type int16 -o ' + path_cur_out)
            os.system('sct_image -i ' + path_cur_out + ' -setorient RPI -o ' + path_cur_out)
        path_img2convert.append(path_cur_out)

    return path_img2convert

def transform_nii_seg(seg_lst, path_out, path_gold):

    path_seg2convert = []
    for seg_path in seg_lst:
        path_cur = seg_path
        path_cur_out = path_out + '_'.join(seg_path.split('/')[5:7]) + '_seg.nii.gz'
        if not os.path.isfile(path_cur_out):
            shutil.copyfile(path_cur, path_cur_out)
            os.system('sct_image -i ' + path_cur_out + ' -setorient RPI -o ' + path_cur_out)

        path_cur_ctr = path_cur_out.split('.')[0] + '_centerline.nii.gz'
        if not os.path.isfile(path_cur_ctr):
            os.chdir(path_out)
            os.system('sct_process_segmentation -i ' + path_cur_out + ' -p centerline -ofolder ' + path_out)
            os.system('sct_image -i ' + path_cur_ctr + ' -type int16')
            path_input_header = path_cur_out.split('_seg')[0] + '.nii.gz'
            os.system('sct_image -i ' + path_input_header + ' -copy-header ' + path_cur_ctr)

        path_cur_gold = path_gold + '_'.join(seg_path.split('/')[5:7]) + '_centerline_gold.nii.gz'
        if not os.path.isfile(path_cur_gold) and os.path.isfile(path_cur_ctr):
            shutil.copyfile(path_cur_ctr, path_cur_gold)

        if os.path.isfile(path_cur_out):
            path_seg2convert.append(path_cur_out)
        if os.path.isfile(path_cur_ctr):
            path_seg2convert.append(path_cur_ctr)

    return path_seg2convert


def convert_nii2img(path_nii2convert, path_out):

    fname_img = []
    for img in path_nii2convert:
        path_cur = img
        path_cur_out = path_out + img.split('.')[0].split('/')[-1] + '.img'
        if not img.split('.')[0].split('/')[-1].endswith('_seg') and not img.split('.')[0].split('/')[-1].endswith('_seg_centerline'):
            fname_img.append(img.split('.')[0].split('/')[-1] + '.img')
        if not os.path.isfile(path_cur_out):
            os.system('sct_convert -i ' + path_cur + ' -o ' + path_cur_out)

    return fname_img


def prepare_dataset(path_local, constrast_dct, path_sct_testing_large):

    dataset_dict = {}
    for cc in constrast_dct:
    
        path_local_gold = path_local + 'gold_' + cc + '/'
        path_local_input_nii = path_local + 'input_nii_' + cc + '/'
        path_local_input_img = path_local + 'input_img_' + cc + '/'
        folder2create_lst = [path_local_input_nii, path_local_input_img, path_local_gold]
        create_folders_local(folder2create_lst)

        print '\n\n***************Contrast of Interest: ' + cc + ' ***************'
        path_fname_img, path_fname_seg = find_img_testing(path_sct_testing_large, cc)

        path_img2convert = transform_nii_img(path_fname_img, path_local_input_nii)
        path_seg2convert = transform_nii_seg(path_fname_seg, path_local_input_nii, path_local_gold)
        path_imgseg2convert = path_img2convert + path_seg2convert
        fname_img_lst = convert_nii2img(path_imgseg2convert, path_local_input_img)
        dataset_dict[cc] = fname_img_lst

    pickle.dump(dataset_dict, open(path_local + 'dataset_dict.pkl', "wb"))

def prepare_train(path_local, path_outdoor, cc, nb_img):

    print '\nExperiment: '
    print '... contrast: ' + cc
    print '... nb image used for training: ' + str(nb_img) + '\n'

    path_outdoor_cur = path_outdoor + 'input_img_' + cc + '/'

    path_local_res_img = path_local + 'output_img_' + cc + '_'+ str(nb_img) + '/'
    path_local_res_nii = path_local + 'output_nii_' + cc + '_'+ str(nb_img) + '/'
    path_local_res_pkl = path_local + 'output_pkl_' + cc + '_'+ str(nb_img) + '/'
    path_local_train = path_local + 'input_train_' + cc + '_'+ str(nb_img) + '/'
    folder2create_lst = [path_local_train, path_local_res_img, path_local_res_nii, path_local_res_pkl]

    with open(path_local + 'dataset_dict.pkl') as outfile:    
        data_pkl = pickle.load(outfile)
        outfile.close()

    fname_subj_lst = data_pkl[cc]

    nb_sub_train = int(float(len(fname_subj_lst))/(50*nb_img))+1
    path_folder_sub_train = []
    for i in range(nb_sub_train):
        path2create = path_local_train + str(i).zfill(3) + '/'
        path_folder_sub_train.append(path2create)
        folder2create_lst.append(path2create)

    create_folders_local(folder2create_lst)

    if os.listdir(path2create) == []: 
        path_fname_img_rdn = [f.split('.')[0] for f in fname_subj_lst]    
        random.shuffle(path_fname_img_rdn)
        tuple_fname_multi = []
        for j in range(0, len(fname_subj_lst), nb_img):
            s = path_fname_img_rdn[j:j+nb_img]
            if len(path_fname_img_rdn[j:j+nb_img])==nb_img:
                tuple_fname_multi.append(s)

        for i, tt in enumerate(tuple_fname_multi):
            stg, stg_seg = '', ''
            for tt_tt in tt:
                stg += path_outdoor_cur + tt_tt + '\n'
                stg_seg += path_outdoor_cur + tt_tt + '_seg' + '\n'
            path2save = path_folder_sub_train[int(float(i)/50)]
            with open(path2save + str(i).zfill(3) + '.txt', 'w') as text_file:
                text_file.write(stg)
                text_file.close()
            with open(path2save + str(i).zfill(3) + '_ctr.txt', 'w') as text_file:
                text_file.write(stg_seg)
                text_file.close()

    return path_local_train


def send_data2ferguson(path_local, path_ferguson, cc, nb_img):

    path_local_train_cur = prepare_train(path_local, path_ferguson, cc, nb_img)

    pickle_ferguson = {
                        'contrast': cc,
                        'nb_image_train': nb_img
                        }
    path_pickle_ferguson = path_local + 'ferguson_config.pkl'
    pickle.dump(pickle_ferguson, open(path_pickle_ferguson, "wb"))

    os.system('scp -r ' + path_local + 'input_img_' + contrast_of_interest + '/' + ' ferguson:' + path_ferguson)
    os.system('scp -r ' + path_local_train_cur + ' ferguson:' + path_ferguson)
    os.system('scp ' + path_pickle_ferguson + ' ferguson:' + path_ferguson)



def pull_img_convert_nii_remove_img(path_local, path_ferguson, cc, nb_img):

    path_ferguson_res = path_ferguson + 'output_img_' + cc + '_'+ str(nb_img) + '/'
    path_local_res_img = path_local + 'output_img_' + cc + '_'+ str(nb_img) + '/'
    path_local_res_nii = path_local + 'output_nii_' + cc + '_'+ str(nb_img) + '/'

    # Pull .img results from ferguson
    os.system('scp -r ferguson:' + path_ferguson_res + ' ' + '/'.join(path_local_res_img.split('/')[:-2]) + '/')

    # Convert .img to .nii
    # Remove .img files
    for f in os.listdir(path_local_res_img):
        if not f.startswith('.'):
            path_res_cur = path_local_res_nii + f + '/'
            if not os.path.exists(path_res_cur):
                os.makedirs(path_res_cur)

            training_subj = f.split('__')

            if os.path.isdir(path_local_res_img+f):
                for ff in os.listdir(path_local_res_img+f):
                    if ff.endswith('_ctr.hdr'):

                        path_cur = path_local_res_img + f + '/' + ff
                        path_cur_out = path_res_cur + ff.split('_ctr')[0] + '_centerline_pred.nii.gz'
                        img = nib.load(path_cur)
                        nib.save(img, path_cur_out)

                    elif ff == 'time.txt':
                        os.rename(path_local_res_img + f + '/time.txt', path_local_res_nii + f + '/time.txt')

                os.system('rm -r ' + path_local_res_img + f)



########## NEW
def prepare_dataset_cnn(path_local, cc, path_train_cnn):

    with bz2.BZ2File(path_train_cnn, 'rb') as f:
        datasets_dict = pickle.load(f)
        f.close()

    fname_training_img = datasets_dict['training']['raw_images']
    fname_training_img = [f[0].split(cc)[0] for f in fname_training_img]

    with open(path_local + 'dataset_lst_' + cc + '.pkl') as outfile:    
        data_lst = pickle.load(outfile)
        outfile.close()
    
    fname_training_img_sdika = [f for ff in fname_training_img for f in data_lst if ff in f]
    fname_testing_img = list(set(data_lst)-set(fname_training_img_sdika))

    output_file = open(path_local + 'dataset_lst_cnn_' + cc + '.pkl', 'wb')
    pickle.dump(fname_testing_img, output_file)
    output_file.close()


try:
    from keras.models import Sequential
    from keras.layers.core import Dense, Activation, Dropout, Flatten
    from keras.layers.convolutional import Convolution2D, MaxPooling2D
    from keras.layers.normalization import BatchNormalization
    from keras.optimizers import SGD, Adadelta
    from keras.utils import np_utils
except ImportError:
    pass

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

def extract_patches_from_image(image_file, patches_coordinates, patch_size=32, slice_of_interest=None):
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
        patch = np.reshape(image_file.get_values(np.array([grid_voxel[:, 0], grid_voxel[:, 1], grid_voxel[:, 2]]), interpolation_mode=1, border='reflect'), (patch_size, patch_size))

        if patch.shape[0] == patch_size and patch.shape[1] == patch_size:
            result.append(np.expand_dims(patch, axis=0))
    if len(result) != 0:
        return np.concatenate(result, axis=0)
    else:
        return None

def prediction_cnn(im_data, model, initial_resolution, initial_resize, threshold, patch_size, path_output):

    time_prediction = time.time()

    nx, ny, nz, nt, px, py, pz, pt = im_data.dim
    nb_voxels_image = nx * ny * nz

    # first round of patch prediction
    initial_coordinates_x = range(int(initial_resize[0]*nx), nx-int(initial_resize[0]*nx), initial_resolution[0])
    initial_coordinates_y = range(int(initial_resize[1]*ny), ny-int(initial_resize[1]*ny), initial_resolution[1])
    X, Y = np.meshgrid(initial_coordinates_x, initial_coordinates_y)
    X, Y = X.ravel(), Y.ravel()
    initial_coordinates = [[X[i], Y[i]] for i in range(len(X))]

    print '... Initial prediction:\n'
    input_image = im_data.copy()
    input_image.data *= 0
    for slice_number in range(0, nz, initial_resolution[2]):
        print '... ... slice #' + str(slice_number) + '/' + str(nz)
        patches = extract_patches_from_image(im_data, initial_coordinates, patch_size=patch_size, slice_of_interest=slice_number)
        patches = np.asarray(patches, dtype=int)
        patches = patches.reshape(patches.shape[0], patches.shape[1], patches.shape[2])

        X_test = patches
        y_pred = model.predict(X_test)
        y_pred = y_pred[:,1]

        for i,pp in enumerate(y_pred):
            input_image.data[initial_coordinates[i][0], initial_coordinates[i][1], slice_number] = pp

    input_image.setFileName(path_output)
    input_image.save()

    print '\n... Time to predict cord location: ' + str(np.round(time.time() - time_prediction)) + ' seconds'

def init_cnn_model(path_model):

    fname_model = path_model.split('.')[0]
    params_cnn = {'patch_size': [32, 32],
                  'number_of_channels': 1,
                  'batch_size': 128,
                  'number_of_features': [[32, 32], [64, 64]],
                  'loss': 'categorical_crossentropy'
                  }
    model = KerasConvNet(params_cnn)
    model.create_model()
    model.load(fname_model)

    return model

def prepare_prediction_cnn(path_local, model, cc, param_dct, thrsh):

    path_nii = path_local + 'input_nii_' + cc + '/'
    path_output_nii_cnn = path_local + 'cnn_nii_' + cc + '/'
    path_output_img_cnn = path_local + 'cnn_img_' + cc + '/'
    create_folders_local([path_output_nii_cnn, path_output_img_cnn])

    patch_size = param_dct['patch_size']
    initial_resolution = param_dct['initial_resolution']
    initial_resize = param_dct['initial_resize']

    with open(path_local + 'cnn_dataset_lst_' + cc + '.pkl') as outfile:    
        testing_lst = pickle.load(outfile)
        outfile.close()

    path_nii2convert_lst = []
    for fname_img in testing_lst:
        subject_name = fname_img.split('.')[0]
        fname_input = path_nii + subject_name + '.nii.gz'
        fname_output = path_output_nii_cnn + subject_name + '_pred.nii.gz'

        if not fname_output == '/Users/chgroc/data/data_sdika/cnn_nii_t2/mgh-3t_marco_005E_t2_pred.nii.gz':
            if not os.path.isfile(fname_output):
                print fname_output
                im_data = Image(fname_input)

                tick = time.time()

                im_data.data = 255.0 * (im_data.data - np.percentile(im_data.data, 0)) / np.abs(np.percentile(im_data.data, 0) - np.percentile(im_data.data, 100))

                prediction_cnn(im_data, model, initial_resolution, initial_resize,
                                thrsh, patch_size, fname_output)

                os.system('sct_image -i ' + fname_output + ' -setorient RPI -o ' + fname_output)
            
                path_nii2convert_lst.append(fname_output)

    convert_nii2img(path_nii2convert_lst, path_output_img_cnn)


def send_dataCNN2ferguson(path_local, path_ferguson, cc, llambda):

    pickle_ferguson = {
                        'contrast': cc,
                        'lambda': llambda
                        }
    path_pickle_ferguson = path_local + 'ferguson_cnn_config.pkl'
    pickle.dump(pickle_ferguson, open(path_pickle_ferguson, "wb"))

    print 'scp -r ' + path_local + 'cnn_img_' + cc + '/' + ' ferguson:' + path_ferguson
    os.system('scp -r ' + path_local + 'cnn_img_' + cc + '/' + ' ferguson:' + path_ferguson)
    # os.system('scp ' + path_pickle_ferguson + ' ferguson:' + path_ferguson)


def pull_CNNimg_convert_nii_remove_img(path_local, path_ferguson, cc, llambda):

    path_ferguson_res = path_ferguson + 'cnn_output_img_' + cc + '_' + llambda + '/'
    path_local_res_img = path_local + 'cnn_output_img_' + cc + '_' + llambda + '/'
    path_local_res_nii = path_local + 'cnn_output_nii_' + cc + '_' + llambda + '/'

    create_folders_local([path_local_res_nii])

    # Pull .img results from ferguson
    os.system('scp -r ferguson:' + path_ferguson_res + ' ' + path_local)

    # Convert .img to .nii
    # Remove .img files
    for ff in os.listdir(path_local_res_img):
        if ff.endswith('_ctr.hdr'):
            fname_cur = path_local_res_img + ff
            fname_cur_out = path_local_res_nii + ff.split('_ctr')[0] + '_centerline_pred.nii.gz'
            img = nib.load(fname_cur)
            nib.save(img, fname_cur_out)

    os.system('rm -r ' + path_local_res_img)


def _compute_stats(img_pred, img_true, img_seg_true):
    """
        -> mse = Mean Squared Error on distance between predicted and true centerlines
        -> maxmove = Distance max entre un point de la centerline predite et de la centerline gold standard
        -> zcoverage = Pourcentage des slices de moelle ou la centerline predite est dans la sc_seg_manual
    """

    stats_dct = {
                    'mse': None,
                    'maxmove': None,
                    'zcoverage': None
                }


    count_slice, slice_coverage = 0, 0
    mse_dist = []
    for z in range(img_true.dim[2]):

        if np.sum(img_true.data[:,:,z]):
            x_true, y_true = [np.where(img_true.data[:,:,z] > 0)[i][0] 
                                for i in range(len(np.where(img_true.data[:,:,z] > 0)))]
            x_pred, y_pred = [np.where(img_pred.data[:,:,z] > 0)[i][0]
                                for i in range(len(np.where(img_pred.data[:,:,z] > 0)))]
           
            xx_seg, yy_seg = np.where(img_seg_true.data[:,:,z]==1.0)
            xx_yy = [[x,y] for x, y in zip(xx_seg,yy_seg)]
            if [x_pred, y_pred] in xx_yy:
                slice_coverage += 1

            x_true, y_true = img_true.transfo_pix2phys([[x_true, y_true, z]])[0][0], img_true.transfo_pix2phys([[x_true, y_true, z]])[0][1]
            x_pred, y_pred = img_pred.transfo_pix2phys([[x_pred, y_pred, z]])[0][0], img_pred.transfo_pix2phys([[x_pred, y_pred, z]])[0][1]

            dist = ((x_true-x_pred))**2 + ((y_true-y_pred))**2
            mse_dist.append(dist)

            count_slice += 1

    if len(mse_dist):
        stats_dct['mse'] = sqrt(sum(mse_dist)/float(count_slice))
        stats_dct['maxmove'] = sqrt(max(mse_dist))
        stats_dct['zcoverage'] = float(slice_coverage*100.0)/count_slice

    return stats_dct


def _compute_stats_file(fname_ctr_pred, fname_ctr_true, fname_seg_true, folder_out, fname_out):

    img_pred = Image(fname_ctr_pred)
    img_true = Image(fname_ctr_true)
    img_seg_true = Image(fname_seg_true)

    stats_file_dct = _compute_stats(img_pred, img_true, img_seg_true)

    create_folders_local([folder_out])

    pickle.dump(stats_file_dct, open(fname_out, "wb"))


def _compute_stats_folder(subj_name_lst, cc, llambda, folder_out, fname_out):

    stats_folder_dct = {}

    mse_lst, maxmove_lst, zcoverage_lst = [], [], []
    for subj in subj_name_lst:
        with open(folder_out + 'res_' + cc + '_' + llambda + '_' + subj + '.pkl') as outfile:    
            subj_metrics = pickle.load(outfile)
            outfile.close()
        mse_lst.append(subj_metrics['mse'])
        maxmove_lst.append(subj_metrics['maxmove'])
        zcoverage_lst.append(subj_metrics['zcoverage'])

    stats_folder_dct['avg_mse'] = round(np.mean(mse_lst),2)
    stats_folder_dct['avg_maxmove'] = round(np.mean(maxmove_lst),2)
    stats_folder_dct['cmpt_fail_subj_test'] = round(sum(elt >= 10.0 for elt in maxmove_lst)*100.0/len(maxmove_lst),2)
    stats_folder_dct['avg_zcoverage'] = round(np.mean(zcoverage_lst),2)

    print stats_folder_dct
    pickle.dump(stats_folder_dct, open(fname_out, "wb"))



def compute_dataset_stats(path_local, cc, llambda):

    path_local_nii = path_local + 'cnn_output_nii_' + cc + '_' + llambda + '/'
    path_local_res_pkl = path_local + 'cnn_pkl_' + cc + '_' + llambda + '/'
    path_local_gold = path_local + 'gold_' + cc + '/'
    path_local_seg = path_local + 'input_nii_' + cc + '/'
    fname_pkl_out = path_local_res_pkl + 'res_' + cc + '_' + llambda + '_'

    subj_name_lst = []
    for ff in os.listdir(path_local_nii):
        print ff
        if ff.endswith('_centerline_pred.nii.gz'):
            subj_name_cur = ff.split('_pred_centerline_pred.nii.gz')[0]
            subj_name_lst.append(subj_name_cur)
            fname_subpkl_out = fname_pkl_out + subj_name_cur + '.pkl'
            
            if not os.path.isfile(fname_subpkl_out):
                subj_name_lst.append(subj_name_cur)
                path_cur_pred = path_local_nii + ff
                path_cur_gold = path_local_gold + subj_name_cur + '_centerline_gold.nii.gz'
                path_cur_gold_seg = path_local_seg + subj_name_cur + '_seg.nii.gz'

                _compute_stats_file(path_cur_pred, path_cur_gold, path_cur_gold_seg, path_local_res_pkl, fname_subpkl_out)

    fname_pkl_out_all = fname_pkl_out + 'all.pkl'
    if not os.path.isfile(fname_pkl_out_all):
        _compute_stats_folder(subj_name_lst, cc, llambda, path_local_res_pkl, fname_pkl_out_all)


# ******************************************************************************************


# ****************************      USER CASE      *****************************************

def readCommand(  ):
    "Processes the command used to run from the command line"
    parser = argparse.ArgumentParser('CNN-Sdika Pipeline')
    parser.add_argument('-ofolder', '--output_folder', help='Output Folder', required = False)
    parser.add_argument('-c', '--contrast', help='Contrast of Interest', required = False)
    parser.add_argument('-l', '--llambda', help='Lambda Sdika', required = False)
    parser.add_argument('-s', '--step', help='Prepare (step=0) or Push (step=1) or Pull (step 2) or Compute metrics (step=3) or Display results (step=4)', 
                                        required = False)
    parser.add_argument('-n', '--nb_train_img', help='Nb Training Images', required = False)

    arguments = parser.parse_args()
    return arguments


USAGE_STRING = """
  USAGE:      python sct_sdika.py -ofolder '/Users/chgroc/data/data_sdika/' <options>
                  -> ...s=0 >> CNN prediction
                  -> ...s=1 >> Push data to Ferguson
                  -> ...s=2 >> Pull data from Ferguson
                  -> ...s=3 >> Evaluate Sdika algo by computing metrics
                  -> ...s=4 >> Display results
                 """

if __name__ == '__main__':

    # Read input
    parse_arg = readCommand()

    if not parse_arg.output_folder:
        print USAGE_STRING
    else:
        path_local_sdika = parse_arg.output_folder

    path_ferguson = '/home/neuropoly/code/spine-cnn/'

    if not parse_arg.nb_train_img:
        nb_train_img = 1
    else:
        nb_train_img = int(parse_arg.nb_train_img)  

    if not parse_arg.step:
        step = 0
    else:
        step = int(parse_arg.step) 

    # Format of parser arguments
    contrast_of_interest = str(parse_arg.contrast)

    if contrast_of_interest == 't2':
        path_model = '/Users/chgroc/data/spine_detection/CNN_000035840256_000000_weights'
        threshold = 0.6720970273017883         # Set a value (float) or Trial path
        llambda = str(0.35)
        grid_search_dct={'patch_size':32,
                        'initial_resolution': [2, 2, 1],
                        'initial_resize': [0.1, 0.25]
                        }
    elif contrast_of_interest == 't2s':
        path_model = '/Users/chgroc/data/spine_detection/CNN_000005131008_000000_weights'
        threshold = 0.06963882595300674        # Set a value (float) or Trial path
        # llambda = str(0.35)
        grid_search_dct={'patch_size':32,
                        'initial_resolution': [2, 2, 1],
                        'initial_resize': [0.25, 0.25]
                        }
    else:
        if not parse_arg.llambda:
            llambda = str(1)
        else:
            llambda = str(parse_arg.llambda)  

    if not step:
        model = init_cnn_model(path_model)
        prepare_prediction_cnn(path_local_sdika, model, contrast_of_interest, grid_search_dct, threshold)
        # FAIRE UN PULL DES RESULTATS FERGUSSON PUIS PUSH NEW RESULTS

    elif step == 1:
        send_dataCNN2ferguson(path_local_sdika, path_ferguson, contrast_of_interest, llambda)

    elif step == 2:
        pull_CNNimg_convert_nii_remove_img(path_local_sdika, path_ferguson, contrast_of_interest, llambda)

    elif step == 3:
        compute_dataset_stats(path_local_sdika, contrast_of_interest, llambda)

    elif step == 4:
        with open(path_local_sdika + 'patho_dct_t1.pkl') as outfile:    
            patho_t1 = pickle.load(outfile)
            outfile.close()
        with open(path_local_sdika + 'patho_dct_t2s.pkl') as outfile:    
            patho_t2s = pickle.load(outfile)
            outfile.close()
        with open(path_local_sdika + 'patho_dct_t2.pkl') as outfile:    
            patho_t2 = pickle.load(outfile)
            outfile.close()

        print patho_t1.keys()
        print patho_t2s.keys()
        print patho_t2.keys()

        all_lst_t2 = [elt for dd in patho_t2 for elt in patho_t2[dd]]
        all_lst_t1 = [elt for dd in patho_t1 for elt in patho_t1[dd]]
        all_lst_t2s = [elt for dd in patho_t2s for elt in patho_t2s[dd]]
        all_lst = [a for l in [all_lst_t2, all_lst_t1, all_lst_t2s] for a in l]

        patient_lst = []
        patho_lst = []
        for cc in [patho_t2, patho_t1, patho_t2s]:
            for dd in cc:
                if not dd == u'HC':
                    for pp in cc[dd]:
                        if not pp in patient_lst:
                            patho_lst.append(dd)
                            patient_lst.append(pp)
        print patho_lst
        from collections import Counter
        print Counter(patho_lst)

    elif step == 5:

        path_seg_out = path_local_sdika + 'output_svm_propseg_' + contrast_of_interest + '/'
        path_seg_out = path_local_sdika + 'output_svm_propseg_' + contrast_of_interest + '/'
        path_seg_out_propseg = path_local_sdika + 'output_propseg_' + contrast_of_interest + '/'
        path_data = path_local_sdika + 'input_nii_' + contrast_of_interest + '/'
        path_seg_out_cur = path_seg_out + str(nb_train_img) + '/'
        path_seg_out_propseg = path_local_sdika + 'output_propseg_' + contrast_of_interest + '/'


        fname_out_pd = path_seg_out + str(nb_train_img) + '.pkl'

        if not os.path.isfile(fname_out_pd):
            with open(path_local_sdika + 'test_valid_' + contrast_of_interest + '.pkl') as outfile:    
                train_test_pd = pickle.load(outfile)
                outfile.close()

            print train_test_pd

            res_pd = train_test_pd[train_test_pd.valid_test=='test'][['patho', 'resol', 'subj_name']]
            subj_name_lst = res_pd.subj_name.values.tolist()

            res_pd['dice_svm'] = [0.0 for i in range(len(subj_name_lst))]
            res_pd[' '] = [' ' for i in range(len(subj_name_lst))]
            for file in os.listdir(path_seg_out_cur):
                if file.endswith('.nii.gz'):
                    file_src = path_seg_out_cur+file
                    file_dst = path_data + file
                    subj_id = file.split('_'+contrast_of_interest)[0]
                    file_dice = path_seg_out_cur+subj_id+'.txt'
                    if not os.path.isfile(file_dice):
                        os.system('sct_dice_coefficient -i ' + file_src + ' -d ' + file_dst + ' -o ' + file_dice)
                    text = open(file_dice, 'r').read()
                    print 'svm'
                    if len(text.split('= '))>1:
                        print float(text.split('= ')[1])
                        res_pd.loc[res_pd.subj_name==subj_id,'dice_svm'] = float(text.split('= ')[1])
                    else:
                        os.system('sct_register_multimodal -i ' + file_src + ' -d ' + file_dst + ' -identity 1 -ofolder ' + path_seg_out_cur)
                        file_src_reg = file_src.split('.nii.gz')[0] + '_src_reg.nii.gz'
                        os.system('sct_dice_coefficient -i ' + file_src_reg + ' -d ' + file_dst + ' -o ' + file_dice)
                        text = open(file_dice, 'r').read()
                        if len(text.split('= '))>1:
                            res_pd.loc[res_pd.subj_name==subj_id,'dice_svm'] = float(text.split('= ')[1])

            res_pd['dice_propseg'] = [0.0 for i in range(len(subj_name_lst))]
            for file in os.listdir(path_seg_out_propseg):
                if file.endswith('_seg.nii.gz'):
                    file_src = path_seg_out_propseg+file
                    file_dst = path_data + file
                    subj_id = file.split('_'+contrast_of_interest)[0]
                    file_dice = path_seg_out_propseg+subj_id+'.txt'
                    if not os.path.isfile(file_dice):
                        os.system('sct_dice_coefficient -i ' + file_src + ' -d ' + file_dst + ' -o ' + file_dice)
                    text = open(file_dice, 'r').read()
                    print 'propseg'
                    if len(text.split('= '))>1:
                        print float(text.split('= ')[1])
                        res_pd.loc[res_pd.subj_name==subj_id,'dice_propseg'] = float(text.split('= ')[1])
                    else:
                        os.system('sct_register_multimodal -i ' + file_src + ' -d ' + file_dst + ' -identity 1 -ofolder ' + path_seg_out_propseg)
                        file_src_reg = file_src.split('.nii.gz')[0] + '_src_reg.nii.gz'
                        os.system('sct_dice_coefficient -i ' + file_src_reg + ' -d ' + file_dst + ' -o ' + file_dice)
                        text = open(file_dice, 'r').read()
                        res_pd.loc[res_pd.subj_name==subj_id,'dice_svm'] = float(text.split('= ')[1])

            res_pd = res_pd[res_pd.dice_svm != 0.0]
            print res_pd
            res_pd.to_pickle(fname_out_pd)

        else:
            with open(fname_out_pd) as outfile:    
                res_pd = pickle.load(outfile)
                outfile.close()

        print res_pd
        stg_propseg = 'Mean = ' + str(round(np.mean(res_pd.dice_propseg.values.tolist()),2))
        stg_propseg += '\nStd = ' + str(round(np.std(res_pd.dice_propseg.values.tolist()),2))
        stg_propseg += '\nMedian = ' + str(round(np.median(res_pd.dice_propseg.values.tolist()),2))
        stg_svm = 'Mean = ' + str(round(np.mean(res_pd.dice_svm.values.tolist()),2))
        stg_svm += '\nStd = ' + str(round(np.std(res_pd.dice_svm.values.tolist()),2))
        stg_svm += '\nMedian = ' + str(round(np.median(res_pd.dice_svm.values.tolist()),2))
        stg_svm += '\nMin = ' + str(round(np.min(res_pd.dice_svm.values.tolist()),2))
        print stg_svm
        print stg_propseg
        print res_pd[res_pd.dice_svm<0.50]


        # y_label_stg = 'DICE coefficient per testing subject'

        # y_lim_min, y_lim_max = -0.01, 1.01
        # y_stg_loc = 0.5

        # sns.set(style="whitegrid", palette="Set1", font_scale=1.3)
        # palette_swarm = dict(patient = 'crimson', hc = 'darkblue')
        # fig, axes = plt.subplots(1, 1, sharey='col', figsize=(24, 8))
        # fig.subplots_adjust(left=0.05, bottom=0.05)
        # color_lst = [(0.40000000596046448, 0.7607843279838562, 0.64705884456634521), (0.55432528607985565, 0.62711267120697922, 0.79595541393055635)]
        
        # a = plt.subplot(1, 2, 1)
        # sns.violinplot(x=' ', y='dice_propseg', data=res_pd,
        #                       inner="quartile", cut=0, scale="width",
        #                       sharey=True,  color=color_lst[0])
        # sns.swarmplot(x=' ', y='dice_propseg', data=res_pd,
        #                       hue='patho', size=5, palette=palette_swarm)
        # a.set_ylabel(y_label_stg, fontsize=13)
        # a.set_xlabel('PropSeg without initialization', fontsize=13)
        # a.set_ylim([y_lim_min,y_lim_max])

        # b = plt.subplot(1, 2, 2)
        # sns.violinplot(x=' ', y='dice_svm', data=res_pd,
        #                       inner="quartile", cut=0, scale="width",
        #                       sharey=True,  color=color_lst[1])
        # sns.swarmplot(x=' ', y='dice_svm', data=res_pd,
        #                       hue='patho', size=5, palette=palette_swarm)
        # b.set_ylabel(y_label_stg, fontsize=13)
        # b.set_xlabel('PropSeg with Sdika initialization', fontsize=13)
        # b.set_ylim([y_lim_min,y_lim_max])

        # a.text(0.3, y_stg_loc, stg_propseg, fontsize=13)
        # b.text(0.3, y_stg_loc, stg_svm, fontsize=13)

        # # plt.show()
        # fig.tight_layout()
        # fig.savefig(path_seg_out + str(nb_train_img) + '.png')
        # plt.close()

    print TODO_STRING




