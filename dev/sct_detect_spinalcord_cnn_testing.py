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









####################################################################################################################
#   User Case


# *********************** PATH & CONFIG ***********************
# fname_local_script = '/Users/chgroc/spinalcordtoolbox/dev/sct_detect_spinalcord_cnn_ferguson.py'
# path_ferguson = '/home/neuropoly/code/spine-ms/'
# path_sct_testing_large = '/Volumes/data_shared/sct_testing/large/'
path_local_sdika = '/Users/chgroc/data/data_sdika/'
# create_folders_local([path_local_sdika])
# # contrast_lst = ['t2', 't1', 't2s']
# contrast_lst = ['t2']

prepare_dataset_cnn(path_local_sdika, 't2', '/Volumes/data_processing/bdeleener/machine_learning/filemanager_large_nobrain_nopad/datasets.pbz2')




path_model = '/Users/benjamindeleener/data/machine_learning/results_pipeline_cnn/large/CNN_000015360256_000000_weights'
threshold = 0.8936676383018494

path_model = path_model.split('.')[0]
params_cnn = {'patch_size': [32, 32],
              'number_of_channels': 1,
              'batch_size': 128,
              'number_of_features': [[32, 32], [64, 64]],
              'loss': 'categorical_crossentropy'
              }
model = KerasConvNet(params_cnn)
model.create_model()
model.load(path_model)

grid_search_dict_t2={'patch_size':32,
                    'initial_resolution': [2, 2, 1],
                    'initial_resize': [0.1, 0.25],
                    'initial_list_offset': [[xv, yv, zv] for xv in range(-1,1) for yv in range(-1,1) for zv in range(-5,5) if [xv, yv, zv] != [0, 0, 0]]
                    }
patch_size = grid_search_dict_t2['patch_size']
initial_resolution = grid_search_dict_t2['initial_resolution']
initial_resize = grid_search_dict_t2['initial_resize']
initial_list_offset = grid_search_dict_t2['initial_list_offset']

    # # Input Image
    # fname_input = arguments['-i']
    # folder_input, subject_name = os.path.split(fname_input)
    # subject_name = subject_name.split('.nii.gz')[0]
    # im_data = Image(fname_input)
    # print '\nInput Image: ' + fname_input

    # # Output Folder
    # fname_output = arguments['-o']
    # folder_output, subject_name_out = os.path.split(fname_output)
    # folder_output += '/'
    # prefixe_output = subject_name.split('.nii.gz')[0]
    # print '\nOutput Folder: ' + folder_output

    # #### Brouillon
    # # spiral_coord(np.array(range(180)).reshape(10,18),10,18)
    # # plot_cOm(folder_output + 'e23185_t2coord_pos_tmp.pkl', im_data.data)
    # #### Brouillon

    # tick = time.time()

    # im_data.data = 255.0 * (im_data.data - np.percentile(im_data.data, 0)) / np.abs(np.percentile(im_data.data, 0) - np.percentile(im_data.data, 100))

    # print '\nRun Initial patch-based prediction'
    # fname_seg = prediction_init(im_data, model, initial_resolution, initial_resize, initial_list_offset, 
    #                              threshold, feature_fct, patch_size, folder_output + prefixe_output, verbose)
    































# # *********************** PREPARE DATASET ***********************
# prepare_dataset(path_local_sdika, contrast_lst, path_sct_testing_large)

# # *********************** SEND SCRIPT TO FERGUSON ***********************
# os.system('scp ' + fname_local_script + ' ferguson:' + path_ferguson)

# # *********************** SEND DATA TO FERGUSON ***********************
# send_data2ferguson(path_local_sdika, path_ferguson, 't2', 1)
# # send_data2ferguson(path_local_sdika, path_ferguson, 't1', 1)

# # *********************** PULL RESULTS FROM FERGUSON ***********************
# # pull_img_convert_nii_remove_img(path_local_sdika, path_ferguson, 't2', 1)
# # pull_img_convert_nii_remove_img(path_local_sdika, path_ferguson, 't1', 1)
    

