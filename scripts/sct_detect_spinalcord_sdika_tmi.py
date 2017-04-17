# ****************************      IMPORT      *****************************************  
# Utils Imports
import pickle
import os
import nibabel as nib #### A changer en utilisant Image
import shutil
import numpy as np
from math import sqrt
from collections import Counter
import random
import json
import argparse
import seaborn as sns
import matplotlib.pyplot as plt
import itertools
import pandas as pd
from scipy.stats import ttest_rel, ttest_ind, f_oneway, normaltest, bartlett, norm, kruskal, kstest, wilcoxon, anderson, shapiro
# SCT Imports
from msct_image import Image
import sct_utils as sct
import time
# ***************************************************************************************

# ****************************      UTILS FUNCTIONS      ********************************

def create_folders_local(folder2create_lst):
    """
    
      Create folders if not exist
    
          Inputs:
              - folder2create_lst [list of string]: list of folder paths to create
          Outputs:
              -
    
    """           
    
    for folder2create in folder2create_lst:
        if not os.path.exists(folder2create):
            os.makedirs(folder2create)


def partition_resol(path_local, cc):

    fname_pkl_out = path_local + 'resol_dct_' + cc + '.pkl'
    # if not os.path.isfile(fname_pkl_out):
    path_dataset_pkl = path_local + 'dataset_lst_' + cc + '.pkl'
    dataset = pickle.load(open(path_dataset_pkl, 'r'))
    dataset_subj_lst = [f.split('.img')[0].split('_'+cc)[0] for f in dataset]
    dataset_path_lst = [path_local + 'input_nii_' + cc + '/' + f.split('.img')[0]+'.nii.gz' for f in dataset]

    resol_dct = {'sag': [], 'ax': [], 'iso': []}
    in_plane_ax, in_plane_iso, in_plane_sag = [], [], []
    thick_ax, thick_iso, thick_sag = [], [], []
    for img_path, img_subj in zip(dataset_path_lst, dataset_subj_lst):
        img = Image(img_path)

        resol_lst = [round(dd) for dd in img.dim[4:7]]
        if resol_lst.count(resol_lst[0]) == len(resol_lst):
            resol_dct['iso'].append(img_subj)
            in_plane_iso.append(img.dim[4])
            in_plane_iso.append(img.dim[5])
            thick_iso.append(img.dim[6])
        elif resol_lst[1]<resol_lst[0]:
            resol_dct['sag'].append(img_subj)
            in_plane_sag.append(img.dim[5])
            in_plane_sag.append(img.dim[6])
            thick_sag.append(img.dim[4])
        else:
            resol_dct['ax'].append(img_subj)
            in_plane_ax.append(img.dim[4])
            in_plane_ax.append(img.dim[5])
            thick_ax.append(img.dim[6])

        del img

    print '\n ax'
    print len(resol_dct['ax'])
    if len(resol_dct['ax']):
        print min(in_plane_ax), max(in_plane_ax)
        print min(thick_ax), max(thick_ax)
        print thick_ax
    print '\n iso'
    print len(resol_dct['iso'])
    if len(resol_dct['iso']):
        print min(in_plane_iso), max(in_plane_iso)
        print min(thick_iso), max(thick_iso)
    print '\n sag'
    print len(resol_dct['sag'])
    if len(resol_dct['sag']):
        print min(in_plane_sag), max(in_plane_sag)
        print min(thick_sag), max(thick_sag)

        # pickle.dump(resol_dct, open(fname_pkl_out, "wb"))
    # else:
    #     with open(fname_pkl_out) as outfile:    
    #         resol_dct = pickle.load(outfile)
    #         outfile.close()

    # return resol_dct

# ***************************************************************************************



# ****************************      STEP 0 FUNCTIONS      *******************************

def find_img_testing(path_large, contrast, path_local):
    """
    
      Explore a database folder (path_large)...
      ...and extract path to images for a given contrast (contrast)
    
          Inputs:
              - path_large [string]: path to database
              - contrast [string]: contrast of interest ('t2', 't1', 't2s')
          Outputs:
              - path_img [list of string]: list of image path
              - path_seg [list of string]: list of segmentation path
    
    """   

    center_lst, pathology_lst, path_img, path_seg = [], [], [], []
    for subj_fold in os.listdir(path_large):
        path_subj_fold = path_large + subj_fold + '/'

        if os.path.isdir(path_subj_fold):
            contrast_fold_lst = [contrast_fold for contrast_fold in os.listdir(path_subj_fold) 
                                                    if os.path.isdir(path_subj_fold+contrast_fold+'/')]
            contrast_fold_lst_oI = [contrast_fold for contrast_fold in contrast_fold_lst 
                                                    if contrast_fold==contrast or contrast_fold.startswith(contrast+'_')]
            
            # If this subject folder contains a subfolder related to the contrast of interest
            if len(contrast_fold_lst_oI):

                # Depending on the number of folder of interest:
                if len(contrast_fold_lst_oI)>1:
                    # In our case, we prefer axial images when available
                    ax_candidates = [tt for tt in contrast_fold_lst_oI if 'ax' in tt]
                    if len(ax_candidates):
                        contrast_fold_oI = ax_candidates[0]
                    else:
                        contrast_fold_oI = contrast_fold_lst_oI[0]                                               
                else:
                    contrast_fold_oI = contrast_fold_lst_oI[0]

                # For each subject and for each contrast, we want to pick only one image
                path_contrast_fold = path_subj_fold+contrast_fold_oI+'/'

                # If segmentation_description.json is available
                if os.path.exists(path_contrast_fold+'segmentation_description.json'):

                    with open(path_contrast_fold+'segmentation_description.json') as data_file:    
                        data_seg_description = json.load(data_file)
                        data_file.close()

                    # If manual segmentation of the cord is available
                    if len(data_seg_description['cord']):

                        # Extract data information from the dataset_description.json
                        with open(path_subj_fold+'dataset_description.json') as data_file:    
                            data_description = json.load(data_file)
                            data_file.close()

                        path_img_cur = path_contrast_fold+contrast_fold_oI+'.nii.gz'
                        path_seg_cur = path_contrast_fold+contrast_fold_oI+'_seg_manual.nii.gz'
                        if os.path.exists(path_img_cur) and os.path.exists(path_seg_cur):
                            path_img.append(path_img_cur)
                            path_seg.append(path_seg_cur)
                            center_lst.append(data_description['Center'])
                            pathology_lst.append(data_description['Pathology'])
                        else:
                            print '\nWARNING: file lacks: ' + path_contrast_fold + '\n'


    img_patho_lstoflst = [[i.split('/')[-3].split('.nii.gz')[0].split('_t2')[0], p] for i,p in zip(path_img,pathology_lst)]
    img_patho_dct = {}
    for ii_pp in img_patho_lstoflst:
        if not ii_pp[1] in img_patho_dct:
            img_patho_dct[ii_pp[1]] = []
        img_patho_dct[ii_pp[1]].append(ii_pp[0])
    if '' in img_patho_dct:
        for ii in img_patho_dct['']:
            img_patho_dct['HC'].append(ii)
        del img_patho_dct['']
    if u'NC' in img_patho_dct:
        for ii in img_patho_dct[u'NC']:
            img_patho_dct['HC'].append(ii)
        del img_patho_dct[u'NC']
    print img_patho_dct.keys()
    fname_pkl_out = path_local + 'patho_dct_' + contrast + '.pkl'
    pickle.dump(img_patho_dct, open(fname_pkl_out, "wb"))

    # Remove duplicates
    center_lst = list(set(center_lst))
    center_lst = [center for center in center_lst if center != ""]
    # Remove HC and non specified pathologies
    pathology_lst = [patho for patho in pathology_lst if patho != "" and patho != "HC" and patho != "NC"]
    pathology_dct = {x:pathology_lst.count(x) for x in pathology_lst}

    print '\n\n***************Contrast of Interest: ' + contrast + ' ***************'
    print '# of Subjects: ' + str(len(path_img))
    print '# of Centers: ' + str(len(center_lst))
    print 'Centers: ' + ', '.join(center_lst)
    print 'Pathologies:'
    print pathology_dct
    print '\n'

    return path_img, path_seg

def transform_nii_img(img_lst, path_out):
    """
    
      List .nii images which need to be converted to .img format
      + set same orientation RPI
      + set same value format (int16)
    
          Inputs:
              - img_lst [list of string]: list of path to image to transform
              - path_out [string]: path to folder where to save transformed images
          Outputs:
              - path_img2convert [list of string]: list of image paths to convert
    
    """   

    path_img2convert = []
    for img_path in img_lst:
        path_cur = img_path
        path_cur_out = path_out + '_'.join(img_path.split('/')[5:7]) + '.nii.gz'
        if not os.path.isfile(path_cur_out):
            shutil.copyfile(path_cur, path_cur_out)
            sct.run('sct_image -i ' + path_cur_out + ' -type int16 -o ' + path_cur_out)
            sct.run('sct_image -i ' + path_cur_out + ' -setorient RPI -o ' + path_cur_out)
            # os.system('sct_image -i ' + path_cur_out + ' -type int16 -o ' + path_cur_out)
            # os.system('sct_image -i ' + path_cur_out + ' -setorient RPI -o ' + path_cur_out)
        path_img2convert.append(path_cur_out)

    return path_img2convert

def transform_nii_seg(seg_lst, path_out, path_gold):
    """
    
      List .nii segmentations which need to be converted to .img format
      + set same orientation RPI
      + set same value format (int16)
      + set same header than related image
      + extract centerline from '*_seg_manual.nii.gz' to create gold standard
    
          Inputs:
              - seg_lst [list of string]: list of path to segmentation to transform
              - path_out [string]: path to folder where to save transformed segmentations
              - path_gold [string]: path to folder where to save gold standard centerline
          Outputs:
              - path_segs2convert [list of string]: list of segmentation paths to convert
    
    """

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
    """
    
      Convert .nii images to .img format
    
          Inputs:
              - path_nii2convert [list of string]: list of path to images to convert
              - path_out [string]: path to folder where to save converted images
          Outputs:
              - fname_img [list of string]: list of converted images (.img format) paths
    
    """ 

    fname_img = []
    for img in path_nii2convert:
        path_cur = img
        path_cur_out = path_out + img.split('.')[0].split('/')[-1] + '.img'
        if not img.split('.')[0].split('/')[-1].endswith('_seg') and not img.split('.')[0].split('/')[-1].endswith('_seg_centerline'):
            fname_img.append(img.split('.')[0].split('/')[-1] + '.img')
        if not os.path.isfile(path_cur_out):
            os.system('sct_convert -i ' + path_cur + ' -o ' + path_cur_out)

    return fname_img

def info_resol(fname_lst):

    resol_lst = []
    in_plane_ax, in_plane_iso, in_plane_sag = [], [], []
    thick_ax, thick_iso, thick_sag = [], [], []
    for img_path in fname_lst:
      img = Image(img_path)

      resol_cur_lst = [round(dd) for dd in img.dim[4:7]]
      if resol_cur_lst.count(resol_cur_lst[0]) == len(resol_cur_lst):
        resol_lst.append('iso')
        in_plane_iso.append(img.dim[4])
        thick_iso.append(img.dim[6])
      elif resol_cur_lst[1]<resol_cur_lst[0]:
        resol_lst.append('sag')
        in_plane_sag.append(img.dim[5])
        thick_sag.append(img.dim[4])
      else:
        resol_lst.append('ax')
        in_plane_ax.append(img.dim[5])
        thick_ax.append(img.dim[6])

      del img

    resol_dct = {'ax': {}, 'sag': {}, 'iso': {}}
    
    if len([r for r in resol_lst if r=='ax']):
        resol_dct['ax']['nb'] = len([r for r in resol_lst if r=='ax'])
        resol_dct['ax']['in_plane'] = in_plane_ax
        resol_dct['ax']['thick'] = thick_ax
    
    if len([r for r in resol_lst if r=='iso']):
        resol_dct['iso']['nb'] = len([r for r in resol_lst if r=='iso'])
        resol_dct['iso']['in_plane'] = in_plane_iso
        resol_dct['iso']['thick'] = thick_iso
    
    if len([r for r in resol_lst if r=='sag']):
        resol_dct['sag']['nb'] = len([r for r in resol_lst if r=='sag'])
        resol_dct['sag']['in_plane'] = in_plane_sag
        resol_dct['sag']['thick'] = thick_sag

    return resol_dct

def find_resol(fname_lst, info_pd):

    resol_lst = []
    in_plane_ax, in_plane_iso, in_plane_sag = [], [], []
    thick_ax, thick_iso, thick_sag = [], [], []
    for img_path, img_subj in zip(fname_lst, info_pd.subj_name.values.tolist()):
      img = Image(img_path)

      resol_cur_lst = [round(dd) for dd in img.dim[4:7]]
      if resol_cur_lst.count(resol_cur_lst[0]) == len(resol_cur_lst):
        resol_lst.append('iso')
        in_plane_iso.append(img.dim[4])
        in_plane_iso.append(img.dim[5])
        thick_iso.append(img.dim[6])
      elif resol_cur_lst[1]<resol_cur_lst[0]:
        resol_lst.append('sag')
        in_plane_sag.append(img.dim[5])
        in_plane_sag.append(img.dim[6])
        thick_sag.append(img.dim[4])
      else:
        resol_lst.append('ax')
        in_plane_ax.append(img.dim[4])
        in_plane_ax.append(img.dim[5])
        thick_ax.append(img.dim[6])

      del img

    print '\n ax'
    print len([r for r in resol_lst if r=='ax'])
    if len([r for r in resol_lst if r=='ax']):
        print min(in_plane_ax), max(in_plane_ax)
        print min(thick_ax), max(thick_ax)
        print thick_ax
    print '\n iso'
    print len([r for r in resol_lst if r=='iso'])
    if len([r for r in resol_lst if r=='iso']):
        print min(in_plane_iso), max(in_plane_iso)
        print min(thick_iso), max(thick_iso)
    print '\n sag'
    print len([r for r in resol_lst if r=='sag'])
    if len([r for r in resol_lst if r=='sag']):
        print min(in_plane_sag), max(in_plane_sag)
        print min(thick_sag), max(thick_sag)

    info_pd['resol'] = resol_lst

    return info_pd

def prepare_dataset(path_local, cc, path_sct_testing_large):

    path_local_gold = path_local + 'gold/' + cc + '/'
    path_local_input_nii = path_local + 'input_nii/' + cc + '/'
    path_local_input_img = path_local + 'input_img/' + cc + '/'

    with open(path_local + 'path_' + cc + '.pkl') as outfile:    
        data_pd = pickle.load(outfile)
        outfile.close()
    with open(path_local + 'info_' + cc + '.pkl') as outfile:    
        info_pd = pickle.load(outfile)
        outfile.close()

    path_fname_img = [pp+'.nii.gz' for pp in data_pd.path_sct.values.tolist()]
    path_fname_seg = [pp+'_seg_manual.nii.gz' for pp in data_pd.path_sct.values.tolist()]

    path_img2convert = transform_nii_img(path_fname_img, path_local_input_nii)

    if not 'resol' in info_pd:
        info_pd = find_resol(path_img2convert, info_pd)
        info_pd.to_pickle(path_local + 'info_' + cc + '.pkl')

    path_seg2convert = transform_nii_seg(path_fname_seg, path_local_input_nii, path_local_gold)
    path_imgseg2convert = path_img2convert + path_seg2convert
    fname_img_lst = convert_nii2img(path_imgseg2convert, path_local_input_img)

    data_pd['path_loc'] = fname_img_lst
    print data_pd

    data_pd.to_pickle(path_local + 'path_' + cc + '.pkl')



# ****************************      STEP 0 FUNCTIONS      *******************************

def panda_dataset(path_local, cc, path_large, nb_train_valid):

  info_dct = {'subj_name': [], 'patho': [], 'center': []}
  path_dct = {'subj_name': [], 'path_sct': []}
  for subj_fold in os.listdir(path_large):
    path_subj_fold = path_large + subj_fold + '/'
    if os.path.isdir(path_subj_fold):
        contrast_fold_lst = [contrast_fold for contrast_fold in os.listdir(path_subj_fold) 
                                                if os.path.isdir(path_subj_fold+contrast_fold+'/')]
        contrast_fold_lst_oI = [contrast_fold for contrast_fold in contrast_fold_lst 
                                                if contrast_fold==cc or contrast_fold.startswith(cc+'_')]
        
        # If this subject folder contains a subfolder related to the contrast of interest
        if len(contrast_fold_lst_oI):
            # Depending on the number of folder of interest:
            if len(contrast_fold_lst_oI)>1:
                # In our case, we prefer axial images when available
                ax_candidates = [tt for tt in contrast_fold_lst_oI if 'ax' in tt]
                if len(ax_candidates):
                    contrast_fold_oI = ax_candidates[0]
                else:
                    sup_candidates = [tt for tt in contrast_fold_lst_oI if 'sup' in tt]
                    if len(sup_candidates):
                      contrast_fold_oI = sup_candidates[0]
                    else:
                      contrast_fold_oI = contrast_fold_lst_oI[0]                                               
            else:
                contrast_fold_oI = contrast_fold_lst_oI[0]

            # For each subject and for each contrast, we want to pick only one image
            path_contrast_fold = path_subj_fold+contrast_fold_oI+'/'

            # If segmentation_description.json is available
            if os.path.exists(path_contrast_fold+'segmentation_description.json'):

                with open(path_contrast_fold+'segmentation_description.json') as data_file:    
                    data_seg_description = json.load(data_file)
                    data_file.close()

                # If manual segmentation of the cord is available
                if len(data_seg_description['cord']):

                    if contrast_fold_oI != 'dmri':
                        path_dct['subj_name'].append(subj_fold + '_' + contrast_fold_oI)
                    else:
                        path_dct['subj_name'].append(subj_fold + '_' + 'dwi_mean')
                    info_dct['subj_name'].append(subj_fold + '_' + contrast_fold_oI)
                    if contrast_fold_oI != 'dmri':
                        path_dct['path_sct'].append(path_contrast_fold + contrast_fold_oI)
                    else:
                        path_dct['path_sct'].append(path_contrast_fold + 'dwi_mean')

                    # Extract data information from the dataset_description.json
                    with open(path_subj_fold+'dataset_description.json') as data_file:    
                        data_description = json.load(data_file)
                        data_file.close()

                    info_dct['center'].append(str(data_description['Center']))
                    if str(data_description['Pathology']) == '' or str(data_description['Pathology']) == 'NC':
                        info_dct['patho'].append('HC')
                    else:
                        info_dct['patho'].append(str(data_description['Pathology']))


  info_pd = pd.DataFrame.from_dict(info_dct)
  path_pd = pd.DataFrame.from_dict(path_dct)

  hc_lst = info_pd[info_pd.patho=='HC'].subj_name.values.tolist()
  data_lst = info_pd.subj_name.values.tolist()
  lambda_rdn = 0.23
  random.shuffle(hc_lst, lambda: lambda_rdn)
  training_lst = hc_lst[:nb_train_valid]

  print training_lst
  print np.unique([t.split('_')[0] for t in training_lst])

  info_pd['train_test'] = ['test' for i in range(len(data_lst))]
  for s in training_lst:
      info_pd.loc[info_pd.subj_name==s,'train_test'] = 'train'

  print info_pd
  info_pd.to_pickle(path_local + 'info_' + cc + '.pkl')
  path_pd.to_pickle(path_local + 'path_' + cc + '.pkl')

def prepare_train(path_local, path_outdoor, cc, nb_img_lst, nb_boostrap):

    with open(path_local + 'info_' + cc + '.pkl') as outfile:    
        data_pd = pickle.load(outfile)
        outfile.close()

    valid_lst = data_pd[data_pd.train_test == 'train']['subj_name'].values.tolist()
    print valid_lst
    test_lst = data_pd[data_pd.train_test == 'test']['subj_name'].values.tolist()

    max_k = max(nb_img_lst)
    print '\nExperiment: '
    print '... contrast: ' + cc
    print '... nb image used for training: ' + str(max_k) + '\n'
    print '... nb image used for validation: ' + str(len(valid_lst)-max_k) + '\n'
    print '... nb image used for testing: ' + str(len(test_lst)) + '\n'

    train_idx_lst = []
    train_lst = []
    while len(train_lst)<nb_boostrap:
        idx_lst = random.sample(range(len(valid_lst)), max_k)
        if not sorted(idx_lst) in train_idx_lst:
            train_idx_lst.append(idx_lst)
            train_lst_cur = []
            for idx in idx_lst:
                train_lst_cur.append(valid_lst[idx])
            train_lst.append(train_lst_cur)

    print train_lst
    print len(train_lst)
    print len(train_lst[0])

    path_outdoor_cur = path_outdoor + 'input_img/' + cc + '/'
    path_local_train_max = path_local + 'input_train/' + cc + '/' + cc + '_' + str(max_k) + '/'
    if os.listdir(path_local_train_max) == []: 
        for k in list_k:
            path_local_train = path_local + 'input_train/' + cc + '/' + cc + '_' + str(k) + '/'
            for b in range(nb_boostrap):
                train_lst_bk = random.sample(train_lst[b], k)
                stg, stg_seg, stg_val = '', '', ''
                for tt_tt in train_lst_bk:
                    stg += path_outdoor_cur + tt_tt + '\n'
                    stg_seg += path_outdoor_cur + tt_tt + '_seg' + '\n'
                for v in valid_lst:
                    if v not in train_lst[b]:
                        stg_val += v + '\n'
                path2save = path_local_train
                with open(path2save + str(b).zfill(3) + '.txt', 'w') as text_file:
                    text_file.write(stg)
                    text_file.close()
                with open(path2save + str(b).zfill(3) + '_ctr.txt', 'w') as text_file:
                    text_file.write(stg_seg)
                    text_file.close()
                with open(path2save + str(b).zfill(3) + '_valid.txt', 'w') as text_file:
                    text_file.write(stg_val)
                    text_file.close()


def send_data2ferguson(path_local, pp_ferguson, cc, nb_img, data_lst, path_train, rot_bool=False, lambda_bool=False, trainer='', dyn=False):
    """
    
      MAIN FUNCTION OF STEP 1
      Prepare training strategy and save it in 'ferguson_config.pkl'
      + send data to ferguson
      + send training files to ferguson
      + send training strategy to ferguson
    
          Inputs:
              - path_local [string]: working folder
              - path_ferguson [string]: ferguson working folder
              - cc [string]: contrast of interest
              - nb_img [int]: nb of images for training
          Outputs:
              - 
    
    """
    pickle_ferguson = {
                        'contrast': cc,
                        'nb_image_train': nb_img,
                        'valid_subj': data_lst,
                        'path_ferguson': pp_ferguson,
                        'dyn': dyn,
                        'rot': rot_bool,
                        'lambda': lambda_bool,
                        'best_trainer': trainer
                        }
    path_pickle_ferguson = path_local + 'ferguson_config.pkl'
    output_file = open(path_pickle_ferguson, 'wb')
    pickle.dump(pickle_ferguson, output_file)
    output_file.close()

    os.system('scp -r ' + path_train + ' ferguson:' + pp_ferguson)
    os.system('scp ' + path_pickle_ferguson + ' ferguson:' + pp_ferguson)

    print pickle_ferguson


# ****************************      STEP 2 FUNCTIONS      *******************************

def pull_img_convert_nii_remove_img(path_local, path_ferguson, cc, nb_img):

    path_ferguson_res = path_ferguson + 'output_img_' + cc + '_'+ str(nb_img) + '/'
    path_local_res_img_scp = path_local+'output_img/'+cc+'/'
    path_local_res_nii = path_local+'output_nii/'+cc+'/'+str(nb_img)+'/'
    path_local_res_time = path_local+'output_time/'+cc+'/'+str(nb_img)+'/'

    # Pull .img results from ferguson
    os.system('scp -r ferguson:' + path_ferguson_res + ' ' + path_local_res_img_scp)
    
    path_local_res_img = path_local+'output_img/'+cc+'/'+str(nb_img)+'/'
    os.rename(path_local_res_img_scp+'output_img_'+cc+'_'+str(nb_img)+'/', path_local_res_img[:-1])

    # Convert .img to .nii
    # Remove .img files
    for f in os.listdir(path_local_res_img):
        if not f.startswith('.'):
            path_res_cur = path_local_res_nii + f + '/'
            path_res_cur_time = path_local_res_time + f + '/'
            create_folders_local([path_res_cur, path_res_cur_time])

            if os.path.isdir(path_local_res_img+f):
                for ff in os.listdir(path_local_res_img+f):
                    if ff.endswith('_ctr.hdr'):
                        path_cur = path_local_res_img + f + '/' + ff
                        path_cur_out = path_res_cur + ff.split('_ctr')[0] + '_centerline_pred.nii.gz'
                        img = nib.load(path_cur)
                        nib.save(img, path_cur_out)
                    elif ff.endswith('.txt') and ff != 'time.txt':
                        shutil.copyfile(path_local_res_img + f + '/' + ff, path_res_cur_time + ff)

                # os.system('rm -r ' + path_local_res_img + f)

def pull_img_rot(path_local, path_ferguson, cc, stg):

    path_ferguson_res = path_ferguson + 'output_img_' + cc + '_'+ stg + '/'
    path_local_res_img_scp = path_local + 'output_img/' + cc + '/'
    path_local_res_nii = path_local + 'output_nii/' + cc + '/' + stg + '/'

    # Pull .img results from ferguson
    print path_ferguson_res
    print path_local_res_img_scp
    os.system('scp -r ferguson:' + path_ferguson_res + ' ' + path_local_res_img_scp)
    
    path_local_res_img = path_local + 'output_img/' + cc + '/' + stg + '/'
    os.rename(path_local_res_img_scp + 'output_img_' + cc + '_' + stg + '/', path_local_res_img[:-1])

    # Convert .img to .nii
    # Remove .img files
    for f in os.listdir(path_local_res_img):
        if not f.startswith('.'):
            path_res_cur = path_local_res_nii + f + '/'
            create_folders_local([path_res_cur])

            if os.path.isdir(path_local_res_img+f):
                for ff in os.listdir(path_local_res_img+f):
                    if ff.endswith('_ctr.hdr'):
                        path_cur = path_local_res_img + f + '/' + ff
                        path_cur_out = path_res_cur + ff.split('_ctr')[0] + '_centerline_pred.nii.gz'
                        try:
                            img = nib.load(path_cur)
                            nib.save(img, path_cur_out)
                        except Exception: 
                            pass

def pull_img_test(path_local, path_ferguson, cc):

    path_ferguson_res = path_ferguson + 'output_img_' + cc + '/'
    path_ferguson_res_dyn = path_ferguson + 'output_img_' + cc + '_dyn/'
    path_local_res_img_scp = path_local + 'output_img/' + cc + '/'
    path_local_res_nii = path_local + 'output_nii/' + cc + '/optic/'
    
    path_local_time = path_local + 'output_time/' + cc + '/optic/'
    path_local_time_dyn = path_local + 'output_time/' + cc + '/dyn/'
    if not os.path.exists(path_local_time):
        os.makedirs(path_local_time)
    if not os.path.exists(path_local_time_dyn):
        os.makedirs(path_local_time_dyn)

    os.system('scp -r ferguson:' + path_ferguson_res + ' ' + path_local_res_img_scp)
    os.system('scp -r ferguson:' + path_ferguson_res_dyn + ' ' + path_local_res_img_scp)
    
    path_local_res_img = path_local + 'output_img/' + cc + '/optic/'
    os.rename(path_local_res_img_scp + 'output_img_' + cc + '/', path_local_res_img[:-1])

    path_local_res_img_dyn = path_local + 'output_img/' + cc + '/dyn/'
    os.rename(path_local_res_img_scp + 'output_img_' + cc + '_dyn/', path_local_res_img_dyn[:-1])

    for f in os.listdir(path_local_res_img):
        if not f.startswith('.'):
            path_res_cur = path_local_res_nii + f + '/'
            create_folders_local([path_res_cur])

            if os.path.isdir(path_local_res_img+f):
                for ff in os.listdir(path_local_res_img+f):
                    if ff.endswith('_ctr.hdr'):
                        path_cur = path_local_res_img + f + '/' + ff
                        path_cur_out = path_res_cur + ff.split('_ctr')[0] + '_centerline_pred.nii.gz'
                        try:
                            img = nib.load(path_cur)
                            nib.save(img, path_cur_out)
                        except Exception: 
                            pass
                    elif ff.endswith('_time.txt'):
                        shutil.copyfile(path_local_res_img + f + '/' + ff, path_local_time+ff)
                        shutil.copyfile(path_local_res_img_dyn + f + '/' + ff, path_local_time_dyn+ff)  




# ******************************************************************************************

# ****************************      STEP 3 FUNCTIONS      *******************************

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
            if np.sum(img_pred.data[:,:,z]):
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

def _compute_stats_file(fname_ctr_pred, fname_ctr_true, fname_seg_true, fname_out):

    img_pred = Image(fname_ctr_pred)
    img_true = Image(fname_ctr_true)
    img_seg_true = Image(fname_seg_true)

    stats_file_dct = _compute_stats(img_pred, img_true, img_seg_true)

    pickle.dump(stats_file_dct, open(fname_out, "wb"))


def compute_dataset_stats(path_local, cc, suf):

    if not suf=='hough':
        path_local_nii = path_local + 'output_nii/' + cc + '/' + suf + '/'
        path_local_res_pkl = path_local + 'output_pkl/' + cc + '/' + suf + '/'
    else:
        path_local_nii = path_local + 'propseg_nii/' + cc + '/'
        path_local_res_pkl = path_local + 'output_pkl/' + cc + '/'
    path_local_gold = path_local + 'gold/' + cc + '/'
    path_local_seg = path_local + 'input_nii/' + cc + '/'

    for f in os.listdir(path_local_nii):
        if not f.startswith('.'):
            path_res_cur = path_local_nii + f + '/'
            
            if suf == 'hough':
                folder_subpkl_out = path_local_res_pkl + suf + '/'
                path_res_cur = path_local_nii
            else:
                folder_subpkl_out = path_local_res_pkl + f + '/'
                create_folders_local([folder_subpkl_out])                
            
            for ff in os.listdir(path_res_cur):
                if ff.endswith('_centerline_pred.nii.gz'):
                    subj_name_cur = ff.split('_centerline_pred.nii.gz')[0]
                    if subj_name_cur=='sct_example_data_dmri':
                        subj_name_cur='unf_sct_example_data_dmri'
                    fname_subpkl_out = folder_subpkl_out + 'res_' + subj_name_cur + '.pkl'

                    if not os.path.isfile(fname_subpkl_out):
                        path_cur_pred = path_res_cur + ff
                        path_cur_gold = path_local_gold + subj_name_cur + '_centerline_gold.nii.gz'
                        path_cur_gold_seg = path_local_seg + subj_name_cur + '_seg.nii.gz'
                        _compute_stats_file(path_cur_pred, path_cur_gold, path_cur_gold_seg, fname_subpkl_out)

# ******************************************************************************************

# ****************************      STEP 4 FUNCTIONS      *******************************

def find_id_extr_df(df, mm):

  if 'zcoverage' in mm:
    return [df[mm].max(), df[df[mm] == df[mm].max()]['id'].values.tolist()[0], df[mm].min(), df[df[mm] == df[mm].min()]['id'].values.tolist()[0]]
  else:
    return [df[mm].min(), df[df[mm] == df[mm].min()]['id'].values.tolist()[0], df[mm].max(), df[df[mm] == df[mm].max()]['id'].values.tolist()[0]]

def panda_boostrap_k(path_folder_pkl, k_lst):

    k_lst = [str(k) for k in k_lst]

    boostrap_k_dct = {'it': []}
    for k in k_lst:
        boostrap_k_dct['mse_moy_'+k] = []
        boostrap_k_dct['mse_std_'+k] = []
        boostrap_k_dct['mse_med_'+k] = []

    print boostrap_k_dct

    for fold in os.listdir(path_folder_pkl):
      path_nb_cur = path_folder_pkl + fold + '/'
      if os.path.isdir(path_nb_cur) and fold in k_lst:

        for tr_subj in os.listdir(path_nb_cur):

            path_cur = path_nb_cur + tr_subj + '/'

            if os.path.isdir(path_cur):
                # boostrap_k_dct['it'].append(tr_subj)
    #           metric_k_dct['k'].append(fold)

                metric_cur_dct = {'maxmove': [], 'mse': [], 'zcoverage': []}
                for file in os.listdir(path_cur):
                    if file.endswith('.pkl'):
                        with open(path_cur+file) as outfile:    
                            metrics = pickle.load(outfile)
                            outfile.close()
                  
                        for mm in metrics:
                            if mm in metric_cur_dct:
                                metric_cur_dct[mm].append(metrics[mm])

                boostrap_k_dct['mse_std_'+str(fold)].append(np.std(metric_cur_dct['mse']))
                boostrap_k_dct['mse_moy_'+str(fold)].append(np.mean(metric_cur_dct['mse']))
                boostrap_k_dct['mse_med_'+str(fold)].append(np.median(metric_cur_dct['mse']))

                
        if boostrap_k_dct['it'] == []:
            boostrap_k_dct['it'] = range(len(boostrap_k_dct['mse_std_'+str(fold)]))

    boostrap_k_pd = pd.DataFrame.from_dict(boostrap_k_dct)
    print boostrap_k_pd

    return boostrap_k_pd


def panda_testing(path_optic_pkl, path_hough_pkl='', info_pd=None, cc=''):

    res_dct = {'mse': [], 'zcoverage': [], 'subj_name': [], 'algo': [], 'contrast': [], 'patho': []}

    for f in os.listdir(path_optic_pkl):
        if os.path.isdir(path_optic_pkl+f):
            path_res = path_optic_pkl+f+'/'
            for file in os.listdir(path_res):
                if file.endswith('.pkl'):
                    with open(path_res+file) as outfile:    
                        metrics = pickle.load(outfile)
                        outfile.close()
              
                    for mm in metrics:
                        if mm in res_dct:
                            res_dct[mm].append(metrics[mm])

                    res_dct['subj_name'].append(file.split('res_')[1].split('.pkl')[0])
                    res_dct['algo'].append('optic')
                    res_dct['contrast'].append(cc)
                    patho_cur = info_pd[info_pd.subj_name==file.split('res_')[1].split('.pkl')[0]].patho.values.tolist()[0]
                    if patho_cur != 'HC':
                        res_dct['patho'].append('patho')
                    else:
                        res_dct['patho'].append(patho_cur)

    if len(path_hough_pkl):
        for file in os.listdir(path_hough_pkl):
            if file.endswith('.pkl'):
                with open(path_hough_pkl+file) as outfile:    
                    metrics = pickle.load(outfile)
                    outfile.close()
          
                for mm in metrics:
                    if mm in res_dct:
                        res_dct[mm].append(metrics[mm])

                res_dct['subj_name'].append(file.split('res_')[1].split('.pkl')[0])
                res_dct['algo'].append('hough')
                res_dct['contrast'].append(cc)
                patho_cur = info_pd[info_pd.subj_name==file.split('res_')[1].split('.pkl')[0]].patho.values.tolist()[0]
                if patho_cur != 'HC':
                    res_dct['patho'].append('patho')
                else:
                    res_dct['patho'].append(patho_cur)   

    res_pd = pd.DataFrame.from_dict(res_dct)
    # print res_pd
    # print res_pd.mse.values.tolist()
    dct_tmp = dict(Counter(res_pd.subj_name.values.tolist()))
    for i in dct_tmp:
        if dct_tmp[i]==1:
            print i

    return res_pd

def panda_testing_seg(path_data, path_optic, info_pd, cc, algo_name):

    res_dct = {'dice': [], 'time': [], 'subj_name': [], 'algo': [], 'contrast': [], 'patho': []}

    subj_lst = info_pd.subj_name.values.tolist()

    for s in subj_lst:
        fname_dice = path_optic + s + '_dice.txt'
        fname_time = path_optic + s + '_time.txt'
        path_img_cur = path_data + s + '.nii.gz'
        if os.path.isfile(fname_dice) and os.path.isfile(fname_time):

            if len(open(fname_dice, 'r').read()):
                res_dct['dice'].append(float(open(fname_dice, 'r').read().split('= ')[1].split('\n')[0]))
            else:
                fname_gt = path_data + s + '_seg.nii.gz'
                fname_output = path_optic + s + '_seg.nii.gz'
                os.system('sct_register_multimodal -i ' + fname_output + ' -d ' + fname_gt + ' -identity 1 -ofolder ' + path_optic)
                fname_output_reg = fname_output.split('.nii.gz')[0] + '_src_reg.nii.gz'
                os.system('sct_dice_coefficient -i ' + fname_output_reg + ' -d ' + fname_gt + ' -o ' + fname_dice)
                if not len(open(fname_dice, 'r').read()):
                    im_seg_gt = Image(fname_gt)
                    im_seg = Image(fname_output_reg)
                    im_seg_new = im_seg_gt.copy()
                    im_seg_new.data = im_seg.data
                    im_seg_new.setFileName(fname_output_reg)
                    im_seg_new.absolutepath =  fname_output_reg
                    im_seg_new.save()
                    del im_seg_gt
                    del im_seg
                    os.system('sct_dice_coefficient -i ' + fname_output_reg + ' -d ' + fname_gt + ' -o ' + fname_dice)

                res_dct['dice'].append(float(open(fname_dice, 'r').read().split('= ')[1].split('\n')[0]))

            img = Image(path_img_cur)
            nz_cur = img.dim[2]
            del img
            res_dct['time'].append(float(open(fname_time, 'r').read())/nz_cur)
            
            res_dct['subj_name'].append(s)
            res_dct['algo'].append(algo_name)
            res_dct['contrast'].append(cc)
            patho_cur = info_pd[info_pd.subj_name==s].patho.values.tolist()[0]
            if patho_cur != 'HC':
                res_dct['patho'].append('patho')
            else:
                res_dct['patho'].append(patho_cur)

    res_pd = pd.DataFrame.from_dict(res_dct)
    print res_pd
    print res_pd.dice.values.tolist()

    return res_pd

def panda_best_k(path_pkl, boostrap_k_pd, k_lst):

    dct_tmp = {}
    lst_val=[]

    idx_best = boostrap_k_pd['mse_moy_1'].idxmin()
    print idx_best
    for k in k_lst[::-1]:
        dct_tmp['mse_'+str(k)] = []
        # idx_best = boostrap_k_pd['mse_moy_'+str(k)].idxmin()
        path_best = path_pkl + str(k) + '/' + str(idx_best).zfill(3) + '/'
        print path_best
        
        for val_pkl in os.listdir(path_best):
            if val_pkl.endswith('.pkl'):
                if str(k)==str(15):
                    lst_val.append(val_pkl)
                
                if val_pkl in lst_val:
                    with open(path_best+val_pkl) as outfile:    
                        metrics = pickle.load(outfile)
                        outfile.close()
                    dct_tmp['mse_'+str(k)].append(metrics['mse'])
        print k
        print len(dct_tmp['mse_'+str(k)])
    
    print lst_val
    best_k_pd = pd.DataFrame.from_dict(dct_tmp)
    print best_k_pd

    return best_k_pd

def panda_best_rot(path_pkl, boostrap_rot_pd, rot_lst):

    dct_tmp = {}
    val_lst = []
    for r in rot_lst:
        idx_best = boostrap_rot_pd['mse_moy_'+r].idxmin()
        # idx_best = '069' if r == '1-0' else '052'
        dct_tmp['mse_'+str(r)] = []
        path_best = path_pkl + str(r) + '/' + str(idx_best).zfill(3) + '/'
        
        for val_pkl in os.listdir(path_best):
            if val_pkl.endswith('.pkl'):
                if len(val_lst)<39:
                    val_lst.append(val_pkl)
                if val_pkl in val_lst and len(dct_tmp['mse_'+str(r)])<35:
                    with open(path_best+val_pkl) as outfile:    
                        metrics = pickle.load(outfile)
                        outfile.close()
                    print path_best+val_pkl
                    dct_tmp['mse_'+str(r)].append(metrics['mse'])
                    if metrics['mse']>2.0:
                        print val_pkl

        print r
        print path_best
    
    for sss in dct_tmp:
        print sss
        print len(dct_tmp[sss])
    best_r_pd = pd.DataFrame.from_dict(dct_tmp)
    # print best_r_pd

    return best_r_pd, path_best

def plot_boostrap_k(boostrap_pd, k_lst, cc):

    toplot_pd = boostrap_pd[k_lst]
    print toplot_pd

    if '60' in k_lst[0]:
        if cc == 't2':
            x_min, x_max = 0.0, 3.5
            y_min, y_max = 0.0, 12.0
        elif cc == 't2s':
            x_min, x_max = 0.0, 8.5
            y_min, y_max = 0.0, 17.0
        elif cc == 'dmri':
            x_min, x_max = 0.0, 9.5
            y_min, y_max = 0.0, 17.0
        else:
            x_min, x_max = 0.0, 6.0
            y_min, y_max = 0.0, 30.0
        color_h = 'red'
        nb_row, nb_col = 1, len(k_lst)
    elif '-' in k_lst[0]:
        if cc == 't2':
            x_min, x_max = 0.0, 6.0
            y_min, y_max = 0.0, 16.0
        elif cc == 't2s':
            x_min, x_max = 0.0, 8.0
            y_min, y_max = 0.0, 17.0
        elif cc == 'dmri':
            x_min, x_max = 0.0, 9.0
            y_min, y_max = 0.0, 17.0
        else:
            x_min, x_max = 0.0, 3.5
            y_min, y_max = 0.0, 30.0
        color_h = 'green'
        nb_row, nb_col = 2, 4
        # nb_row, nb_col = 1, len(k_lst)
    else:
        if cc == 't1':
            x_min, x_max = 0.0, 2.0
            y_min, y_max = 0.0, 7.0
        elif cc == 'dmri':
            x_min, x_max = 0.0, 9.0
            y_min, y_max = 0.0, 11.0     
        else:
            x_min, x_max = 0.0, 3.5
            y_min, y_max = 0.0, 10.0
        color_h = 'blue'
        nb_row, nb_col = 1, len(k_lst)

    sns.set(style="whitegrid", palette="Set1", font_scale=1.3)
    fig, axes = plt.subplots(nb_row, nb_col, sharey='col', figsize=(8*nb_col, 8*nb_row))
    for i, col in enumerate(k_lst):
        a = plt.subplot(nb_row, nb_col, i+1)
        if max(toplot_pd[col]) > x_max:
            plt.hist(toplot_pd[col], bins=50, color=color_h, alpha=0.5, range=[min(toplot_pd[col]), max(toplot_pd[col])])
            plt.xlim([min(toplot_pd[col]), max(toplot_pd[col])])
        else:
            plt.hist(toplot_pd[col], bins=50, color=color_h, alpha=0.5, range=[x_min, x_max])
            plt.xlim([x_min, x_max])
        
        plt.ylabel('Validation subject')
        plt.xlabel(col)

        plt.ylim([y_min, y_max])

    fig.tight_layout()
    plt.show()


def plot_k(path_folder, mm, cc, metric_k_pd):

    dct_tmp = {'Subject': [], 'metric': [], 'nb_train': []}
    for k in list(np.unique(metric_k_pd.k.values.tolist())):
        best_mm = metric_k_pd[(metric_k_pd[mm]==min(metric_k_pd[metric_k_pd.k==k][mm].values.tolist())) & (metric_k_pd.k==k)]
        best_k = best_mm.id_train.values.tolist()[0]
        path_best_k = path_folder + k + '/' + best_k + '/'
        print '\n\nBest Trainer:' + path_best_k
        print best_mm

        for p in os.listdir(path_best_k):
            pkl_cur = path_best_k + p
            if '.pkl' in pkl_cur:
                with open(pkl_cur) as outfile:    
                    res_cur = pickle.load(outfile)
                    outfile.close()
                dct_tmp['Subject'].append(p.split('res_')[1].split(cc)[0])
                dct_tmp['metric'].append(res_cur[mm.split('_')[0]])
                dct_tmp['nb_train'].append(k)

    plot_pd = pd.DataFrame.from_dict(dct_tmp)
    for k in list(np.unique(metric_k_pd.k.values.tolist())):
        printplot_pd[plot_pd.nb_train==k].nb_train.values.tolist()





  # path_output_pkl = path_local + 'output_pkl_' + cc + '/0/'

  # dct_tmp = {'Subject': [], 'resol': [], 'metric': [], 'nb_train': []}
  # for file in os.listdir(path_output_pkl):
  #   path_cur = path_output_pkl + file
  #   if os.path.isfile(path_cur) and '.pkl' in file and 'best_' in file and mm in file:

  #     with open(path_cur) as outfile:    
  #       pd_cur = pickle.load(outfile)
  #       outfile.close()

  #     for pp in pd_cur['Subject'].values.tolist():
  #       dct_tmp['Subject'].append(pp)
  #     for rr in pd_cur['resol'].values.tolist():
  #       dct_tmp['resol'].append(rr)
  #     for m in pd_cur[mm].values.tolist():
  #       dct_tmp['metric'].append(m)
  #     for i in range(len(pd_cur[mm].values.tolist())):
  #       dct_tmp['nb_train'].append(file.split('_'+mm)[0].split('best_')[1])

  # pd_2plot = pd.DataFrame.from_dict(dct_tmp)
  # # print pd_2plot

  # nb_img_train_str_lst = ['01', '05', '10', '15', '20', '25']

  # if mm != 'zcoverage':
  #     if cc == 't2':
  #       y_lim_min, y_lim_max = 0.0, 20.0
  #     elif cc == 't1':
  #       y_lim_min, y_lim_max = 0.0, 5.0
  #     else:
  #       y_lim_min, y_lim_max = 0.0, 7.0

  # else:
  #     y_lim_min, y_lim_max = 55.0, 101.0

  # sns.set(style="whitegrid", palette="Set1", font_scale=1.3)
  # fig, axes = plt.subplots(1, 1, sharey='col', figsize=(8*6, 8))
  # palette_swarm = dict(patient = 'crimson', hc = 'darkblue')
  # a = plt.subplot(1, 1, 1)
  # sns.violinplot(x='nb_train', y='metric', data=pd_2plot, order=nb_img_train_str_lst, 
  #           inner="quartile", cut=0, 
  #           scale="count", sharey=True, color='white')
  #           # palette=sns.color_palette("Greens",10)[:6])
  # sns.swarmplot(x='nb_train', y='metric', data=pd_2plot, order=nb_img_train_str_lst, 
  #                     size=5, color='grey')
  
  # plt.ylim([y_lim_min, y_lim_max])
  # a.set_ylabel('')
  # a.set_xlabel('')
  # plt.yticks(size=25)
  # fig.tight_layout()
  # # plt.show()
  # fig.savefig(path_local+'plot_nb_train_img_comparison/plot_comparison_' + cc + '_' + mm + '.png')
  # plt.close()
  



  # median_lst, nb_subj_lst, std_lst, extrm_lst = [], [], [], []
  # pvalue_lst, pvalue_hc_lst, pvalue_patient_lst, pvalue_iso_lst, pvalue_no_iso_lst = [], [], [], [], []

  # for i_f,f in enumerate(nb_img_train_str_lst):
  #   if f in pd_2plot.nb_train.values.tolist():
  #     values_cur = pd_2plot[pd_2plot.nb_train==f]['metric'].values.tolist()
  #     median_lst.append(np.median(values_cur))
  #     nb_subj_lst.append(len(values_cur))
  #     std_lst.append(np.std(values_cur))
  #     if mm == 'zcoverage':
  #       extrm_lst.append(min(values_cur))
  #     else:
  #       extrm_lst.append(max(values_cur))

  #     if f != nb_img_train_str_lst[-1]:
  #       values_cur_next = pd_2plot[pd_2plot.nb_train==nb_img_train_str_lst[i_f+1]]['metric'].values.tolist()
  #       pvalue_lst.append(ttest_ind(values_cur, values_cur_next)[1])

  #       values_cur_hc = pd_2plot[(pd_2plot.nb_train==f) & (pd_2plot.Subject=='hc')]['metric'].values.tolist()
  #       values_cur_hc_next = pd_2plot[(pd_2plot.nb_train==nb_img_train_str_lst[i_f+1]) & (pd_2plot.Subject=='hc')]['metric'].values.tolist()
      
  #       values_cur_patient = pd_2plot[(pd_2plot.nb_train==f) & (pd_2plot.Subject=='patient')]['metric'].values.tolist()
  #       values_cur_patient_next = pd_2plot[(pd_2plot.nb_train==nb_img_train_str_lst[i_f+1]) & (pd_2plot.Subject=='patient')]['metric'].values.tolist()
      
  #       values_cur_iso = pd_2plot[(pd_2plot.nb_train==f) & (pd_2plot.resol=='iso')]['metric'].values.tolist()
  #       values_cur_iso_next = pd_2plot[(pd_2plot.nb_train==nb_img_train_str_lst[i_f+1]) & (pd_2plot.resol=='iso')]['metric'].values.tolist()
      
  #       values_cur_not = pd_2plot[(pd_2plot.nb_train==f) & (pd_2plot.resol=='not')]['metric'].values.tolist()
  #       values_cur_not_next = pd_2plot[(pd_2plot.nb_train==nb_img_train_str_lst[i_f+1]) & (pd_2plot.resol=='not')]['metric'].values.tolist()
              
  #       pvalue_hc_lst.append(ttest_ind(values_cur_hc, values_cur_hc_next)[1])
  #       pvalue_patient_lst.append(ttest_ind(values_cur_patient, values_cur_patient_next)[1])
  #       pvalue_iso_lst.append(ttest_ind(values_cur_iso, values_cur_iso_next)[1])
  #       pvalue_no_iso_lst.append(ttest_ind(values_cur_not, values_cur_not_next)[1])

  #     else:
  #       for l in [pvalue_lst, pvalue_hc_lst, pvalue_patient_lst, pvalue_iso_lst, pvalue_no_iso_lst]:
  #         l.append(-1.0)
  #   else:
  #     for l in [median_lst, nb_subj_lst, std_lst, extrm_lst, pvalue_lst, pvalue_hc_lst, pvalue_patient_lst, pvalue_iso_lst, pvalue_no_iso_lst]:
  #       l.append(-1.0)

  # stats_pd = pd.DataFrame({'nb_train': nb_img_train_str_lst, 
  #                           'nb_test': nb_subj_lst,
  #                           'Median': median_lst,
  #                           'Std': std_lst,
  #                           'Extremum': extrm_lst,
  #                           'p-value': pvalue_lst,
  #                           'p-value_HC': pvalue_hc_lst,
  #                           'p-value_patient': pvalue_patient_lst,
  #                           'p-value_iso': pvalue_iso_lst,
  #                           'p-value_no_iso': pvalue_no_iso_lst                               
  #                           })


  # stats_pd.to_excel(path_local+'plot_nb_train_img_comparison/excel_' + cc + '_' + mm + '.xls', 
  #               sheet_name='sheet1')



def test_trainers_best(path_local, cc, mm, pp_ferg):

    path_folder_pkl = path_local + 'output_pkl_' + cc + '/'
    dct_tmp = {}
    for nn in os.listdir(path_folder_pkl):
        file_cur = path_folder_pkl + str(nn) + '.pkl'

        if os.path.isfile(file_cur):

          if nn != '0' and nn != '666':

            with open(file_cur) as outfile:    
              data_pd = pickle.load(outfile)
              outfile.close()

            if len(data_pd[data_pd[mm+'_med']==find_id_extr_df(data_pd, mm+'_med')[0]]['id'].values.tolist())>1:
              mm_avg = mm + '_moy'
            else:
              mm_avg = mm + '_med'

            val_best, fold_best = find_id_extr_df(data_pd, mm_avg)[0], find_id_extr_df(data_pd, mm_avg)[1]

            if mm == 'zcoverage':
                len_fail = len(data_pd[data_pd[mm+'_med']<= 90]['id'].values.tolist())
                len_sucess = len(data_pd[data_pd[mm+'_med']> 90]['id'].values.tolist())
                len_tot = len(data_pd['id'].values.tolist())
                print 'Percentage of trainer tel que avg. > 90% : ' + str(round(len_sucess*100.0/len_tot,2))

            elif mm == 'mse':
                len_fail = len(data_pd[data_pd[mm+'_med'] > 2]['id'].values.tolist())
                len_sucess = len(data_pd[data_pd[mm+'_med'] < 2]['id'].values.tolist())
                print len_sucess, len_fail 
                len_tot = len(data_pd['id'].values.tolist())
                print 'Percentage of trainer tel que avg. < 2 mm : ' + str(round(len_sucess*100.0/len_tot,2))

            print val_best, fold_best
            dct_tmp[nn] = [fold_best]

    path_input_train = path_local + 'input_train_' + cc + '_11/'
    path_input_train_best = path_input_train + '000/'

    create_folders_local([path_input_train, path_input_train_best])

    for nn in dct_tmp:
        path_input = path_local + 'input_train_' + cc + '_' + str(nn) + '/'
        for fold in os.listdir(path_input):
          if os.path.isfile(path_input + fold + '/' + str(dct_tmp[nn][0]).zfill(3) + '.txt'):
            file_in = path_input + fold + '/' + str(dct_tmp[nn][0]).zfill(3) + '.txt'
            file_seg_in = path_input + fold + '/' + str(dct_tmp[nn][0]).zfill(3) + '_ctr.txt'

            file_out = path_input_train_best + '0_' + str(nn).zfill(3) + '.txt'
            file_seg_out = path_input_train_best + '0_' + str(nn).zfill(3) + '_ctr.txt'        

            shutil.copyfile(file_in, file_out)
            shutil.copyfile(file_seg_in, file_seg_out)

    with open(path_local + 'test_valid_' + cc + '.pkl') as outfile:    
      data_pd = pickle.load(outfile)
      outfile.close()

    valid_lst = data_pd[data_pd.valid_test == 'train']['subj_name'].values.tolist()
    test_lst = data_pd[data_pd.valid_test == 'test']['subj_name'].values.tolist()

    with open(path_local + 'dataset_lst_' + cc + '.pkl') as outfile:    
      data_dct = pickle.load(outfile)
      outfile.close()

    valid_data_lst, test_data_lst = [], []
    for dd in data_dct:
        ok_bool = 0
        for tt in test_lst:
          if tt in dd and not ok_bool:
            test_data_lst.append(dd)
            ok_bool = 1
        for vv in valid_lst:
          if vv in dd and not ok_bool:
            valid_data_lst.append(dd)
            ok_bool = 1

    send_data2ferguson(path_local, pp_ferg, cc, 11, test_data_lst, path_input_train)

# ******************************************************************************************


def plot_comparison_clf(path_local, pd_tot, cc_lst):

    for clf in ['hough', 'optic']:
        for mm_name in ['mse', 'zcoverage', 'dice']:

            if mm_name in pd_tot:
                pd_2plot = pd_tot[pd_tot.algo==clf]

                if mm_name == 'mse':
                    y_lim_min, y_lim_max = -1, 50
                elif mm_name == 'dice':
                    y_lim_min, y_lim_max = -0.01, 1.01
                else:
                    y_lim_min, y_lim_max = -1, 101

                sns.set(style="whitegrid", palette="Set1", font_scale=1.3)

                fig, axes = plt.subplots(1, 1, sharey='col', figsize=(32, 16))
                fig.subplots_adjust(left=0.05, bottom=0.05)

                palette_violon = dict(t2='navajowhite', t1='skyblue', t2s='mediumaquamarine', dmri='lightpink')
                palette_swarm = dict(HC = 'royalblue', patho = 'firebrick')

                a = plt.subplot(1, 1, 1)
                sns.violinplot(x='contrast', y=mm_name, data=pd_2plot, order=cc_lst,
                                      inner=None, cut=0, scale="count",
                                      sharey=True,  palette=palette_violon, bw=0.15)
                # a_violon = sns.violinplot(x='contrast', y=mm_name, data=pd_2plot, order=cc_lst,
                #                       inner=None, cut=0, scale="count",
                #                       sharey=True,  bw=0.15, hue='patho', split=True)
                a_swarm = sns.swarmplot(x='contrast', y=mm_name, data=pd_2plot, order=cc_lst, hue='patho',
                                        size=8, palette=palette_swarm)

                a_swarm.legend_.remove()
                # a_violon.legend_.remove()

                a.set_ylim([y_lim_min,y_lim_max])
                plt.yticks(size=30)
                a.set_ylabel('')    
                a.set_xlabel('')

                plt.show()
                fig.tight_layout()
                fig.savefig(path_local+'plots/'+clf+'_'+mm_name+'_bis.png')
                plt.close()

                
                for cc in cc_lst:
                    print '***** ' + clf + '_' + mm_name + '_' + cc + ' *****'
                    pd_2mean = pd_2plot[pd_2plot.contrast==cc]
                    values = pd_2mean[mm_name].values.tolist()
                    values = [v for v in values if str(v)!='nan']
                    print 'Mean:' + str(round(np.mean(values),2))
                    print 'Std:' + str(round(np.std(values),2))
                    values_hc = pd_2mean[pd_2mean.patho=='HC'][mm_name].values.tolist()
                    values_hc = [v for v in values_hc if str(v)!='nan']
                    print 'Mean_HC:' + str(round(np.mean(values_hc),2))
                    print 'Std_HC:' + str(round(np.std(values_hc),2))
                    values_patho = pd_2mean[pd_2mean.patho=='patho'][mm_name].values.tolist()
                    values_patho = [v for v in values_patho if str(v)!='nan']
                    print 'Mean_patho:' + str(round(np.mean(values_patho),2))
                    print 'Std_patho:' + str(round(np.std(values_patho),2))

def compute_dice(path_local, nb_img=1):

    res_concat = []
    for cc in ['t2', 't1', 't2s']:
        path_seg_out = path_local + 'output_svm_propseg_' + cc + '/'
        path_seg_out = path_local + 'output_svm_propseg_' + cc + '/'
        path_seg_out_propseg = path_local + 'output_propseg_' + cc + '/'
        path_data = path_local + 'input_nii_' + cc + '/'
        path_seg_out_cur = path_seg_out + str(nb_img) + '/'
        path_seg_out_propseg = path_local + 'output_propseg_' + cc + '/'

        fname_out_pd = path_seg_out + str(nb_img) + '.pkl'

        if not os.path.isfile(fname_out_pd):
            with open(path_local + 'test_valid_' + cc + '.pkl') as outfile:    
                train_test_pd = pickle.load(outfile)
                outfile.close()

            res_pd = train_test_pd[train_test_pd.valid_test=='test'][['patho', 'resol', 'subj_name']]
            subj_name_lst = res_pd.subj_name.values.tolist()
            res_pd['contrast'] = [cc for i in range(len(subj_name_lst))]

            res_pd['dice_svm'] = [0.0 for i in range( len(subj_name_lst))]
            # res_pd[' '] = [' ' for i in range(len(subj_name_lst))]
            for file in os.listdir(path_seg_out_cur):
                if file.endswith('.nii.gz'):
                    file_src = path_seg_out_cur+file
                    file_dst = path_data + file
                    subj_id = file.split('_'+cc)[0]
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
                    subj_id = file.split('_'+cc)[0]
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
            res_pd.to_pickle(fname_out_pd)

        else:
            with open(fname_out_pd) as outfile:    
                res_pd = pickle.load(outfile)
                outfile.close()

        res_concat.append(res_pd)

        stg_propseg = 'Mean = ' + str(round(np.mean(res_pd.dice_propseg.values.tolist()),2))
        stg_propseg += '\nStd = ' + str(round(np.std(res_pd.dice_propseg.values.tolist()),2))
        stg_propseg += '\nMedian = ' + str(round(np.median(res_pd.dice_propseg.values.tolist()),2))
        stg_svm = 'Mean = ' + str(round(np.mean(res_pd.dice_svm.values.tolist()),2))
        stg_svm += '\nStd = ' + str(round(np.std(res_pd.dice_svm.values.tolist()),2))
        stg_svm += '\nMedian = ' + str(round(np.median(res_pd.dice_svm.values.tolist()),2))
        stg_svm += '\nMin = ' + str(round(np.min(res_pd.dice_svm.values.tolist()),2))
        stg_svm += '\nttest = ' + str(ttest_ind(res_pd.dice_propseg.values.tolist(), 
                                        res_pd.dice_svm.values.tolist())[1])
        
        print '\n\n' + cc
        print '\nOptiC:'
        print stg_svm
        print '\nPropSeg:'
        print stg_propseg
        # print len(res_pd.dice_svm.values.tolist())
        # print res_pd[res_pd.dice_svm<0.50]


    # res_tot = pd.concat(res_concat)
    # print res_tot
    # stg_propseg = 'Mean = ' + str(round(np.mean(res_tot.dice_propseg.values.tolist()),2))
    # stg_propseg += '\nStd = ' + str(round(np.std(res_tot.dice_propseg.values.tolist()),2))
    # stg_propseg += '\nMedian = ' + str(round(np.median(res_tot.dice_propseg.values.tolist()),2))
    # stg_svm = 'Mean = ' + str(round(np.mean(res_tot.dice_svm.values.tolist()),2))
    # stg_svm += '\nStd = ' + str(round(np.std(res_tot.dice_svm.values.tolist()),2))
    # stg_svm += '\nMedian = ' + str(round(np.median(res_tot.dice_svm.values.tolist()),2))
    # stg_svm += '\nMin = ' + str(round(np.min(res_tot.dice_svm.values.tolist()),2))
    
    # print '\nOptiC:'
    # print stg_svm
    # print '\nPropSeg:'
    # print stg_propseg

    # stg_propseg = 'Mean = ' + str(round(np.mean(res_tot[res_tot.patho=='patient'].dice_propseg.values.tolist()),2))
    # stg_propseg += '\nStd = ' + str(round(np.std(res_tot[res_tot.patho=='patient'].dice_propseg.values.tolist()),2))
    # stg_propseg += '\nMedian = ' + str(round(np.median(res_tot[res_tot.patho=='patient'].dice_propseg.values.tolist()),2))
    # stg_svm = 'Mean = ' + str(round(np.mean(res_tot[res_tot.patho=='patient'].dice_svm.values.tolist()),2))
    # stg_svm += '\nStd = ' + str(round(np.std(res_tot[res_tot.patho=='patient'].dice_svm.values.tolist()),2))
    # stg_svm += '\nMedian = ' + str(round(np.median(res_tot[res_tot.patho=='patient'].dice_svm.values.tolist()),2))
    # stg_svm += '\nMin = ' + str(round(np.min(res_tot[res_tot.patho=='patient'].dice_svm.values.tolist()),2))
    
    # print '\nOptiC:'
    # print stg_svm
    # print '\nPropSeg:'
    # print stg_propseg

    # y_lim_min, y_lim_max = -0.01, 1.01
    # sns.set(style="whitegrid", palette="Set1", font_scale=1.3)
    # fig, axes = plt.subplots(1, 2, sharey='col', figsize=(24, 8))
    # fig.subplots_adjust(left=0.05, bottom=0.05)
    # order_lst=['t2', 't1', 't2s']

    # for ii, dd in enumerate(['dice_propseg', 'dice_svm']):
    #     if dd == 'dice_propseg':
    #         palette_violon = dict(t2='gold', t1='royalblue', t2s='mediumvioletred')
    #     else:
    #         palette_violon = dict(t2='khaki', t1='cornflowerblue', t2s='pink')
    #     a = plt.subplot(1, 2, ii+1)
    #     sns.violinplot(x='contrast', y=dd, data=res_tot, order=order_lst,
    #                           inner="quartile", cut=0, scale="count",
    #                           sharey=True,  palette=palette_violon)

    #     a_swarm = sns.swarmplot(x='contrast', y=dd, data=res_tot, 
    #                             order=order_lst, size=3,
    #                             color=(0.2,0.2,0.2))
    #     # a_swarm.legend_.remove()

    #     a.set_ylim([y_lim_min,y_lim_max])
    #     plt.yticks(size=30)
    #     a.set_ylabel('')    
    #     a.set_xlabel('')

    # plt.show()
    # fig.tight_layout()
    # fig.savefig(path_local + 'plot_comparison/propseg_dice.png')
    # plt.close()

def computation_time_optic_dyn(path_optic, path_dyn, data_pd, path_data):

    time_optic_lst, time_dyn_lst = [], []
    subj_lst = data_pd.subj_name.values.tolist()
    for s in subj_lst:
        path_img_cur = path_data+s+'.nii.gz'
        img = Image(path_img_cur)
        nz_cur = img.dim[2]
        del img

        path_optic_cur = path_optic+s+'_time.txt'
        path_dyn_cur = path_dyn+s+'_time.txt'
        if os.path.isfile(path_optic_cur) and os.path.isfile(path_dyn_cur):
            time_optic_lst.append(float(open(path_optic_cur, 'r').read())/nz_cur)
            time_dyn_lst.append(float(open(path_dyn_cur, 'r').read())/nz_cur)

    return time_optic_lst, time_dyn_lst


def data_stats():


    c_dct = {'t2': 0.0, 't2s': 0.0, 't1': 0.0, 'dmri': 0.0}
    center_dct = {}
    patho_dct = {}
    resol_dct = {'iso': {'nb': 0, 'in_plane': [], 'thick': []},
                    'sag': {'nb': 0, 'in_plane': [], 'thick': []},
                    'ax': {'nb': 0, 'in_plane': [], 'thick': []}}

    for c in ['t2', 't2s', 't1', 'dmri']:
        with open(path_local_sdika + 'info_' + c + '.pkl') as outfile:    
            data_pd = pickle.load(outfile)
            outfile.close()
        # path_lst = [path_local_sdika + 'input_nii/' + c + '/' + s + '.nii.gz' for s in data_pd.subj_name.values.tolist()]
        # resol_dct_cur = info_resol(path_lst)
        # for r in resol_dct:
        #     if 'nb' in resol_dct_cur[r]:
        #         resol_dct[r]['nb'] += resol_dct_cur[r]['nb']
        #     if 'in_plane' in resol_dct[r] and 'in_plane' in resol_dct_cur[r]:
        #         for rr in resol_dct_cur[r]['in_plane']:
        #             resol_dct[r]['in_plane'].append(rr)
        #     if 'thick' in resol_dct[r] and 'thick' in resol_dct_cur[r]:
        #         for rr in resol_dct_cur[r]['thick']:
        #             resol_dct[r]['thick'].append(rr)

        c_dct[c] += len(data_pd['center'].values.tolist())
        center_lst_cur = data_pd['center'].values.tolist()
        for cc in np.unique(center_lst_cur):
            if cc not in center_dct:
                center_dct[cc] = []
            s_lst = data_pd[data_pd.center==cc]
            s_lst = [s.split('_'+c)[0] for s in s_lst.subj_name.values.tolist()]

            for s in s_lst:
                if not s in center_dct[cc]:
                    center_dct[cc].append(s)

        path_lst_cur = data_pd['patho'].values.tolist()
        for p in np.unique(path_lst_cur):
            if p not in patho_dct:
                patho_dct[p] = []
            s_lst = data_pd[data_pd.patho==p]
            s_lst = [s.split('_'+c)[0] for s in s_lst.subj_name.values.tolist()]

            for s in s_lst:
                if not s in patho_dct[p]:
                    patho_dct[p].append(s)


    # with open(path_local_sdika + 'resOl_tmp.pkl', 'wb') as f:
    #     pickle.dump(resol_dct, f)
    #     f.close()
    # print resol_dct

    

    keys_lst = center_dct.keys()
    print keys_lst
    for k in center_dct.keys():
        print '\n' + str(k) + ' : ' + str(len(center_dct[k]))
    print '\nContrast dct:'
    print c_dct
    print '\nNb of image: ' + str(sum([c_dct[cc] for cc in c_dct]))
    for p in patho_dct:
        print '\n' + p + ' : ' + str(len(patho_dct[p]))
    print patho_dct['unknown']
    print '\nNb of subj: ' + str(sum([len(patho_dct[p]) for p in patho_dct]))

    with open(path_local_sdika + 'resOl_tmp.pkl') as outfile:    
        resol_dct = pickle.load(outfile)
        outfile.close()
    for r in resol_dct:
        print '\n' + r + ' : ' + str(resol_dct[r]['nb'])

    for r in resol_dct:
        print '\n' + r
        thick = resol_dct[r]['thick']
        print min(thick), max(thick)
        in_plane = resol_dct[r]['in_plane']
        print min(in_plane), max(in_plane)

    order_lst = ['in_plane']
    dct_cur = {'orient': [], 'in_plane': []}
    for r in resol_dct:
        for i in resol_dct[r]['in_plane']:
            dct_cur['in_plane'].append(i)
            dct_cur['orient'].append('in_plane')


    
    cmpt_dct = dict(Counter(dct_cur['in_plane']))
    keys_lst = np.unique(dct_cur['in_plane'])
    for k in keys_lst:
        print '\n' + str(k) + ' : ' + str(cmpt_dct[k])


    pd_res = pd.DataFrame.from_dict(dct_cur)
    # print pd_res
    sns.set(style="white", font_scale=1.3)
    palette_violon = dict(in_plane='skyblue')           
    fig, axes = plt.subplots(1, 1, sharey='col', figsize=(16, 16))
    fig.subplots_adjust(left=0.05, bottom=0.05)
    a = plt.subplot(1, 1, 1)
    sns.violinplot(x='orient', y='in_plane', data=pd_res, order=order_lst,
                            inner=None, cut=0, scale="count",
                            sharey=True,  palette=palette_violon, bw=0.15)
    # sns.swarmplot(x='orient', y='in_plane', data=pd_res, order=order_lst, size=3,
    #                     color=(0.2,0.2,0.2))
    a.set_ylabel('')
    a.set_xlabel('')
    plt.yticks(size=16)
    plt.show()
    # fig.tight_layout()
    plt.close()



    order_lst = ['thick']
    dct_cur = {'orient': [], 'thick': []}
    for r in resol_dct:
        for i in resol_dct[r]['thick']:
            dct_cur['thick'].append(i)
            dct_cur['orient'].append('thick')

    cmpt_dct = dict(Counter(dct_cur['thick']))
    keys_lst = np.unique(dct_cur['thick'])
    for k in keys_lst:
        print '\n' + str(k) + ' : ' + str(cmpt_dct[k])

    pd_res = pd.DataFrame.from_dict(dct_cur)
    # print pd_res
    sns.set(style="white", font_scale=1.3)
    palette_violon = dict(thick='plum')           
    fig, axes = plt.subplots(1, 1, sharey='col', figsize=(16, 16))
    fig.subplots_adjust(left=0.05, bottom=0.05)
    a = plt.subplot(1, 1, 1)
    sns.violinplot(x='orient', y='thick', data=pd_res, order=order_lst,
                            inner=None, cut=0, scale="count",
                            sharey=True,  palette=palette_violon, bw=0.15)
    # sns.swarmplot(x='orient', y='thick', data=pd_res, order=order_lst, size=2.5,
    #                     color=(0.2,0.2,0.2))
    a.set_ylabel('')
    a.set_xlabel('')
    plt.yticks(size=16)
    plt.show()
    # fig.tight_layout()
    plt.close()

    print '\n\n'
    print 'ALPHA PLOT'
    print '\n\n'


def prediction_propseg(path_local, cc, testing_lst):

    path_nii = path_local + 'input_nii/' + cc + '/'
    path_output_nii_propseg = path_local + 'propseg_nii/' + cc + '/'
    path_gt = path_local + 'gold/' + cc + '/'
    create_folders_local([path_output_nii_propseg])

    if cc == 't2s':
        cc = 't2'
    elif cc == 'dmri':
        cc = 't1'

    path_nii2convert_lst = []
    for subject_name in testing_lst:
        fname_input = path_nii + subject_name + '.nii.gz'
        fname_output = path_output_nii_propseg + subject_name + '_centerline_pred.nii.gz'
        fname_gt = path_nii + subject_name + '_seg.nii.gz'
        if not os.path.isfile(fname_output):

            start = time.time()
            os.system('sct_propseg -i ' + fname_input + ' -c ' + cc + ' -ofolder ' + path_output_nii_propseg + ' -centerline-binary')
            end = time.time()
            with open(path_output_nii_propseg + subject_name + '_time.txt', 'w') as text_file:
                text_file.write(str(end-start))
                text_file.close()

            fname_seg = path_output_nii_propseg + subject_name + '_seg.nii.gz'
            if os.path.isfile(fname_seg):
                os.rename(path_output_nii_propseg + subject_name + '_centerline.nii.gz', fname_output)
            else:
                im_data = Image(fname_input)
                im_pred = im_data.copy()
                im_pred.data *= 0
                im_pred.setFileName(fname_output)
                im_pred.save()

            os.system('sct_image -i ' + fname_output + ' -setorient RPI -o ' + fname_output)
        
        fname_dice = path_output_nii_propseg + subject_name + '_dice.txt'
        if not os.path.isfile(fname_dice):
            fname_seg = path_output_nii_propseg + subject_name + '_seg.nii.gz'
            if not os.path.isfile(fname_seg):
                im_data = Image(fname_input)
                im_pred = im_data.copy()
                im_pred.data *= 0
                im_pred.setFileName(fname_seg)
                im_pred.save()
            os.system('sct_dice_coefficient -i ' + fname_gt + ' -d ' + fname_seg + ' -o ' + fname_dice)



def prediction_propseg_optic(path_local, cc, testing_lst, path_ctr):

    path_nii = path_local + 'input_nii/' + cc + '/'
    path_gt = path_local + 'gold/' + cc + '/'
    path_output_nii_propseg = path_local + 'propseg_optic_nii/' + cc + '/'
    create_folders_local([path_output_nii_propseg])

    if cc == 't2s':
        cc = 't2'
    elif cc == 'dmri':
        cc = 't1'

    path_nii2convert_lst = []
    for subject_name in testing_lst:
        fname_input = path_nii + subject_name + '.nii.gz'
        fname_ctr = path_ctr + subject_name + '_centerline_pred.nii.gz'
        fname_output = path_output_nii_propseg + subject_name + '_seg.nii.gz'
        fname_dice = path_output_nii_propseg + subject_name + '_dice.txt'
        fname_gt = path_nii + subject_name + '_seg.nii.gz'
        if not os.path.isfile(fname_output) and os.path.isfile(fname_ctr):
            start = time.time()
            os.system('sct_propseg -i ' + fname_input + ' -c ' + cc + ' -ofolder ' + path_output_nii_propseg + ' -init-centerline ' + fname_ctr)
            end = time.time()
            with open(path_output_nii_propseg + subject_name + '_time.txt', 'w') as text_file:
                text_file.write(str(end-start))
                text_file.close()
            os.remove(path_output_nii_propseg + subject_name + '_centerline.nii.gz')
            os.system('sct_image -i ' + fname_output + ' -setorient RPI -o ' + fname_output)
            os.system('sct_dice_coefficient -i ' + fname_gt + ' -d ' + fname_output + ' -o ' + fname_dice)


# ******************************************************************************************

def readCommand(  ):
    "Processes the command used to run from the command line"
    parser = argparse.ArgumentParser('Sdika Pipeline')
    parser.add_argument('-ofolder', '--output_folder', help='Output Folder', required = False)
    parser.add_argument('-c', '--contrast', help='Contrast of Interest', required = True)
    parser.add_argument('-n', '--nb_train_img', help='Nb Training Images', required = False)
    parser.add_argument('-s', '--step', help='Prepare (step=0) or Push (step=1) or Pull (step 2) or Compute metrics (step=3) or Display results (step=4)', 
                                        required = False)
    arguments = parser.parse_args()
    return arguments


USAGE_STRING = """
  USAGE:      python sct_sdika.py -ofolder '/Users/chgroc/data/data_sdika/' <options>
  EXAMPLES:   (1) python sct_sdika.py -ofolder '/Users/chgroc/data/data_sdika/' -c t2
                  -> Run Sdika Algorithm on T2w images dataset...
              (2) python sct_sdika.py -ofolder '/Users/chgroc/data/data_sdika/' -c t2 -n 3
                  -> ...Using 3 training images...
              (3) python sct_sdika.py -ofolder '/Users/chgroc/data/data_sdika/' -c t2 -n 3 -s 1
                  -> ...s=0 >> Prepare dataset
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
        create_folders_local([path_local_sdika])

        fname_local_script = '/Users/chgroc/spinalcordtoolbox/dev/sct_detect_spinalcord_sdika_ferguson.py'
        fname_local_script_rot = '/Users/chgroc/spinalcordtoolbox/dev/sct_detect_spinalcord_sdika_ferguson_rot.py'
        fname_local_script_lambda = '/Users/chgroc/spinalcordtoolbox/dev/sct_detect_spinalcord_sdika_ferguson_lambda.py'
        fname_local_script_test = '/Users/chgroc/spinalcordtoolbox/dev/sct_detect_spinalcord_sdika_ferguson_test.py'        
        fname_local_script_testing = '/Users/chgroc/spinalcordtoolbox/dev/sct_detect_spinalcord_sdika_ferguson_testing.py'        
        path_ferguson = '/home/neuropoly/code/spine-ms-tmi-dwi/'
        path_sct_testing_large = '/Volumes/Public_JCA/sct_testing/large/'

        # Format of parser arguments
        contrast_of_interest = str(parse_arg.contrast)
        if not parse_arg.step:
            step = 0
        else:
            step = int(parse_arg.step)
        if not parse_arg.nb_train_img:
            nb_train_img = 1
        else:
            nb_train_img = int(parse_arg.nb_train_img)         

        # Dataset description
        if not step:
            create_folders_local([path_local_sdika+'gold/', path_local_sdika+'input_img/', path_local_sdika+'input_nii/'])
            create_folders_local([path_local_sdika+'gold/'+contrast_of_interest+'/', 
                                  path_local_sdika+'input_img/'+contrast_of_interest+'/',
                                  path_local_sdika+'input_nii/'+contrast_of_interest+'/'])
            
            # if not os.path.isfile(path_local_sdika + 'info_' + contrast_of_interest + '.pkl'):
            panda_dataset(path_local_sdika, contrast_of_interest, path_sct_testing_large, 40)
            prepare_dataset(path_local_sdika, contrast_of_interest, path_sct_testing_large)

            # os.system('scp ' + fname_local_script + ' ferguson:' + path_ferguson)
            # os.system('scp -r ' + path_local_sdika+'input_img' + ' ferguson:' + path_ferguson)



        # Train k-Boostrap
        elif step == 1:

            list_k = [1, 5, 10, 15]

            create_folders_local([path_local_sdika+'output_img/',
                                  path_local_sdika+'output_nii/', 
                                  path_local_sdika+'output_pkl/',
                                  path_local_sdika+'input_train/'])
            create_folders_local([path_local_sdika+'output_img/'+contrast_of_interest+'/',
                                  path_local_sdika+'output_nii/'+contrast_of_interest+'/', 
                                  path_local_sdika+'output_pkl/'+contrast_of_interest+'/',
                                  path_local_sdika+'input_train/'+contrast_of_interest+'/'])
            for k in list_k:
              create_folders_local([path_local_sdika+'output_nii/'+contrast_of_interest+'/'+str(k)+'/', 
                                    path_local_sdika+'output_pkl/'+contrast_of_interest+'/'+str(k)+'/',
                                    path_local_sdika+'input_train/'+contrast_of_interest+'/'+contrast_of_interest+'_'+str(k)+'/'])

            prepare_train(path_local_sdika, path_ferguson, contrast_of_interest, list_k, 100)

        # Train k
        elif step == 2:
            with open(path_local_sdika + 'info_' + contrast_of_interest + '.pkl') as outfile:    
                data_pd = pickle.load(outfile)
                outfile.close()
            valid_lst = data_pd[data_pd.train_test == 'train']['subj_name'].values.tolist()
            
            path_local_train_cur = path_local_sdika+'input_train/'+contrast_of_interest+'/'+contrast_of_interest+'_'+str(nb_train_img)+'/'
            send_data2ferguson(path_local_sdika, path_ferguson, contrast_of_interest, 
                                nb_train_img, valid_lst, path_local_train_cur)

        # Pull k-Results from ferguson
        elif step == 3:
            create_folders_local([path_local_sdika+'output_time/'])
            create_folders_local([path_local_sdika+'output_time/'+contrast_of_interest+'/'])
            create_folders_local([path_local_sdika+'output_time/'+contrast_of_interest+'/'+str(nb_train_img)+'/'])
            pull_img_convert_nii_remove_img(path_local_sdika, path_ferguson, contrast_of_interest, nb_train_img)

        # Validation k-Results
        elif step == 4:
            compute_dataset_stats(path_local_sdika, contrast_of_interest, str(nb_train_img))

        # Boostrap results & ANOVA & Statistical Power
        elif step == 5:
            list_k = [1, 5, 10, 15]
            path_folder_pkl = path_local_sdika + 'output_pkl/' + contrast_of_interest + '/'
            boostrap_k_pd = panda_boostrap_k(path_folder_pkl, list_k)
            best_k_pd = panda_best_k(path_folder_pkl, boostrap_k_pd, list_k)
            col_names = ['mse_'+str(k) for k in list_k]
            # plot_boostrap_k(best_k_pd, col_names, contrast_of_interest)

            # print '\n3/ The population standard deviations of the groups are all equal.:'
            # print 'Bartlett Test p-value: ' + str(round(bartlett(best_k_pd.mse_1.values.tolist(), 
            #                                         best_k_pd.mse_5.values.tolist(),
            #                                         best_k_pd.mse_10.values.tolist(),
            #                                         best_k_pd.mse_15.values.tolist())[1],3))
            # print '\nANOVA p-value: ' + str(round(f_oneway(best_k_pd.mse_1.values.tolist(), 
            #                                         best_k_pd.mse_5.values.tolist(),
            #                                         best_k_pd.mse_10.values.tolist(),
            #                                         best_k_pd.mse_15.values.tolist())[1],3)) + '\n'

            print '\nThe Kruskal-Wallis H-test:'
            print 'Ho: the population median of all of the groups are equal.'
            print 'It is a non-parametric version of ANOVA.'
            print 'Note that rejecting the null hypothesis does not indicate which of the groups differs.'
            print 'Post-hoc comparisons between groups are required to determine which groups are different.'
            print '\nkruskal p-value: ' + str(round(kruskal(best_k_pd.mse_1.values.tolist(), 
                                                    best_k_pd.mse_5.values.tolist(),
                                                    best_k_pd.mse_10.values.tolist(),
                                                    best_k_pd.mse_15.values.tolist())[1],3)) + '\n'
            for k in col_names:
                print '\n' + k
                print 'Median=' + str(round(np.median(best_k_pd[k].values.tolist()),3))
                print 'Mean=' + str(round(np.mean(best_k_pd[k].values.tolist()),3))
                print 'Std=' + str(round(np.std(best_k_pd[k].values.tolist()),3)) 
                print 'NormalTest=' + str(normaltest(best_k_pd[k].values.tolist())[1])

            print '\nkruskal p-value between k=1 and k=5: ' + str(round(kruskal(best_k_pd.mse_1.values.tolist(), 
                                                best_k_pd.mse_5.values.tolist())[1],6)) + '\n'
            print '\nkruskal p-value between k=1 and k=10: ' + str(round(kruskal(best_k_pd.mse_1.values.tolist(), 
                                                best_k_pd.mse_10.values.tolist())[1],6)) + '\n'
            print '\nkruskal p-value between k=1 and k=15: ' + str(round(kruskal(best_k_pd.mse_1.values.tolist(), 
                                                best_k_pd.mse_15.values.tolist())[1],6)) + '\n'    

        # Rotation
        elif step == 6:

            with open(path_local_sdika + 'info_' + contrast_of_interest + '.pkl') as outfile:    
                data_pd = pickle.load(outfile)
                outfile.close()
            valid_lst = data_pd[data_pd.train_test == 'train']['subj_name'].values.tolist()
            
            if contrast_of_interest == 't1' or contrast_of_interest == 't2' or contrast_of_interest == 't2s' or contrast_of_interest == 'dmri':
                best_k = 1

            os.system('scp ' + fname_local_script_rot + ' ferguson:' + path_ferguson)

            path_local_train_cur = path_local_sdika+'input_train/'+contrast_of_interest+'/'+contrast_of_interest+'_'+str(best_k)+'/'
            
            send_data2ferguson(path_local_sdika, path_ferguson, contrast_of_interest, 
                                best_k, valid_lst, path_local_train_cur, True)

        # Pull Rotation from ferguson
        elif step == 7:
            if contrast_of_interest == 't1' or contrast_of_interest == 't2':
                best_k = 1
            rot2tested = ['0:360:0', '6:60:60', '12:60:60', '36:360:60', '72:360:60']
            for r in rot2tested:
                pull_img_rot(path_local_sdika, path_ferguson, contrast_of_interest, '_'.join(r.split(':')))

        # Validation k-Results
        elif step == 8:
            rot2tested = ['0:360:0', '6:60:60', '12:60:60', '36:360:60', '72:360:60']
            for r in rot2tested:
                compute_dataset_stats(path_local_sdika, contrast_of_interest, '_'.join(r.split(':')))

        # Rotation results
        elif step == 9:
            
            rot2tested = ['0:360:0', '6:60:60', '12:60:60', '36:360:60', '72:360:60']
            rot2tested = ['_'.join(r.split(':')) for r in rot2tested]

            path_folder_pkl = path_local_sdika + 'output_pkl/' + contrast_of_interest + '/'
            boostrap_rot_pd = panda_boostrap_k(path_folder_pkl, rot2tested)
            best_rot_pd = panda_best_rot(path_folder_pkl, boostrap_rot_pd, rot2tested)[0]
            col_names = ['mse_'+str(r) for r in rot2tested]
            # plot_boostrap_k(best_rot_pd, col_names, contrast_of_interest)

            # print '\nThe Kruskal-Wallis H-test:'
            # print 'Ho: the population median of all of the groups are equal.'
            # print 'It is a non-parametric version of ANOVA.'
            # print 'Note that rejecting the null hypothesis does not indicate which of the groups differs.'
            # print 'Post-hoc comparisons between groups are required to determine which groups are different.'
            # print '\nkruskal p-value: ' + str(round(kruskal(best_rot_pd[col_names[0]].values.tolist(), 
            #                                         best_rot_pd[col_names[1]].values.tolist(),
            #                                         best_rot_pd[col_names[2]].values.tolist(),
            #                                         best_rot_pd[col_names[3]].values.tolist(),
            #                                         best_rot_pd[col_names[4]].values.tolist())[1],3)) + '\n'
            for k in col_names:
                print '\n' + k
                print 'Median=' + str(round(np.median(best_rot_pd[k].values.tolist()),3))
                print 'Mean=' + str(round(np.mean(best_rot_pd[k].values.tolist()),3))
                print 'Std=' + str(round(np.std(best_rot_pd[k].values.tolist()),3))
                print normaltest(best_rot_pd[k].values.tolist())[1]

            for k in col_names[1:]:
                print '\nkruskal between 0:360:0 and ' + k + ': p-value='
                print kruskal(best_rot_pd[col_names[0]].values.tolist(), best_rot_pd[k].values.tolist())[1]

        # Lambda
        elif step == 10:

            with open(path_local_sdika + 'info_' + contrast_of_interest + '.pkl') as outfile:    
                data_pd = pickle.load(outfile)
                outfile.close()
            valid_lst = data_pd[data_pd.train_test == 'train']['subj_name'].values.tolist()
            
            if contrast_of_interest == 't1' or contrast_of_interest == 't2' or contrast_of_interest == 't2s' or contrast_of_interest == 'dmri':
                best_k = 1
                best_rot = '0:360:0'

            os.system('scp ' + fname_local_script_lambda + ' ferguson:' + path_ferguson)

            path_local_train_cur = path_local_sdika+'input_train/'+contrast_of_interest+'/'+contrast_of_interest+'_'+str(best_k)+'/'
            
            send_data2ferguson(path_local_sdika, path_ferguson, contrast_of_interest, 
                                best_k, valid_lst, path_local_train_cur, best_rot, True)

        # Pull Lambda from ferguson
        elif step == 11:
            if contrast_of_interest == 't1' or contrast_of_interest == 't2':
                best_k = 1
                best_rot = '0:360:0'
            lambda2tested = ['0-0', '0-3', '0-6', '1-0', '1-3', '1-6', '2-0', '4-0']
            for l in lambda2tested:
                pull_img_rot(path_local_sdika, path_ferguson, contrast_of_interest, l)

        # Validation Rotation-Results
        elif step == 12:
            lambda2tested = ['0-0', '0-3', '0-6', '1-0', '1-3', '1-6', '2-0', '4-0']
            for l in lambda2tested:
                compute_dataset_stats(path_local_sdika, contrast_of_interest, l)

        # Rotation results
        elif step == 13:
            
            lambda2tested = ['0-0', '0-3', '0-6', '1-0', '1-3', '1-6', '2-0', '4-0']

            path_folder_pkl = path_local_sdika + 'output_pkl/' + contrast_of_interest + '/'
            boostrap_lambda_pd = panda_boostrap_k(path_folder_pkl, lambda2tested)
            best_lambda_pd = panda_best_rot(path_folder_pkl, boostrap_lambda_pd, lambda2tested)[0]
            col_names = ['mse_'+str(l) for l in lambda2tested]
            # plot_boostrap_k(best_lambda_pd, col_names, contrast_of_interest)

            for k in col_names:
                print '\n' + k
                # print 'Median=' + str(round(np.median(best_lambda_pd[k].values.tolist()),3))
                print 'Mean=' + str(round(np.mean(best_lambda_pd[k].values.tolist()),3))
                print 'Std=' + str(round(np.std(best_lambda_pd[k].values.tolist()),3))
                # print normaltest(best_lambda_pd[k].values.tolist())[1]

            # # print '\nThe Kruskal-Wallis H-test:'
            # # print 'Ho: the population median of all of the groups are equal.'
            # # print 'It is a non-parametric version of ANOVA.'
            # # print 'Note that rejecting the null hypothesis does not indicate which of the groups differs.'
            # # print 'Post-hoc comparisons between groups are required to determine which groups are different.'
            # # print '\nkruskal p-value between No Regularization Vs. Lambda=1.0: ' + str(kruskal(best_lambda_pd[col_names[0]].values.tolist(),
            # #                                         best_lambda_pd[col_names[3]].values.tolist())[1]) + '\n'
            # # print '\nkruskal p-value Lambda=1.0 Vs. Lambda=4.0: ' + str(kruskal(best_lambda_pd[col_names[3]].values.tolist(),
            # #                                         best_lambda_pd[col_names[7]].values.tolist())[1]) + '\n'
            # # print '\nkruskal p-value Lambda=[0.3,0.6,1.0,1.3,1.6,2.0]: ' + str(kruskal(best_lambda_pd[col_names[1]].values.tolist(),
            # #                             best_lambda_pd[col_names[2]].values.tolist(),
            # #                             best_lambda_pd[col_names[3]].values.tolist(),
            # #                             best_lambda_pd[col_names[4]].values.tolist(),
            # #                             best_lambda_pd[col_names[5]].values.tolist(),
            # #                             best_lambda_pd[col_names[6]].values.tolist())[1]) + '\n'

            for k in col_names:
                print '\nkruskal between ' + col_names[3] + ' and ' + k + ': p-value='
                print kruskal(best_lambda_pd[col_names[3]].values.tolist(), best_lambda_pd[k].values.tolist())[1]

        elif step == 14:
            data_stats()

            # with open(path_local_sdika + 'info_' + contrast_of_interest + '.pkl') as outfile:    
            #     data_pd = pickle.load(outfile)
            #     outfile.close()

            # print data_pd[data_pd.patho != 'HC']

        elif step == 15:

            with open(path_local_sdika + 'info_' + contrast_of_interest + '.pkl') as outfile:    
                data_pd = pickle.load(outfile)
                outfile.close()
            valid_lst = data_pd[data_pd.train_test == 'test']['subj_name'].values.tolist()
            
            if contrast_of_interest == 't1' or contrast_of_interest == 't2' or contrast_of_interest == 't2s' or contrast_of_interest == 'dmri':
                best_k = 1
                best_rot = '0:360:0'
                best_lambda = '1-0'

            path_folder_pkl = path_local_sdika + 'output_pkl/' + contrast_of_interest + '/'
            boostrap_lambda_pd = panda_boostrap_k(path_folder_pkl, [best_lambda])
            best_trainer = panda_best_rot(path_folder_pkl, boostrap_lambda_pd, [best_lambda])[1].split('/')[-2]
            # boostrap_lambda_pd = panda_boostrap_k(path_folder_pkl, ['0_360_0'])
            # best_trainer = panda_best_rot(path_folder_pkl, boostrap_lambda_pd, ['0_360_0'])[1].split('/')[-2]
            print best_trainer

            os.system('scp ' + fname_local_script_testing + ' ferguson:' + path_ferguson)

            path_local_train_cur = path_local_sdika+'input_train/'+contrast_of_interest+'/'+contrast_of_interest+'_'+str(best_k)+'/'
            
            send_data2ferguson(path_local_sdika, path_ferguson, contrast_of_interest, 
                                best_k, valid_lst, path_local_train_cur, best_rot, best_lambda, best_trainer, True)

        elif step == 16:
            pull_img_test(path_local_sdika, path_ferguson, contrast_of_interest)
            compute_dataset_stats(path_local_sdika, contrast_of_interest, 'optic')
            
        elif step == 17:

            cc_lst = ['t2', 't1', 't2s', 'dmri']
            for cc in cc_lst:     
                with open(path_local_sdika + 'info_' + cc + '.pkl') as outfile:    
                    data_pd = pickle.load(outfile)
                    outfile.close()
                valid_lst = data_pd[data_pd.train_test == 'test']['subj_name'].values.tolist()
                prediction_propseg(path_local_sdika, cc, valid_lst)
                # path_hough_pkl = path_local_sdika + 'output_pkl/' + cc + '/hough/'
                # create_folders_local([path_hough_pkl])
                # compute_dataset_stats(path_local_sdika, cc, 'hough')

        elif step == 18:

            cc_lst = ['t2', 't1', 't2s', 'dmri']
            pd_lst = []
            for cc in cc_lst:
                with open(path_local_sdika + 'info_' + cc + '.pkl') as outfile:    
                    data_pd = pickle.load(outfile)
                    outfile.close()
                path_optic_pkl = path_local_sdika + 'output_pkl/' + cc + '/optic/'
                path_hough_pkl = path_local_sdika + 'output_pkl/' + cc + '/hough/'
                pd_lst.append(panda_testing(path_optic_pkl, path_hough_pkl, data_pd, cc))
            pd_tot = pd.concat(pd_lst)

            for m in ['mse', 'zcoverage']:
                print '\n' + m
                z_optic = pd_tot[(pd_tot.algo=='optic')][m].values.tolist()
                z_hough = pd_tot[(pd_tot.algo=='hough')][m].values.tolist()
                z_hough = [z for z in z_hough if str(z) != 'nan']
                print np.mean(z_optic), np.std(z_optic), normaltest(z_optic)[1]
                print np.mean(z_hough), np.std(z_hough), normaltest(z_hough)[1]
                if len(z_optic)<len(z_hough):
                    random.shuffle(z_hough)
                    z_hough = z_hough[:len(z_optic)]
                elif len(z_optic)>len(z_hough):
                    random.shuffle(z_optic)
                    z_optic = z_optic[:len(z_hough)]
                print len(z_optic), len(z_hough)
                print wilcoxon(z_optic, z_hough)[1]


            # plot_comparison_clf(pd_tot, cc_lst)

        elif step == 19:
            cc_lst = ['t2', 't1', 't2s', 'dmri']
            time_optic, time_dyn = [], []
            for cc in cc_lst:
                with open(path_local_sdika + 'info_' + cc + '.pkl') as outfile:    
                    data_pd = pickle.load(outfile)
                    outfile.close()
                data_pd = data_pd[data_pd.train_test == 'test']
                path_optic_time = path_local_sdika + 'output_time/' + cc + '/optic/'
                path_dyn_time = path_local_sdika + 'output_time/' + cc + '/dyn/'
                path_data = path_local_sdika + 'input_nii/' + cc + '/'
                time_optic_cur, time_dyn_cur = computation_time_optic_dyn(path_optic_time, path_dyn_time, data_pd, path_data)
                time_optic.append(time_optic_cur)
                time_dyn.append(time_dyn_cur)
                print cc
                print np.mean(time_optic_cur)
                print np.std(time_optic_cur)
                print np.mean(time_dyn_cur)
                print np.std(time_dyn_cur)
                print ' '

            time_optic = [l for ll in time_optic for l in ll]
            time_dyn = [l for ll in time_dyn for l in ll]

            for ll in [time_optic, time_dyn]:
                print '\nMean=' + str(round(np.mean(ll),3))
                print 'Std=' + str(round(np.std(ll),3))
                print 'NormalTest=' + str(normaltest(ll)[1])
            
            print '\nBartlett Test p-value: ' + str(round(bartlett(time_optic, time_dyn)[1],3))

            print '\nANOVA p-value: ' + str(f_oneway(time_optic, time_dyn)[1]) + '\n'

            print '\nkruskal p-value: ' + str(kruskal(time_optic, time_dyn)[1]) + '\n'

        elif step == 20:

            dct_path = {'t2': '002', 't2s': '023', 't1': '069', 'dmri': '074'}
            for cc in ['t2', 't1', 't2s', 'dmri']:
                with open(path_local_sdika + 'info_' + cc + '.pkl') as outfile:    
                    data_pd = pickle.load(outfile)
                    outfile.close()
                valid_lst = data_pd[data_pd.train_test == 'test']['subj_name'].values.tolist()
                path_ctr = path_local_sdika + 'output_nii/' + cc + '/optic/' + dct_path[cc] + '/'
                prediction_propseg_optic(path_local_sdika, cc, valid_lst, path_ctr)

        elif step == 21:

            cc_lst = ['t2', 't1', 't2s', 'dmri']
            # pd_lst = []
            # for cc in cc_lst:
            #     with open(path_local_sdika + 'info_' + cc + '.pkl') as outfile:    
            #         data_pd = pickle.load(outfile)
            #         outfile.close()
            #     data_pd = data_pd[data_pd.train_test == 'test']
            #     path_optic_dice_time = path_local_sdika + 'propseg_optic_nii/' + cc + '/'
            #     path_propseg_dice_time = path_local_sdika + 'propseg_nii/' + cc + '/'
            #     path_data = path_local_sdika + 'input_nii/' + cc + '/'
            #     pd_lst.append(panda_testing_seg(path_data, path_optic_dice_time, data_pd, cc, 'optic'))
            #     pd_lst.append(panda_testing_seg(path_data, path_propseg_dice_time, data_pd, cc, 'hough'))
            # pd_tot = pd.concat(pd_lst)

            # with open(path_local_sdika + 'dice_time.pkl', 'wb') as f:
            #     pickle.dump(pd_tot, f)
            #     f.close()
            with open(path_local_sdika + 'dice_time.pkl') as outfile:    
                data_pd = pickle.load(outfile)
                outfile.close()

            # for m in ['dice', 'time']:
            #     print '\n' + m
            #     z_optic = data_pd[(data_pd.algo=='optic')][m].values.tolist()
            #     z_hough = data_pd[(data_pd.algo=='hough')][m].values.tolist()
            #     print np.mean(z_optic), np.std(z_optic), normaltest(z_optic)[1], len(z_optic)
            #     print np.mean(z_hough), np.std(z_hough), normaltest(z_hough)[1], len(z_hough)
            #     if len(z_optic)<len(z_hough):
            #         random.shuffle(z_hough)
            #         z_hough = z_hough[:len(z_optic)]
            #     elif len(z_optic)>len(z_hough):
            #         random.shuffle(z_optic)
            #         z_optic = z_optic[:len(z_hough)]
            #     print len(z_optic), len(z_hough)
            #     print wilcoxon(z_optic, z_hough)[1]

            # print data_pd[(data_pd.dice<0.5)&(data_pd.algo=='optic')]
            plot_comparison_clf(path_local_sdika, data_pd, cc_lst)

        elif step == 22:
            # list_cc = ['t1', 't2', 't2s', 'dmri']
            list_cc = ['dmri']
            dict_hyper = {
                            # 'k': [1, 5, 10, 15],
                            # 'rot': ['0_360_0', '6_60_60', '12_60_60', '36_360_60', '72_360_60'],
                            'lambda': ['0-0', '0-3', '0-6', '1-0', '1-3', '1-6', '2-0', '4-0']
                        }
            l_stg = ['0.0', '0.3', '0.6', '1.0', '1.3', '1.6', '2.0', '4.0']

            for c in list_cc:
                print '\nContrast'
                for h in dict_hyper:
                    path_folder_pkl = path_local_sdika + 'output_pkl/' + c + '/'
                    print dict_hyper[h]
                    boostrap_k_pd = panda_boostrap_k(path_folder_pkl, dict_hyper[h])
                    if h == 'k':
                        best_k_pd = panda_best_k(path_folder_pkl, boostrap_k_pd, dict_hyper[h])
                    else:
                        best_k_pd = panda_best_rot(path_folder_pkl, boostrap_k_pd, dict_hyper[h])[0]
                    col_names = ['mse_'+str(k) for k in dict_hyper[h]]

                    dct_tmp = {'param': [], 'value': []}

                    for i_k, k in enumerate(col_names):
                        for i in range(len(best_k_pd[k].values.tolist())):
                            dct_tmp['value'].append(best_k_pd[k].values.tolist()[i])
                            if h == 'lambda':
                                dct_tmp['param'].append(l_stg[i_k])
                            else:
                                dct_tmp['param'].append(k.split('mse_')[1].zfill(2))

                    # print '\nkruskal p-value between k=1 and k=5: ' + str(round(kruskal(best_k_pd.mse_1.values.tolist(), 
                    #                                     best_k_pd.mse_5.values.tolist())[1],6)) + '\n'
                    # print '\nkruskal p-value between k=1 and k=10: ' + str(round(kruskal(best_k_pd.mse_1.values.tolist(), 
                    #                                     best_k_pd.mse_10.values.tolist())[1],6)) + '\n'
                    # print '\nkruskal p-value between k=1 and k=15: ' + str(round(kruskal(best_k_pd.mse_1.values.tolist(), 
                    #                                     best_k_pd.mse_15.values.tolist())[1],6)) + '\n'

                    pd2plot = pd.DataFrame.from_dict(dct_tmp)

                    sns.set(style="whitegrid", font_scale=1.3)
                    fig, axes = plt.subplots(1, 1, sharey='col', figsize=(10, 10))
                    fig.subplots_adjust(left=0.1, bottom=0.05)
                    a = plt.subplot(1, 1, 1)
                    ax = sns.pointplot(x='param', y='value', data=pd2plot, ci=68, linestyles=[' '], color='firebrick')
                    a.set_ylabel('')
                    a.set_xlabel('')
                    plt.yticks(size=30)
                    plt.xticks(size=30)
                    if h=='lambda':
                        if c == 't2':
                            ax.set_ylim([0.9, 1.6])
                        elif c == 't2s':
                            ax.set_ylim([0.8, 1.6])
                        elif c == 't1':
                            ax.set_ylim([0.8, 1.1])
                        elif c == 'dmri':
                            ax.set_ylim([0.8, 1.3])
                    # plt.show()
                    fig.savefig(path_local_sdika + 'plots/' + c + '_' + h +'.png')
                    if h=='lambda':
                        print np.mean(pd2plot[pd2plot.param=='0.0'].value.values.tolist())
                        if c == 't2':
                            ax.set_ylim([124.7, 124.8])
                        elif c == 't2s':
                            ax.set_ylim([113.7, 113.8])
                        elif c == 't1':
                            ax.set_ylim([137.2, 137.3])
                        elif c == 'dmri':
                            ax.set_ylim([79.1, 79.2])
                        fig.savefig(path_local_sdika + 'plots/' + c + '_' + h +'_bis.png')
                    plt.close()
                    





        # elif step == 15:
        #     with open(path_local_sdika + 'info_' + contrast_of_interest + '.pkl') as outfile:    
        #         data_pd = pickle.load(outfile)
        #         outfile.close()
        #     valid_lst = data_pd[data_pd.train_test == 'train']['subj_name'].values.tolist()
            
        #     if contrast_of_interest == 't1' or contrast_of_interest == 't2' or contrast_of_interest == 't2s' or contrast_of_interest == 'dmri':
        #         best_k = 1
        #         best_rot = '0:360:0'

        #     os.system('scp ' + fname_local_script_test + ' ferguson:' + path_ferguson)

        #     path_local_train_cur = path_local_sdika+'input_train/'+contrast_of_interest+'/'+contrast_of_interest+'_'+str(best_k)+'/'
            
        #     send_data2ferguson(path_local_sdika, path_ferguson, contrast_of_interest, 
        #                         best_k, valid_lst, path_local_train_cur, best_rot, True)

        # elif step == 16:
        #     # if contrast_of_interest == 't1' or contrast_of_interest == 't2':
        #     #     best_k = 1
        #     #     best_rot = '0:360:0'
        #     lambda2tested = ['0-6']
        #     # for l in lambda2tested:
        #     #     pull_img_rot(path_local_sdika, path_ferguson, contrast_of_interest, l)
        #     #     compute_dataset_stats(path_local_sdika, contrast_of_interest, l)
        #     path_folder_pkl = path_local_sdika + 'output_pkl/' + contrast_of_interest + '/'
        #     boostrap_lambda_pd = panda_boostrap_k(path_folder_pkl, lambda2tested)
        #     best_lambda_pd = panda_best_rot(path_folder_pkl, boostrap_lambda_pd, lambda2tested)

        # # TEST
        # elif step == 14:

        #     with open(path_local_sdika + 'info_' + contrast_of_interest + '.pkl') as outfile:    
        #         data_pd = pickle.load(outfile)
        #         outfile.close()
        #     valid_lst = data_pd[data_pd.train_test == 'test']
        #     valid_lst = valid_lst[valid_lst.patho != 'HC']['subj_name'].values.tolist()
        #     if contrast_of_interest == 't1':
        #         best_k = 1
        #         best_rot = '0:360:0'

        #     os.system('scp ' + fname_local_script_test + ' ferguson:' + path_ferguson)

        #     path_local_train_cur = path_local_sdika+'input_train/'+contrast_of_interest+'/'+contrast_of_interest+'_'+str(best_k)+'/'
            
        #     send_data2ferguson(path_local_sdika, path_ferguson, contrast_of_interest, 
        #                         best_k, valid_lst, path_local_train_cur, best_rot, True)

        # # Pull Lambda from ferguson
        # elif step == 15:
        #     if contrast_of_interest == 't1':
        #         best_k = 1
        #         best_rot = '0:360:0'
        #     lambda2tested = ['1-0', '4-0']
        #     for l in lambda2tested:
        #         pull_img_rot(path_local_sdika, path_ferguson, contrast_of_interest, l)
        #         compute_dataset_stats(path_local_sdika, contrast_of_interest, l)

        # elif step == 16:
        #     lambda2tested = ['1-0', '4-0']

        #     path_folder_pkl = path_local_sdika + 'output_pkl/' + contrast_of_interest + '/'
        #     boostrap_lambda_pd = panda_boostrap_k(path_folder_pkl, lambda2tested)
        #     best_lambda_pd = panda_best_rot(path_folder_pkl, boostrap_lambda_pd, lambda2tested)
        #     col_names = ['mse_'+str(l) for l in lambda2tested]
        #     plot_boostrap_k(best_lambda_pd, col_names)

        #     for k in col_names:
        #         print '\n' + k
        #         print 'Median=' + str(round(np.median(best_lambda_pd[k].values.tolist()),3))
        #         print 'Mean=' + str(round(np.mean(best_lambda_pd[k].values.tolist()),3))
        #         print 'Std=' + str(round(np.std(best_lambda_pd[k].values.tolist()),3))


        #     print '\nThe Kruskal-Wallis H-test:'
        #     print 'Ho: the population median of all of the groups are equal.'
        #     print 'It is a non-parametric version of ANOVA.'
        #     print 'Note that rejecting the null hypothesis does not indicate which of the groups differs.'
        #     print 'Post-hoc comparisons between groups are required to determine which groups are different.'
        #     print '\nkruskal p-value Lambda=1.0 Vs. Lambda=4.0: ' + str(kruskal(best_lambda_pd[col_names[0]].values.tolist(),
        #                                             best_lambda_pd[col_names[1]].values.tolist())[1]) + '\n'




        # # Lambda
        # elif step == 17:
        #     with open(path_local_sdika + 'info_' + contrast_of_interest + '.pkl') as outfile:    
        #         data_pd = pickle.load(outfile)
        #         outfile.close()
        #     test_lst = data_pd[data_pd.train_test == 'train']['subj_name'].values.tolist()
        #     test_lst = [t for t in test_lst if t.startswith('twh')]
        #     print len(test_lst)
            # path_local_gold = path_local_sdika + 'gold/' + contrast_of_interest + '/'
            # path_local_seg = path_local_sdika + 'input_nii/' + contrast_of_interest + '/'

            # dct_tmp = {}
            # for t in test_lst:
            #     dct_tmp[t] = {'0_360_0': [], '36_360_60': []}

            #     for r,rr in zip(['0_360_0/002', '36_360_60/064'],['0_360_0', '36_360_60']):
            #         path_local_nii = path_local_sdika + 'output_nii/' + contrast_of_interest + '/'+r+'/'
            #         path_cur_pred = path_local_nii + t + '_centerline_pred.nii.gz'
            #         path_cur_gold = path_local_gold + t + '_centerline_gold.nii.gz'
            #         path_cur_gold_seg = path_local_seg + t + '_seg.nii.gz'
            #         if os.path.isfile(path_cur_pred):

            #             img_pred = Image(path_cur_pred)
            #             img_true = Image(path_cur_gold)
            #             img_seg_true = Image(path_cur_gold_seg)

            #             for z in range(img_true.dim[2]):

            #                 if np.sum(img_true.data[:,:,z]):
            #                     x_true, y_true = [np.where(img_true.data[:,:,z] > 0)[i][0] 
            #                                         for i in range(len(np.where(img_true.data[:,:,z] > 0)))]
            #                     x_pred, y_pred = [np.where(img_pred.data[:,:,z] > 0)[i][0]
            #                                         for i in range(len(np.where(img_pred.data[:,:,z] > 0)))]

            #                     x_true, y_true = img_true.transfo_pix2phys([[x_true, y_true, z]])[0][0], img_true.transfo_pix2phys([[x_true, y_true, z]])[0][1]
            #                     x_pred, y_pred = img_pred.transfo_pix2phys([[x_pred, y_pred, z]])[0][0], img_pred.transfo_pix2phys([[x_pred, y_pred, z]])[0][1]

            #                     dct_tmp[t][rr].append(((x_true-x_pred))**2 + ((y_true-y_pred))**2)
            # with open(path_local_sdika + 'rot_' + contrast_of_interest + '.pkl', 'wb') as f:
            #     pickle.dump(dct_tmp, f)
            #     f.close()
            # with open(path_local_sdika + 'rot_' + contrast_of_interest + '.pkl') as outfile:    
            #     iii = pickle.load(outfile)
            #     outfile.close()
            # print iii


    ############################################    
        # # TEST
        # elif step == 11:

        #     if contrast_of_interest == 't1':
        #         best_k = 1
        #     with open(path_local_sdika + 'info_' + contrast_of_interest + '.pkl') as outfile:    
        #         data_pd = pickle.load(outfile)
        #         outfile.close()
        #     test_lst = data_pd[data_pd.train_test == 'test']['subj_name'].values.tolist()

        #     os.system('scp ' + fname_local_script_test + ' ferguson:' + path_ferguson)

        #     path_local_train_cur = path_local_sdika+'input_train/'+contrast_of_interest+'/'+contrast_of_interest+'_'+str(best_k)+'/'

        #     send_data2ferguson(path_local_sdika, path_ferguson, contrast_of_interest, 
        #                         nb_train_img, test_lst, path_local_train_cur, True)

        # # TEST
        # elif step == 12:
        #     if contrast_of_interest == 't1':
        #         best_k = 1
        #     rot2tested = ['0:360:0', '36:360:60']
        #     for r in rot2tested:
        #         pull_img_rot(path_local_sdika, path_ferguson, contrast_of_interest, r)

        #     for r in rot2tested:
        #         compute_dataset_stats(path_local_sdika, contrast_of_interest, '_'.join(r.split(':')))

        # # TEST
        # elif step == 13:
        #     rot2tested = ['0:360:0', '36:360:60']
        #     rot2tested = ['_'.join(r.split(':')) for r in rot2tested]

        #     path_folder_pkl = path_local_sdika + 'output_pkl/' + contrast_of_interest + '/'
        #     boostrap_rot_pd = panda_boostrap_k(path_folder_pkl, rot2tested)
        #     best_rot_pd = panda_best_rot(path_folder_pkl, boostrap_rot_pd, rot2tested)
        #     col_names = ['mse_'+str(r) for r in rot2tested]
        #     plot_boostrap_k(best_rot_pd, col_names)

        #     print '\nThe ANOVA test has important assumptions that must be satisfied:'
        #     print 'in order for the associated p-value to be valid.'
        #     print '\n1/ The samples are independent'
        #     print '\n2/ Each sample is from a normally distributed population:'
        #     print 'Normal Test p-values:'

        #     for k in col_names:
        #         print k
        #         print normaltest(best_rot_pd[k].values.tolist())[1]

        #     print '\n3/ The population standard deviations of the groups are all equal.:'
        #     print 'Bartlett Test p-value: ' + str(round(bartlett(best_rot_pd[col_names[0]].values.tolist(), 
        #                                             best_rot_pd[col_names[1]].values.tolist())[1],3))
        #     print '\nANOVA p-value: ' + str(round(f_oneway(best_rot_pd[col_names[0]].values.tolist(), 
        #                                             best_rot_pd[col_names[1]].values.tolist())[1],3)) + '\n'

        #     print '\nThe Kruskal-Wallis H-test:'
        #     print 'Ho: the population median of all of the groups are equal.'
        #     print 'It is a non-parametric version of ANOVA.'
        #     print 'Note that rejecting the null hypothesis does not indicate which of the groups differs.'
        #     print 'Post-hoc comparisons between groups are required to determine which groups are different.'
        #     print '\nkruskal p-value: ' + str(round(kruskal(best_rot_pd[col_names[0]].values.tolist(), 
        #                                             best_rot_pd[col_names[1]].values.tolist())[1],3)) + '\n'
        #     for k in col_names:
        #         print '\n' + k
        #         print 'Median=' + str(round(np.median(best_rot_pd[k].values.tolist()),3))
        #         print 'Mean=' + str(round(np.mean(best_rot_pd[k].values.tolist()),3))
        #         print 'Std=' + str(round(np.std(best_rot_pd[k].values.tolist()),3))

        #     print '\nkruskal between 0:360:0 and 36:360:60: p-value='
        #     print round(kruskal(best_rot_pd[col_names[0]].values.tolist(), best_rot_pd[col_names[1]].values.tolist())[1],3)

############################################






        # elif step == 5:
        #     # Pull Testing Results from ferguson
        #     pull_img_convert_nii_remove_img(path_local_sdika, path_ferguson, contrast_of_interest, 11)
        
        # elif step == 6:
        #     # Compute metrics / Evaluate performance of Sdika algorithm
        #     compute_dataset_stats(path_locpanda_best_k(path_folder_pkl, boostrap_k_pd)al_sdika, contrast_of_interest, 11)
        
        # elif step == 7:
        #     clf_lst = ['Hough', 'OptiC']
        #     nbimg_cc_dct = {'t2': 1, 't1': 1, 't2s': 1}
        #     # plot_comparison_clf(path_local_sdika, clf_lst, nbimg_cc_dct, '11')
        #     plot_comparison_clf_patho(path_local_sdika, clf_lst, nbimg_cc_dct, '11')

        # elif step == 8:
        #     prepare_svmhog(path_local_sdika, contrast_of_interest, '/home/neuropoly/code/spine-ms/', nb_train_img)

        # elif step == 9:
        #     create_folders_local([path_local_sdika+'output_img_'+contrast_of_interest+'_111/', 
        #                             path_local_sdika+'output_nii_'+contrast_of_interest+'_111/'])
        #     pull_img_convert_nii_remove_img(path_local_sdika, '/home/neuropoly/code/spine-ms/', 
        #                                         contrast_of_interest, 111)

        # elif step == 10:
        #     compute_dataset_stats(path_local_sdika, contrast_of_interest, 111)

        # elif step == 11:
        #     panda_trainer(path_local_sdika, contrast_of_interest)
        #     prepare_svmhog_test(path_local_sdika, contrast_of_interest, '/home/neuropoly/code/spine-ms/', 'mse')

        # elif step == 12:
        #     create_folders_local([path_local_sdika+'output_img_'+contrast_of_interest+'_1111/', 
        #         path_local_sdika+'output_nii_'+contrast_of_interest+'_1111/'])
        #     pull_img_convert_nii_remove_img(path_local_sdika, '/home/neuropoly/code/spine-ms/', 
        #                             contrast_of_interest, 1111)
        
        # elif step == 13:
        #     compute_dataset_stats(path_local_sdika, contrast_of_interest, 1111)
        
        # elif step == 14:
        #     clf_lst = ['SVM', 'OptiC']
        #     nbimg_cc_dct = {'t2': 1, 't1': 1, 't2s': 1}
        #     plot_comparison_clf(path_local_sdika, clf_lst, nbimg_cc_dct, '11')
        #     # plot_comparison_clf_patho(path_local_sdika, clf_lst, nbimg_cc_dct, '11')

        # elif step == 15:
        #     compute_dice(path_local_sdika)
        
        # elif step == 16:
        #     computation_time(path_local_sdika, contrast_of_interest)