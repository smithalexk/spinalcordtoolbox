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

import matplotlib.pyplot as plt
import seaborn as sns



def create_folders_local(folder2create_lst):

    for folder2create in folder2create_lst:
        if not os.path.exists(folder2create):
            os.makedirs(folder2create)

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

    output_file = open(path_local + 'cnn_dataset_lst_' + cc + '.pkl', 'wb')
    pickle.dump(fname_testing_img, output_file)
    output_file.close()

def prediction_propseg(path_local, cc):

    path_nii = path_local + 'input_nii_' + cc + '/'
    path_output_nii_propseg = path_local + 'propseg_nii_' + cc + '/'
    create_folders_local([path_output_nii_propseg])

    with open(path_local + 'cnn_dataset_lst_' + cc + '.pkl') as outfile:    
        testing_lst = pickle.load(outfile)
        outfile.close()

    if cc == 't2s':
        cc = 't2'

    path_nii2convert_lst = []
    for fname_img in testing_lst:
        subject_name = fname_img.split('.')[0]
        fname_input = path_nii + subject_name + '.nii.gz'
        fname_output = path_output_nii_propseg + subject_name + '_pred.nii.gz'

        os.system('sct_propseg -i ' + fname_input + ' -c ' + cc + ' -ofolder ' + path_output_nii_propseg + ' -centerline-binary')

        fname_seg = path_output_nii_propseg + subject_name + '_seg.nii.gz'
        if os.path.isfile(fname_seg):
            os.remove(fname_seg)
            os.rename(path_output_nii_propseg + subject_name + '_centerline.nii.gz', fname_output)
        else:
            im_data = Image(fname_input)
            im_pred = im_data.copy()
            im_pred.data *= 0
            im_pred.setFileName(fname_output)
            im_pred.save()

        os.system('sct_image -i ' + fname_output + ' -setorient RPI -o ' + fname_output)

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

    print len(mse_dist), count_slice
    if len(mse_dist):
        stats_dct['mse'] = sqrt(sum(mse_dist)/float(len(mse_dist)))
        stats_dct['maxmove'] = sqrt(max(mse_dist))
        stats_dct['zcoverage'] = float(slice_coverage*100.0)/count_slice
    else:
        stats_dct['mse'] = None
        stats_dct['maxmove'] = None
        stats_dct['zcoverage'] = None

    return stats_dct


def _compute_stats_file(fname_ctr_pred, fname_ctr_true, fname_seg_true, folder_out, fname_out):

    img_pred = Image(fname_ctr_pred)
    img_true = Image(fname_ctr_true)
    img_seg_true = Image(fname_seg_true)

    stats_file_dct = _compute_stats(img_pred, img_true, img_seg_true)

    create_folders_local([folder_out])

    print stats_file_dct

    pickle.dump(stats_file_dct, open(fname_out, "wb"))


def _compute_stats_folder(subj_name_lst, cc, folder_out, fname_out):

    stats_folder_dct = {}

    mse_lst, maxmove_lst, zcoverage_lst = [], [], []
    for subj in subj_name_lst:
        with open(folder_out + 'res_' + cc + '_' + subj + '.pkl') as outfile:    
            subj_metrics = pickle.load(outfile)
            outfile.close()
        if subj_metrics['mse'] is not None:
            mse_lst.append(subj_metrics['mse'])
            maxmove_lst.append(subj_metrics['maxmove'])
            zcoverage_lst.append(subj_metrics['zcoverage'])

    stats_folder_dct['avg_mse'] = round(np.mean(mse_lst),2)
    stats_folder_dct['avg_maxmove'] = round(np.mean(maxmove_lst),2)
    stats_folder_dct['cmpt_fail_subj_test'] = round(sum(elt >= 10.0 for elt in maxmove_lst)*100.0/len(maxmove_lst),2)
    stats_folder_dct['avg_zcoverage'] = round(np.mean(zcoverage_lst),2)

    print stats_folder_dct
    pickle.dump(stats_folder_dct, open(fname_out, "wb"))



def compute_dataset_stats(path_local, cc):

    path_local_nii = path_local + 'propseg_nii_' + cc + '/'
    path_local_res_pkl = path_local + 'propseg_pkl_' + cc + '/'
    path_local_gold = path_local + 'gold_' + cc + '/'
    path_local_seg = path_local + 'input_nii_' + cc + '/'
    fname_pkl_out = path_local_res_pkl + 'res_' + cc + '_'

    subj_name_lst = []
    for ff in os.listdir(path_local_nii):
        print ff
        if ff.endswith('_pred.nii.gz'):
            subj_name_cur = ff.split('_pred.nii.gz')[0]
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
        _compute_stats_folder(subj_name_lst, cc, path_local_res_pkl, fname_pkl_out_all)

def display_results(path_local, cc):

    path_local_res_pkl = path_local + 'propseg_pkl_' + cc + '/'

    for f in os.listdir(path_local_res_pkl):
        if f.endswith('_all.pkl') and f.startswith('res_'+cc):
            with open(path_local_res_pkl + f) as outfile:    
                metrics = pickle.load(outfile)
                outfile.close()
            print '\n' + f
            print metrics


# ******************************************************************************************


def plot_comparison_nb_train(path_local, cc):

    path_best_train_lst = [pp for pp in os.listdir(path_local) if pp.startswith('plot_best_train_' + cc + '_')]
    mm_lst = list(np.unique([pp.split('_')[-1] for pp in path_best_train_lst]))
    nb_train_lst = [pp.split('_')[-2] for pp in path_best_train_lst]
    path_best_train_lst = [path_local + pp + '/' + f for pp in path_best_train_lst for f in os.listdir(path_local + pp) if f.endswith('.pkl')]
    create_folders_local([path_local+'plot_nb_train_img_comparison/'])

    path_best_train_dct = {}
    for pp in path_best_train_lst:
        for mm in mm_lst:
            if mm in pp.split('/')[-2]:
                if not mm in path_best_train_dct:
                    path_best_train_dct[mm]={}
                for nn in nb_train_lst:
                    if nn in pp.split('/')[-2]:
                        path_best_train_dct[mm][nn] = pp


    res_best_train_dct = {}
    for mm in mm_lst:
        fname_test_lst = [pickle.load(open(path_best_train_dct[mm][f]))['fname_test'] for f in path_best_train_dct[mm]]
        fname_test_lst = list(set(fname_test_lst[0]).intersection(*fname_test_lst[1:]))


        res_best_train_dct[mm] = {}
        for f in path_best_train_dct[mm]:
            with open(path_best_train_dct[mm][f]) as outfile:    
                res = pickle.load(outfile)
                outfile.close()

            res_best_train_dct[mm][f] = []
            for test_smple in fname_test_lst:
                res_best_train_dct[mm][f].append(res['all'][res['fname_test'].index(test_smple)])

    nb_row_plot = len(path_best_train_dct)
    nb_col_plot = len(path_best_train_dct[list(path_best_train_dct.keys())[0]])

    nb_img_train_lst = [int(ll) for ll in list(path_best_train_dct[list(path_best_train_dct.keys())[0]].keys())]
    nb_img_train_lst.sort()
    nb_img_train_lst = [str(ll) for ll in nb_img_train_lst]

    sns.set(style="whitegrid", palette="pastel", color_codes=True)
    fig, axes = plt.subplots(nb_row_plot, nb_col_plot, sharey='col', figsize=(8*nb_col_plot, 8*nb_row_plot))
    cmpt = 1
    # color_lst = sns.color_palette("husl", n_colors=max([int(ii) for ii in nb_img_train_lst]))
    # random.shuffle(color_lst, lambda:0.25)
    color_lst = sns.light_palette("orange", n_colors=max([int(ii) for ii in nb_img_train_lst]))
    fig.subplots_adjust(left=0.05, bottom=0.05)
    for mm in mm_lst:
        for f in nb_img_train_lst:
            a = plt.subplot(nb_row_plot, nb_col_plot,cmpt)
            sns.violinplot(data=res_best_train_dct[mm][f], inner="quartile", cut=0, scale="count", sharey=True, color=color_lst[int(f)-1])
            sns.swarmplot(data=res_best_train_dct[mm][f], palette='deep', size=4)
            a.set_ylabel(mm, fontsize=13)
            a.set_xlabel('# of training image: ' + f, fontsize=13)

            stg = '# of testing subj: ' + str(len(res_best_train_dct[mm][f]))
            stg += '\nMean: ' + str(round(np.mean(res_best_train_dct[mm][f]),2))
            stg += '\nStd: ' + str(round(np.std(res_best_train_dct[mm][f]),2))

            if mm != 'zcoverage':
                stg += '\nMax: ' + str(round(np.max(res_best_train_dct[mm][f]),2))

                if cc == 't2':
                    y_lim_min, y_lim_max = 0.01, 30
                y_stg_loc = y_lim_max-10

            else:
                stg += '\nMin: ' + str(round(np.min(res_best_train_dct[mm][f]),2))

                if cc == 't2':
                    y_lim_min, y_lim_max = 60, 101
                elif cc == 't1':
                    y_lim_min, y_lim_max = 85, 101
                y_stg_loc = y_lim_min+20

            a.set_ylim([y_lim_min,y_lim_max])
            
            a.text(0.2, y_stg_loc, stg, fontsize=13)

            cmpt += 1
    plt.show()
    fig.tight_layout()
    fig.savefig(path_local+'plot_nb_train_img_comparison/plot_comparison_' + cc + '.png')
    plt.close()


def panda_seg(path_local, cc, nb_img, mm):

    fname_out_pd = path_local + 'train_test_' + cc + '.pkl'

    fname_pd_trainer = path_local + 'output_pkl_' + cc + '/' + str(nb_img) + '.pkl'
    with open(fname_pd_trainer) as outfile:    
        score_trainers = pickle.load(outfile)
        outfile.close()

    from sct_sdika import find_id_extr_df
    best_trainer_idx = find_id_extr_df(score_trainers, mm+'_moy')[1]

    path_input_train = path_local + 'input_train_' + cc + '_' + str(nb_img) + '/' + str(int(best_trainer_idx.split('0')[1])/50).zfill(3) + '/' + best_trainer_idx + '.txt'
    
    train_subj_lst = []
    text = open(path_input_train, 'r').read()
    for tt in text.split('\n'):
        name = tt.split('/')[-1]
        if len(name):
            train_subj_lst.append(name.split('_'+cc)[0])

    fname_test_valid_pd = path_local + 'test_valid_' + cc + '.pkl'
    with open(fname_test_valid_pd) as outfile:    
        test_valid_pd = pickle.load(outfile)
        outfile.close()
    train_test_pd = test_valid_pd[['patho', 'resol', 'subj_name']]
    subj_name_lst = train_test_pd.subj_name.values.tolist()
    train_test_pd['train_test'] = ['train' if f in train_subj_lst else 'test' for f in subj_name_lst]

    train_test_pd.to_pickle(fname_out_pd)
  
    print '# of patient in: '
    print '... training:' + str(len(train_test_pd[(train_test_pd.patho == 'patient') & (train_test_pd.train_test == 'train')]))
    print '... testing:' + str(len(train_test_pd[(train_test_pd.patho == 'patient') & (train_test_pd.train_test == 'test')]))
    print '# of no-iso in: '
    print '... training:' + str(len(train_test_pd[(train_test_pd.resol == 'not') & (train_test_pd.train_test == 'train')]))
    print '... testing:' + str(len(train_test_pd[(train_test_pd.resol == 'not') & (train_test_pd.train_test == 'test')]))

    return path_input_train, train_subj_lst, train_test_pd, test_valid_pd

def test_seg(path_local, cc, nb_img, mm, path_ferguson_cur):

    path_centerline_svm = path_local + 'output_nii_' + cc + '_0/0_' + str(nb_img).zfill(3) + '/'
    path_txt_best, train_subj_lst, train_test_pd, test_valid_pd = panda_seg(path_local, cc, nb_img, mm)

    path_centerline_input = path_local + 'input_propseg_' + cc + '/'
    path_centerline_input_nb_img = path_centerline_input + str(nb_img) + '/'
    create_folders_local([path_centerline_input, path_centerline_input_nb_img])

    all_subj_lst = test_valid_pd.subj_name.values.tolist()
    subj_done_lst = [f.split('_'+cc)[0] for f in os.listdir(path_centerline_svm) if f.endswith('.nii.gz')]
    subj_done_entire_lst = [f for f in os.listdir(path_centerline_svm) if f.endswith('.nii.gz')]

    subj2do_lst = []
    print train_subj_lst
    for subj in all_subj_lst:
        if subj in subj_done_lst and subj not in train_subj_lst:
            subj_cur = subj_done_entire_lst[subj_done_lst.index(subj)]
            path_src = path_centerline_svm + subj_cur
            path_dst = path_centerline_input_nb_img + subj_cur

            shutil.copyfile(path_src, path_dst)
        elif subj not in train_subj_lst:
            subj2do_lst.append(subj)

    # pickle_ferguson = {
    #                   'contrast': cc,
    #                   'nb_image_train': 5228,
    #                   'valid_subj': subj2do_lst,
    #                   'path_ferguson': path_ferguson_cur
    #                   }
    # path_pickle_ferguson = path_local + 'ferguson_config.pkl'
    # output_file = open(path_pickle_ferguson, 'wb')
    # pickle.dump(pickle_ferguson, output_file)
    # output_file.close()

    # os.system('scp -r ' + path_input_train + ' ferguson:' + path_ferguson_cur)
    # os.system('scp ' + path_pickle_ferguson + ' ferguson:' + path_ferguson_cur)


def run_propseg_seg(path_local, cc, nb_img, mm):

    path_centerline_input = path_local + 'output_nii_' + cc + '_666/0_' + str(nb_img).zfill(3) + '/'
    path_data = path_local + 'input_nii_' + cc + '/'

    
    path_seg_out = path_local + 'output_svm_propseg_' + cc + '/'
    path_seg_out_cur = path_seg_out + str(nb_img) + '/'
    create_folders_local([path_seg_out, path_seg_out_cur])
    # path_seg_out_cur=path_seg_out_cur[1:]

    path_seg_out_propseg = path_local + 'output_propseg_' + cc + '/'
    create_folders_local([path_seg_out_propseg])
    # path_seg_out_propseg=path_seg_out_propseg[1:]

    if cc == 't2s':
        cc = 't2'

    for ctr_file in os.listdir(path_centerline_input)[::-1]:
        if '.nii.gz' in ctr_file:
            subj_cur = path_data + ctr_file.split('_centerline_pred')[0] + '.nii.gz'
            ctr_cur = path_centerline_input + ctr_file
            print ctr_file

            seg_propseg_out = path_seg_out_propseg + ctr_file.split('_centerline_pred')[0] + '_seg.nii.gz'
            seg_svm_out = path_seg_out_cur + ctr_file.split('_centerline_pred')[0] + '_seg.nii.gz'

            if not os.path.isfile(seg_svm_out):
                print 'sct_propseg -i ' + subj_cur + ' -c ' + cc + ' -init-centerline ' + ctr_cur + ' -ofolder ' + path_seg_out_cur
                os.system('sct_propseg -i ' + subj_cur + ' -c ' + cc + ' -init-centerline ' + ctr_cur + ' -ofolder ' + path_seg_out_cur)
                if os.path.isfile(path_seg_out_cur+ctr_file.split('_centerline_pred')[0]+'_centerline.nii.gz'):
                    os.remove(path_seg_out_cur+ctr_file.split('_centerline_pred')[0]+'_centerline.nii.gz')
           
            if not os.path.isfile(seg_propseg_out):
                print 'sct_propseg -i ' + subj_cur + ' -c ' + cc + ' -ofolder ' + path_seg_out_propseg
                os.system('sct_propseg -i ' + subj_cur + ' -c ' + cc + ' -ofolder ' + path_seg_out_propseg)
                if os.path.isfile(path_seg_out_propseg+ctr_file.split('_centerline_pred')[0]+'_centerline.nii.gz'):
                    os.remove(path_seg_out_propseg+ctr_file.split('_centerline_pred')[0]+'_centerline.nii.gz')

    # fname_out_pd = path_seg_out + str(nb_img) + '.pkl'

    # with open(path_local + 'train_test_' + cc + '.pkl') as outfile:    
    #     train_test_pd = pickle.load(outfile)
    #     outfile.close()

    # res_pd = train_test_pd[train_test_pd.train_test=='test'][['patho', 'resol', 'subj_name']]
    # subj_name_lst = res_pd.subj_name.values.tolist()

    # res_pd['dice_svm'] = [0.0 for i in range(len(subj_name_lst))]
    # for file in os.listdir(path_seg_out_cur):
    #     if file.endswith('.nii.gz'):
    #         file_src = path_seg_out_cur+file
    #         file_dst = path_data + file.split('_centerline')[0] + '_seg.nii.gz'
    #         subj_id = file.split('_'+cc)[0]
    #         file_dice = path_seg_out_cur+subj_id+'.txt'
    #         os.system('sct_dice_coefficient -i ' + file_src + ' -d ' + file_dst + ' -o ' + file_dice)
    #         text = open(file_dice, 'r').read()
    #         res_pd[res_pd.subj_name==subj_id]['dice_svm'] = float(text.split('= ')[1].split('\n')[0])

    # res_pd['propseg'] = [0.0 for i in range(len(subj_name_lst))]
    # for file in os.listdir(path_seg_out_propseg):
    #     if file.endswith('_seg.nii.gz'):
    #         file_src = path_seg_out_propseg+file
    #         file_dst = path_data + file
    #         subj_id = file.split('_'+cc)[0]
    #         file_dice = path_seg_out_propseg+subj_id+'.txt'
    #         if os.path.isfile(file_src):
    #             os.system('sct_dice_coefficient -i ' + file_src + ' -d ' + file_dst + ' -o ' + file_dice)
    #             text = open(file_dice, 'r').read()
    #             res_pd[res_pd.subj_name==subj_id]['dice_propseg'] = float(text.split('= ')[1].split('\n')[0])

    # print res_pd


# ****************************      USER CASE      *****************************************

def readCommand(  ):
    "Processes the command used to run from the command line"
    parser = argparse.ArgumentParser('CNN-Sdika Pipeline')
    parser.add_argument('-ofolder', '--output_folder', help='Output Folder', required = False)
    parser.add_argument('-c', '--contrast', help='Contrast of Interest', required = False)
    parser.add_argument('-s', '--step', help='Prepare (step=0) or Push (step=1) or Pull (step 2) or Compute metrics (step=3) or Display results (step=4)', 
                                        required = False)
    parser.add_argument('-n', '--nb_train_img', help='Nb Training Images', required = False)

    arguments = parser.parse_args()
    return arguments


USAGE_STRING = """
  USAGE:      Ichhhhh
                 """

if __name__ == '__main__':

    # Read input
    parse_arg = readCommand()

    if not parse_arg.output_folder:
        print USAGE_STRING
    else:
        path_local_sdika = parse_arg.output_folder

    if not parse_arg.step:
        step = 0
    else:
        step = int(parse_arg.step)  
    if not parse_arg.nb_train_img:
        nb_train_img = 1
    else:
        nb_train_img = int(parse_arg.nb_train_img)

    # Format of parser arguments
    contrast_of_interest = str(parse_arg.contrast) 


    prepare_dataset_cnn(path_local_sdika, contrast_of_interest, '/Volumes/data_processing/bdeleener/machine_learning/filemanager_t2s_new/datasets.pbz2')


    if not step:
        prediction_propseg(path_local_sdika, contrast_of_interest)

    elif step == 1:
        compute_dataset_stats(path_local_sdika, contrast_of_interest)

    elif step == 4:
        display_results(path_local_sdika, contrast_of_interest)

    elif step == 5:
        plot_comparison_nb_train(path_local_sdika, contrast_of_interest)

    elif step == 6:
        test_seg(path_local_sdika, contrast_of_interest, nb_train_img, 'zcoverage', '/home/neuropoly/code/spine-ms-t2/')
    
    elif step == 7:
        run_propseg_seg(path_local_sdika, contrast_of_interest, nb_train_img, 'zcoverage')





