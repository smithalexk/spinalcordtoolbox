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

        output_file = open(path_local + 'dataset_lst_' + cc + '.pkl', 'wb')
        pickle.dump(fname_img_lst, output_file)
        output_file.close()

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

    with open(path_local + 'dataset_dict_' + cc + '.pkl') as outfile:    
        fname_subj_lst = pickle.load(outfile)
        outfile.close()

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
    output_file = open(path_pickle_ferguson, 'wb')
    pickle.dump(pickle_ferguson, output_file)
    output_file.close()

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


####################################################################################################################
#   User Case


# *********************** PATH & CONFIG ***********************
fname_local_script = '/Users/chgroc/spinalcordtoolbox/dev/sct_detect_spinalcord_sdika_ferguson.py'
path_ferguson = '/home/neuropoly/code/spine-ms/'
path_sct_testing_large = '/Volumes/data_shared/sct_testing/large/'
path_local_sdika = '/Users/chgroc/data/data_sdika/'
create_folders_local([path_local_sdika])
# contrast_lst = ['t2', 't1', 't2s']
contrast_lst = ['t2']

# *********************** PREPARE DATASET ***********************
# prepare_dataset(path_local_sdika, contrast_lst, path_sct_testing_large)

# *********************** SEND SCRIPT TO FERGUSON ***********************
# os.system('scp ' + fname_local_script + ' ferguson:' + path_ferguson)

# *********************** SEND DATA TO FERGUSON ***********************
# send_data2ferguson(path_local_sdika, path_ferguson, 't2', 1)
# send_data2ferguson(path_local_sdika, path_ferguson, 't1', 1)

# *********************** PULL RESULTS FROM FERGUSON ***********************
pull_img_convert_nii_remove_img(path_local_sdika, path_ferguson, 't2', 1)
# pull_img_convert_nii_remove_img(path_local_sdika, path_ferguson, 't1', 1)
    



#     # compute_nii_stats(path_local_res_nii, path_local_gold, path_local_input_nii, path_local_res_pkl + 'res_')
    
#     fname_pkl = 'res_' + contrast + '_' + seg_ctr + '_' + str(nb_image_train) + '.pkl'
#     # fail_list_t2_1_seg = ['twh_e24751', 'twh_e23699', 'bwh_CIS1_t2_sag', 'bwh_SP4_t2_sag', 'bwh_SP4_t2_sag_stir',
#     #                         'paris_hc_42', 'paris_hc_84', 'paris_hc_82', 'paris_hc_116',
#     #                         'amu_PAM50_VP', 'amu_PAM50_GB', 'amu_PAM50_ALT', '20151025_emil_t2',
#     #                         'unf_pain_08']
#     # fail_list_t2_1_ctr = ['twh_e24751', 'twh_e23699', 'bwh_SP4_t2_sag', 'bwh_SP4_t2_sag_stir',
#     #                         'paris_hc_42', 'paris_hc_82', 'paris_hc_116',
#     #                         'amu_PAM50_VP', 'amu_PAM50_GB', 'amu_PAM50_ALT',
#     #                         'unf_pain_08', 'unf_pain_07', 'unf_pain_20']
#     # display_stats(path_local_res_pkl, fname_pkl, [])
#     plot_violin(path_local_res_pkl, fname_pkl)




# elif part == 2:

#     def pull_img_convert_nii_remove_img(path_ferguson, path_local_img, path_local_nii):

#         # Pull .img results from ferguson
#         os.system('scp -r ferguson:' + path_ferguson + ' ' + '/'.join(path_local_img.split('/')[:-2]) + '/')

#         # Convert .img to .nii
#         # Remove .img files
#         for f in os.listdir(path_local_img):
#             if not f.startswith('.'):
#                 path_res_cur = path_local_nii + f + '/'
#                 if not os.path.exists(path_res_cur):
#                     os.makedirs(path_res_cur)

#                 training_subj = f.split('__')

#                 if os.path.isdir(path_local_img+f):
#                     for ff in os.listdir(path_local_img+f):
#                         if ff.endswith('_ctr.hdr'):

#                             path_cur = path_local_img + f + '/' + ff
#                             path_cur_out = path_res_cur + ff.split('_ctr')[0] + '_centerline_pred.nii.gz'
#                             img = nib.load(path_cur)
#                             nib.save(img, path_cur_out)

#                         elif ff == 'time.txt':
#                             os.rename(path_local_img + f + '/time.txt', path_local_nii + f + '/time.txt')

#                     os.system('rm -r ' + path_local_img + f)

#     def compute_stats_file(fname_centerline, fname_gold_standard, fname_gold_standard_seg):

#         im_pred = Image(fname_centerline)
#         im_true = Image(fname_gold_standard)
#         print fname_centerline
#         print fname_gold_standard
#         im_true_seg = Image(fname_gold_standard_seg)

#         nx, ny, nz, nt, px, py, pz, pt = im_true.dim

#         count_slice, slice_coverage = 0, 0
#         mse_dist = []
#         for z in range(im_true.dim[2]):
#             slice_pred = im_pred.data[:,:,z]
#             slice_true = im_true.data[:,:,z]
#             slice_true_seg = im_true_seg.data[:,:,z]
#             if np.sum(im_true.data[:,:,z]):
#                 x_true, y_true = [np.where(im_true.data[:,:,z] > 0)[i][0] for i in range(len(np.where(im_true.data[:,:,z] > 0)))]
#                 x_pred, y_pred = [np.where(im_pred.data[:,:,z] > 0)[i][0] for i in range(len(np.where(im_pred.data[:,:,z] > 0)))]

#                 xx_seg, yy_seg = np.where(im_true_seg.data[:,:,z]==1.0)
#                 xx_yy = [[x,y] for x, y in zip(xx_seg,yy_seg)]
#                 if [x_pred, y_pred] in xx_yy:
#                     slice_coverage += 1

#                 x_true, y_true = im_true.transfo_pix2phys([[x_true, y_true, z]])[0][0], im_true.transfo_pix2phys([[x_true, y_true, z]])[0][1]
#                 x_pred, y_pred = im_pred.transfo_pix2phys([[x_pred, y_pred, z]])[0][0], im_pred.transfo_pix2phys([[x_pred, y_pred, z]])[0][1]

#                 dist = ((x_true-x_pred))**2 + ((y_true-y_pred))**2
#                 mse_dist.append(dist)

#                 count_slice += 1

#         return sqrt(sum(mse_dist)/float(count_slice)), sqrt(max(mse_dist)), float(slice_coverage*100.0)/count_slice

#     def compute_nii_stats(path_local_nii, path_local_gold, path_local_seg, fname_pkl_out):

#         for f in os.listdir(path_local_nii):
#             if not f.startswith('.'):
#                 print path_local_nii + f + '/'
#                 path_res_cur = path_local_nii + f + '/'

#                 training_subj = f.split('__')

#                 if not os.path.isfile(fname_pkl_out + f + '.pkl'):

#                     time_info = 0.0
#                     mse_cur, max_move_cur, slice_coverage_cur = [], [], []
#                     for ff in os.listdir(path_local_nii+f):
#                         if ff.endswith('_centerline_pred.nii.gz'):
#                             path_cur_pred = path_res_cur + ff
#                             path_cur_gold = path_local_gold + ff.split('_centerline_pred.nii.gz')[0] + '_centerline_gold.nii.gz'
#                             path_cur_gold_seg = path_local_seg + ff.split('_centerline_pred.nii.gz')[0] + '_seg.nii.gz'
#                             print path_cur_pred
#                             mse_cur.append(compute_stats_file(path_cur_pred, path_cur_gold, path_cur_gold_seg)[0])
#                             max_move_cur.append(compute_stats_file(path_cur_pred, path_cur_gold, path_cur_gold_seg)[1])
#                             slice_coverage_cur.append(compute_stats_file(path_cur_pred, path_cur_gold, path_cur_gold_seg)[2])

#                         elif ff == 'time.txt':
#                             with open(path_local_nii + f + '/' + ff) as text_file:
#                                 time_info = round(float(text_file.read()),2)
            
#                     save_dict = {}
#                     save_dict['iteration'] = training_subj
#                     if len(mse_cur):
#                         save_dict['avg_mse'] = round(np.mean(mse_cur),2)
#                         save_dict['avg_max_move'] = round(np.mean(max_move_cur),2)
#                         save_dict['cmpt_fail_subj_test'] = round(sum(elt >= 10.0 for elt in max_move_cur)*100.0/len(mse_cur),2)
#                         save_dict['slice_coverage'] = round(np.mean(slice_coverage_cur),2)
#                         save_dict['boostrap_time'] = time_info
#                     else:
#                         save_dict['avg_mse'], save_dict['avg_max_move'], save_dict['cmpt_fail_subj_test'], save_dict['slice_coverage'], save_dict['boostrap_time'] = None, None, None, None, None
                    
#                     print save_dict
#                     pickle.dump(save_dict, open(fname_pkl_out + f + '.pkl', "wb"))

#     def display_stats(path_local_pkl, fname_out, fail_list=[]):

#         mse, move, fail, cov, time, it = [], [], [], [], [], []
#         for f in os.listdir(path_local_pkl):
#             if not f == fname_out and not f.startswith('.'):
#                 print path_local_pkl + f
#                 with open(path_local_pkl + f) as outfile:    
#                     res = pickle.load(outfile)
#                     outfile.close()

#                 if not any(x in f for x in fail_list):
#                     mse.append(res['avg_mse'])
#                     move.append(res['avg_max_move'])
#                     fail.append(res['cmpt_fail_subj_test'])
#                     cov.append(res['slice_coverage'])
#                     time.append(res['boostrap_time'])
#                     it.append(res['iteration'][0])

#         if nb_image_train > 1:
#             it = [i for i in range(len(mse))]
#         stats_dict = {'iteration': it, 'avg_mse': mse, 'avg_max_move': move, 'cmpt_fail_subj_test': fail, 'slice_coverage': cov, 'boostrap_time': time}
#         pickle.dump(stats_dict, open(path_local_pkl + fname_out, "wb"))

#         if nb_image_train > 1:
#             it = [str(i).zfill(3) for i in it]

#         mean = ['Mean', str(round(np.mean(mse),2)), str(round(np.mean(move),2)), str(round(np.mean(fail),2)), str(round(np.mean(cov),2)), str(round(np.mean(time),2))]
#         std = ['Std', str(round(np.std(mse),2)), str(round(np.std(move),2)), str(round(np.std(fail),2)), str(round(np.std(cov),2)), str(round(np.std(time),2))]
#         maxx = ['Extremum', str(round(np.max(mse),2)), str(round(np.max(move),2)), str(round(np.max(fail),2)), str(round(np.min(cov),2)), str(round(np.max(time),2))]
        
#         mse = [str(i) for i in mse]
#         move = [str(i) for i in move]
#         fail = [str(i) for i in fail]
#         cov = [str(i) for i in cov]
#         time = [str(i) for i in time]

#         head2print = ['It. #', 'Avg. MSE [mm]', 'Avg. Max Move [mm]', 'Cmpt. Fail [%]', 'zCoverage [%]', 'Time [s]']
#         scape = [' ']
#         if nb_image_train > 1:
#             col_width = max(len(word) for word in head2print) + 2
#         else:
#             col_width = max(len(word) for word in it) + 2
#         data2plot = [[ff, mm, mmmm, ss, cc, tt] for ff, mm, mmmm, ss, cc, tt in zip(it, mse, move, fail, cov, time)]
#         data2plot = [head2print] + [scape] + [mean] + [std] + [maxx] + [scape] + data2plot

#         for row in data2plot:
#             print "".join(str(word).ljust(col_width) for word in row)


#     def plot_violin(path, fname_pkl):

#         data = pickle.load(open(path + fname_pkl))
#         import seaborn as sns
#         import matplotlib.pyplot as plt
#         sns.set(style="whitegrid", palette="pastel", color_codes=True)

#         metric_list = ['avg_mse', 'avg_max_move', 'cmpt_fail_subj_test', 'slice_coverage', 'boostrap_time']
#         metric_name_list = ['Avg. MSE [mm]', 'Avg. Max Move [mm]', 'Cmpt. Fail [%]', 'zCoverage [%]', 'Time [s]']
#         for m, n in zip(metric_list, metric_name_list):
#             fig = plt.figure(figsize=(10, 10))
#             a = plt.subplot(111)
#             sns.violinplot(data[m], inner="point", orient="v")
#             stg = 'Mean: ' + str(round(np.mean(data[m]),2))
#             stg += '\nStd: ' + str(round(np.std(data[m]),2))
#             if m != 'slice_coverage':
#                 stg += '\nMax: ' + str(round(np.max(data[m]),2))
#             else:
#                 stg += '\nMin: ' + str(round(np.min(data[m]),2))
#             a.text(0.3,np.max(data[m]),stg,fontsize=15)
#             plt.xlabel(n)
#             plt.savefig(path + 'plot_' + m + '.png')
#             plt.close()

