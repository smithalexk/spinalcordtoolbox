#!/usr/bin/env python
#########################################################################################
#
# This module contains some functions for plotting results of supervised machine learning algorithms
# python msct_machine_learning_results.py
#
# ---------------------------------------------------------------------------------------
# Copyright (c) 2016 Polytechnique Montreal <www.neuro.polymtl.ca>
# Authors: Benjamin De Leener, charley
# Modified: 2016-07-08
#
# About the license: see the file LICENSE.TXT
#########################################################################################

import sys
import matplotlib.pyplot as plt
from matplotlib import rcParams
import numpy as np
import os

try:
    import cPickle as pickle
except:
    import pickle

from msct_image import Image


rcParams['legend.fontsize'] = 10


"""
    TODO:
          - Plot Objective_fct = fct(nb_training samples)
          - Plot Score en fonction du clf model et du contraste
          - Plot Training and Prediction time en fonction du clf model et du contraste
"""

def plot_training_keras_results(fname_in):

    clf_stats = pickle.load(open(fname_in, "rb"))

    def plot_result(x, y, x_legend, text):
        """Plot result as a function of x."""
        x = np.array(x)
        y = np.array(y)
        plt.title('Classification ' + text + ' as a function of %s' % x_legend)
        plt.xlabel('%s' % x_legend)
        plt.ylabel(text)
        plt.grid(True)
        plt.plot(x, y)

    # Plot evolution
    plt.figure()
    plt.subplot(2, 2, 1)
    accuracy, n_examples = map(list, zip(*clf_stats['accuracy_history']))
    plot_result(n_examples, accuracy, "training examples (#)", 'accuracy')
    ax = plt.gca()
    ax.set_ylim((0, 1))

    plt.subplot(2, 2, 2)
    precision, n_examples = map(list, zip(*clf_stats['precision_history']))
    plot_result(n_examples, precision, "training examples (#)", 'precision')
    ax = plt.gca()
    ax.set_ylim((0, 1))

    plt.subplot(2, 2, 3)
    recall, n_examples = map(list, zip(*clf_stats['recall_history']))
    plot_result(n_examples, recall, "training examples (#)", 'recall')
    ax = plt.gca()
    ax.set_ylim((0, 1))

    plt.subplot(2, 2, 4)
    roc_auc, n_examples = map(list, zip(*clf_stats['roc_history']))
    plot_result(n_examples, roc_auc, "training examples (#)", 'roc_auc')
    ax = plt.gca()
    ax.set_ylim((0, 1))

    plt.show()

    # find best result
    best_roc = np.max(roc_auc)
    index_best_roc = np.argmax(roc_auc)
    training_nb = n_examples[index_best_roc]
    best_accuracy = accuracy[index_best_roc]
    best_precision = precision[index_best_roc]
    best_recall = recall[index_best_roc]

    print 'Number of training samples = ', training_nb
    print 'Accuracy = ', best_accuracy
    print 'Precision = ', best_precision
    print 'Recall = ', best_recall
    print 'ROC AUC = ', best_roc


def plot_param_stats(fname_trial, param_dict):

  with open(fname_trial) as outfile:
    trial = pickle.load(outfile)
    outfile.close()

  for param in trial[0]['misc']['vals']:

    param_vals = [t['misc']['vals'][param] for t in trial]
    loss_vals = [-t['result']['loss'] for t in trial]
    trial_id = [t['tid'] for t in trial]
    eval_time = [t['result']['eval_time'] for t in trial]
    trsh_vals = [t['result']['thrsh'] for t in trial]

    fig = plt.figure(1, figsize=(15, 15))

    ax1 = plt.subplot(2, 2, 2)
    ax1.scatter(param_vals, loss_vals, s=20, linewidth=0.01, alpha=0.75)
    ax1.set_xlabel(param, fontsize=16)
    ax1.set_ylabel('Objective metric', fontsize=16)
    if isinstance(param_vals[0][0], int):
      plt.xticks(range(len(param_dict[param])), list(param_dict[param]))
    ax1.grid()

    ax2 = plt.subplot(2, 2, 1)
    ax2.set_xlim(trial_id[0]-1, trial_id[-1]+1)
    ax2.scatter(trial_id, param_vals, s=20, linewidth=0.01, alpha=0.75)
    ax2.set_xlabel('Trial ID', fontsize=16)
    ax2.set_ylabel(param, fontsize=16)
    if isinstance(param_vals[0][0], int):
      plt.yticks(range(len(param_dict[param])), list(param_dict[param]))
    ax2.grid()

    ax3 = plt.subplot(2, 2, 3)
    ax3.scatter(param_vals, eval_time, s=20, linewidth=0.01, alpha=0.75)
    ax3.set_xlabel(param, fontsize=16)
    ax3.set_ylabel('Evaluation time (s)', fontsize=16)
    if isinstance(param_vals[0][0], int):
      plt.xticks(range(len(param_dict[param])), list(param_dict[param]))
    ax3.grid()

    ax4 = plt.subplot(2, 2, 4)
    ax4.scatter(param_vals, trsh_vals, s=20, linewidth=0.01, alpha=0.75)
    ax4.set_xlabel(param, fontsize=16)
    ax4.set_ylabel('Threshold', fontsize=16)
    if isinstance(param_vals[0][0], int):
      plt.xticks(range(len(param_dict[param])), list(param_dict[param]))
    ax4.grid()

    plt.show()

def progress(stats):
    """Report progress information, return a string."""
    s = ''

    if 'n_train' in stats and 'n_test' in stats:
        s += 'Training dataset: ' + str(stats['n_train'] + stats['n_test']) + ' samples (' + str(stats['n_train_pos'] + stats['n_test_pos']) + ' positive)\n'
        s += '... ' + str(round(float(stats['n_train']*100)/(stats['n_train'] + stats['n_test']), 3)) + '% for training HyperOpt\n'
        s += '... ' + str(round(float(stats['n_test']*100)/(stats['n_train'] + stats['n_test']), 3)) + '% for testing HyperOpt\n'

    if 'n_train' in stats:
        s += str(stats['n_train']) + " train samples (" + str(stats['n_train_pos']) + " positive)\n"
        if stats['total_fit_time'] != 0:
            s += 'Training time: ' + str(stats['total_fit_time']) + ' s (' + str(round(float(stats['n_train'])/stats['total_fit_time'],3)) + ' samples/sec)\n'

    if 'n_test' in stats:
        s += str(stats['n_test']) + " test samples (" + str(stats['n_test_pos']) + " positive)\n"
        s += "accuracy: " + str(stats['accuracy']) + "\n"
        s += "precision: " + str(stats['precision']) + "\n"
        s += "recall: " + str(stats['recall']) + "\n"
        s += "roc: " + str(stats['roc']) + "\n"
        if stats['total_predict_time'] != 0:
            s += 'Prediction time: ' + str(stats['total_predict_time']) + ' s (' + str(round(float(stats['n_test'])/stats['total_predict_time'],3)) + ' samples/sec)\n'

    return s


def printProgressReport(fname_test, fname_trial, path_output):

    print '\n#########################################################'
    stg_trial, stg_test = '', ''

    if os.path.isfile(fname_trial):

        stg_trial += '\nFname Trial file: ' + fname_trial
        with open(fname_trial) as outfile:
            trial = pickle.load(outfile)
            outfile.close()

        loss_list = [trial[i]['result']['loss'] for i in range(len(trial))]
        eval_time_list = [trial[i]['result']['eval_time'] for i in range(len(trial))]
        idx_best_params = loss_list.index(min(loss_list))
        best_params = trial[idx_best_params]['misc']['vals']
        threshold = trial[idx_best_params]['result']['thrsh']

        stg_trial += '\nTriral ID: ' + str(idx_best_params)
        stg_trial += '\nparams: '
        stg_trial += str(best_params)
        stg_trial += '\nThreshold = ' + str(threshold)
        stg_trial += '\n'

        path_data, filename = os.path.split(fname_trial)
        model_name, rest = filename.split('trials.pkl')

        fname_pkl_prefix = model_name + 'eval_' + str(idx_best_params).zfill(6)
        fname_pkl = [path_data + '/' + filename for filename in os.listdir(path_data) if filename.startswith(fname_pkl_prefix)][0]

        with open(fname_pkl) as outfile:
            train_report = pickle.load(outfile)
            outfile.close()

        stg_trial += progress(train_report)

    else:
        stg_trial += '\nNo Trial file available!'  

    if os.path.isfile(fname_test):

        stg_test += '\nFname Test file: ' + fname_test + '\n'

        with open(fname_test) as outfile:
            test_report = pickle.load(outfile)
            outfile.close()

        stg_test += progress(test_report)
    else:
        stg_test += '\nNo Test file available!'  
     
    print stg_trial
    print stg_test

    with open(path_output + 'report_trial', 'w') as f:
        f.write(stg_trial)
    with open(path_output + 'report_test', 'w') as f:
        f.write(stg_test)


def plotIntensityStandardization(path_data):

    list_filename = [path_data + f for f in os.listdir(path_data) if f.endswith('.nii.gz')]
    list_filename_cor = [path_data + f for f in os.listdir(path_data) if f.endswith('_cor.nii.gz')]
    list_filename_seg = [path_data + f for f in os.listdir(path_data) if f.endswith('_seg.nii.gz')]
    list_filename_init = [path_data + f.split('_seg.nii.gz')[0] + '.nii.gz' for f in os.listdir(path_data) if f.endswith('_seg.nii.gz')]
    # list_filename_init = list(set(list_filename)-set(list_filename_cor+list_filename_seg))

    # Overview of data
    f = plt.figure(figsize=(15,15))
    b = 50   # n bins
    ax1 = f.add_subplot(2,2,1)
    y_list_before = []
    bincenters_list_before = []
    for i, fname in enumerate(list_filename_init):
        im = Image(fname)
        im_seg = Image(list_filename_seg[i])

        data = im.data
        data_seg = im_seg.data
        data = (data[np.where(data_seg==1)]).flatten()

        y,binEdges=np.histogram(data,bins=b)
        bincenters = 0.5*(binEdges[1:]+binEdges[:-1])
        y = y * 1.0 / np.max(y)
        plt.plot(bincenters,y,'-')
        y_list_before.append(y)
        bincenters_list_before.append(bincenters)

    plt.title('Before Intensity Standardization', y=1.08)
    plt.axis([0,1,0,1])
    plt.tick_params(axis='x', which='both', bottom='off', top='off', labelbottom='off')
    plt.ylabel('Probability')
    plt.xlabel('Normalized Intensity Histogram for all subjects')

    f.add_subplot(2,2,2)
    y_list_after = []
    bincenters_list_after = []
    for i, fname in enumerate(list_filename_cor):
        im = Image(fname)
        im_seg = Image(list_filename_seg[i])

        data = im.data
        data_seg = im_seg.data
        data = (data[np.where(data_seg==1)]).flatten()

        # data = im.data.flatten()
        y,binEdges=np.histogram(data,bins=b)
        bincenters = 0.5*(binEdges[1:]+binEdges[:-1])
        y = y * 1.0 / np.max(y)
        plt.plot(bincenters,y,'-')
        y_list_after.append(y)
        bincenters_list_after.append(bincenters)

    plt.title('After Intensity Standardization', y=1.08)
    plt.axis([0,1,0,1])
    plt.tick_params(axis='x', which='both', bottom='off', top='off', labelbottom='off')
    plt.ylabel('Probability')
    plt.xlabel('Normalized Intensity Histograms for all subjects')

    ax2 = f.add_subplot(2,2,3)
    y_before_mean = sum(y_list_before) / len(y_list_before)
    y_before_std = np.std(y_list_before,axis=0)
    plt.plot(bincenters_list_before[0],y_before_mean,'-', linewidth=3.0)
    plt.plot(bincenters_list_before[0], y_before_std+y_before_mean,'b--')
    plt.plot(bincenters_list_before[0], y_before_mean-y_before_std,'b--')

    # plt.title('Before Intensity Standardization')
    plt.legend(['avg', 'std'])
    plt.axis([0,1,0,1])
    ax2.plot([1.07, 1.07], [0, 2.2], 'k--', transform=ax2.transAxes, clip_on=False, linewidth=3)
    plt.tick_params(axis='x', which='both', bottom='off', top='off', labelbottom='off')
    plt.ylabel('Probability')
    plt.xlabel('Normalized Intensity Histograms averaged across all subjects')

    f.add_subplot(2,2,4)
    y_after_mean = sum(y_list_after) / len(y_list_after)
    y_after_std = np.std(y_list_after,axis=0)
    plt.plot(bincenters_list_after[0], y_after_mean,'-', linewidth=3.0)
    plt.plot(bincenters_list_after[0], y_after_std+y_after_mean,'b--')
    plt.plot(bincenters_list_after[0], y_after_mean-y_after_std,'b--')

    # plt.title('After Intensity Standardization')
    plt.legend(['avg', 'std'])
    plt.axis([0,1.5,0,1])
    plt.tick_params(axis='x', which='both', bottom='off', top='off', labelbottom='off')
    plt.ylabel('Probability')
    plt.xlabel('Normalized Intensity Histogram averaged across all subjects')
    

    plt.suptitle('Normalized Intensity Histograms of cropped images', fontsize=16)
    plt.show()
    # # f.savefig(prefixe + 'overview_intensity_standardization.png')
    # # plt.close()


############################################################################################################################
#                                           Benjamin Part
############################################################################################################################
fname_trial = '/Users/benjamindeleener/data/machine_learning/results_pipeline_cnn/CNN_eval_5120256_000000000000.pkl'
# plot_training_keras_results(fname_trial)

#fname_trial = '/Users/benjamindeleener/data/machine_learning/result_cnn_t2s/CNN_eval_2569472_000000000000.pkl'
fname_trial = '/Users/benjamindeleener/data/machine_learning/results_pipeline_cnn/large/CNN_eval_28160256_000000000000.pkl'

#plot_param_stats(fname_trial, model_hyperparam)
# plot_training_keras_results(fname_trial)
fname_trial = '/Users/benjamindeleener/data/machine_learning/results_pipeline_cnn/large/CNN_eval_20480256_000000000000.pkl'


############################################################################################################################
#                                           charley Part
############################################################################################################################

path_data='/Users/chgroc/data/spine_detection/results2D/'

for fold in os.listdir(path_data):

    if fold.startswith('results'):
        result_folder = path_data+fold + '/'
        if os.path.isfile(result_folder + 'SVM_trials.pkl'):
            fname_trial = result_folder + 'SVM_trials.pkl'
            fname_test = result_folder + 'SVM_test.pkl'
        else:
            fname_trial = result_folder + 'LinearSVM_trials.pkl'
            fname_test = result_folder + 'LinearSVM_test.pkl'

        printProgressReport(fname_test=fname_test, fname_trial=fname_trial, path_output=result_folder)



# path_irs = '/Users/chgroc/data/spine_detection/irs/'
# plotIntensityStandardization(path_irs)



