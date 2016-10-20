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

try:
    import cPickle as pickle
except:
    import pickle

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
  fscore, n_examples = map(list, zip(*clf_stats['roc_history']))
  plot_result(n_examples, fscore, "training examples (#)", 'roc_auc')
  ax = plt.gca()
  ax.set_ylim((0, 1))

  plt.show()

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

model_hyperparam = {'C': [1, 1000],
                    'kernel': 'linear',
                    'probability': True,
                    'class_weight': (None, 'balanced')}

fname_trial = '/Users/chgroc/data/spine_detection/results_0-001_0-5_recall/LinearSVM_trials.pkl'

plot_param_stats(fname_trial, model_hyperparam)
