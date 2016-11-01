import bz2
import sct_utils as sct
import pickle
import os

####################################################################################################################
#   User Case

# path_data_dict = '/Volumes/data_processing/bdeleener/machine_learning/filemanager_t2s_new/datasets.pbz2'
# path_dataset = '/Volumes/data_processing/bdeleener/machine_learning/data_t2s/'
path_data_dict = '/Volumes/data_processing/bdeleener/machine_learning/filemanager_large_nobrain_nopad/datasets.pbz2'
path_dataset = '/Volumes/data_processing/bdeleener/machine_learning/large_nobrain_nopad/'
nb_images_process = -1              # Set it to -1 for processing all testing images
path_folder_output = '/Users/chgroc/data/spine_detection/results3D/'
contrast = 't2'
path_model = '/Users/chgroc/data/spine_detection/results2D/model_t2_linear_001/LinearSVM_train'
# threshold = 0.866772426866          # Set a value (float) or Trial path
threshold = '/Users/chgroc/data/spine_detection/results2D/results_t2_linear_001/LinearSVM_trials.pkl'
int_eval = 1                        # To make or not the error computation
int_remove_tmp = 0                  # To remove or not temporary files
int_verbose = 0                     # 1 for step1, 2 for step2 ...etc.

####################################################################################################################
#       Run Script

# Load Dict
with bz2.BZ2File(path_data_dict, 'rb') as f:
    datasets_dict = pickle.load(f)
    f.close()

# Testing fname images
dataset_path = sct.slash_at_the_end(path_dataset, slash=1)
fname_testing_raw_images = datasets_dict['testing']['raw_images']
fname_testing_gold_images = datasets_dict['testing']['gold_images']
path_fname_images = [dataset_path + f[0] for f in fname_testing_raw_images]
print '\nDataset path: ' + dataset_path

# Nb images to process
if nb_images_process < 0 or nb_images_process > len(path_fname_images):
    nb_images_process = len(path_fname_images)
print '# of images to process: ' + str(nb_images_process) + '\n'

# Find threshold value from training
if isinstance(threshold, str):
    with open(threshold) as outfile:    
        trial = pickle.load(outfile)
        outfile.close()
    loss_list = [trial[i]['result']['loss'] for i in range(len(trial))]
    thrsh_list = [trial[i]['result']['thrsh'] for i in range(len(trial))]
    idx_best_params = loss_list.index(min(loss_list))
    threshold = trial[idx_best_params]['result']['thrsh']

#Patch and grid size
grid_search_dict_t2={'patch_size':32,
                    'initial_resolution': [2, 2, 10],
                    'initial_resize': [0.1, 0.25],
                    'initial_list_offset': [[xv, yv, zv] for xv in range(-1,1) for yv in range(-1,1) for zv in range(-5,5) if [xv, yv, zv] != [0, 0, 0]],
                    'offset': [1,1,4],
                    'max_iter': 3
                    }
grid_search_dict_t2s={'patch_size':32,
                    'initial_resolution': [8, 8, 1],
                    'initial_list_offset': [[xv, yv, zv] for xv in range(-3,3) for yv in range(-3,3) for zv in range(-3,3) if [xv, yv, zv] != [0, 0, 0]],
                    'offset': [1,1,4],
                    'max_iter': 3}

pickle.dump(grid_search_dict_t2, open(path_folder_output + 'grid_search_dict_t2.pkl', "wb"))

# from random import shuffle
# shuffle(path_fname_images)
# path_fname_images = [dataset_path + 'e23185_t2.nii.gz']

for f_in in path_fname_images[:nb_images_process]:

    cmd = 'python ../scripts/sct_detect_spinalcord.py -i ' + f_in
    cmd += ' -o ' + path_folder_output + (f_in.split('/')[-1]).split('.nii.gz')[0] + '_centerline_pred.nii.gz'
    cmd += ' -c ' + contrast
    cmd += ' -param ' + path_folder_output + 'grid_search_dict_t2.pkl'
    cmd += ' -imodel ' + path_model
    cmd += ' -eval ' + str(int_eval)
    cmd += ' -r ' + str(int_remove_tmp)
    cmd += ' -threshold ' + str(threshold)
    cmd += ' -v ' + str(int_verbose)
    
    os.system(cmd)