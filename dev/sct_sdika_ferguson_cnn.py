import os
import time
import pickle

####################################################################################################################
#   User Case

path_ferguson_work = '/home/neuropoly/code/spine-cnn/'
path_config = path_ferguson_work + 'ferguson_cnn_config.pkl'
with open(path_config) as outfile:    
	config = pickle.load(outfile)
	outfile.close()
contrast = config['contrast']
llambda = config['lambda']
path_ferguson_input_img = path_ferguson_work + 'cnn_img_' + contrast + '/'
path_ferguson_res_img = path_ferguson_work + 'cnn_output_img_' + contrast + '_' + llambda + '/'
cmd_line_test = './spine_detect_cnn -ctype=dpdt -lambda=' + llambda + ' NONE '

if not os.path.exists(path_ferguson_res_img):
	os.makedirs(path_ferguson_res_img)

subj_id = []
for f in os.listdir(path_ferguson_input_img):
	if f.endswith('.img'):
		subj_id = f.split('.')[0]
		os.system(cmd_line_test + path_ferguson_input_img + subj_id + ' ' + path_ferguson_res_img + subj_id)
		if os.path.isfile(path_ferguson_res_img + subj_id + '_ctr.txt'):
			os.remove(path_ferguson_res_img + subj_id + '_ctr.txt')