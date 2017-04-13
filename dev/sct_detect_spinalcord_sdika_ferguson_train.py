import os
import time
import pickle
import time


####################################################################################################################
#   User Case


path_ferguson_work = '/home/neuropoly/code/spine-ms-tmi-dwi/'
path_config = path_ferguson_work + 'ferguson_config.pkl'
with open(path_config) as outfile:    
	config = pickle.load(outfile)
	outfile.close()
nb_image_train = 1
rot = '0:360:0'

cc_dct = {'t2': '002', 'dmri': '074', 't1': '069', 't2s': '023'}

cmd_line_train = './spine_train_svm -hogsg -incr=20 --addRot=0:360:0'

for cc in ['t1']:
	path_txt = path_ferguson_work + cc + '_1/' + cc_dct[cc] + '.txt'
	path_txt_ctr = path_ferguson_work + cc + '_1/' + cc_dct[cc] + '_ctr.txt'

	os.system(cmd_line_train + ' ' + cc_dct[cc] + ' ' + path_txt + ' ' + path_txt_ctr + ' --list True')