import os
import time
import pickle
import time

def run_test(cmd, pref, path_train, valid_lst):

	path_ferguson_res_img = path_ferguson_work + 'output_img_' + contrast + pref + '/'
	if not os.path.exists(path_ferguson_res_img):
		os.makedirs(path_ferguson_res_img)

	path_res_cur = path_ferguson_res_img +tt.split('.')[0]+ '/'
	if not os.path.exists(path_res_cur):
		os.makedirs(path_res_cur)

	for ss_test in valid_lst:
		if not ss_test in id_train_subj:
			start = time.time()
			os.system(cmd + '-lambda=' + ll + ' ' + path_train + tt.split('.')[0] + ' ' + path_ferguson_input_img + ss_test + ' ' + path_res_cur + ss_test)
			end = time.time()
			with open(path_res_cur + ss_test + '_time.txt', 'w') as text_file:
				text_file.write(str(end-start))
				text_file.close()
			if os.path.isfile(path_res_cur + ss_test + '_ctr.txt'):
				os.remove(path_res_cur + ss_test + '_ctr.txt')
			if os.path.isfile(path_res_cur + ss_test + '_svm.hdr'):
				os.remove(path_res_cur + ss_test + '_svm.hdr')
			if os.path.isfile(path_res_cur + ss_test + '_svm.img'):
				os.remove(path_res_cur + ss_test + '_svm.img')


####################################################################################################################
#   User Case


path_ferguson_work = '/home/neuropoly/code/spine-ms-tmi-dwi/'
path_config = path_ferguson_work + 'ferguson_config.pkl'
with open(path_config) as outfile:    
	config = pickle.load(outfile)
	outfile.close()
contrast = config['contrast']
nb_image_train = config['nb_image_train']
rot = config['rot']
ll = config['lambda']
valid_subj = config['valid_subj']
dyn = config['dyn']
tr = config['best_trainer']
path_ferguson_input_img = path_ferguson_work + 'input_img/' + contrast + '/'
path_ferguson_train = path_ferguson_work + contrast + '_'+ str(nb_image_train) + '/'

cmd_line_test = './spine_detect -ctype=dpdt '
if dyn:
	cmd_line_test_dyn = './spine_detect -ctype=dp1 '

cmd_line_train = './spine_train_svm -hogsg -incr=20 --addRot=' + rot

path_sub_train = path_ferguson_train

txt_name = [f for f in os.listdir(path_sub_train) if f.endswith('.txt') and not '_ctr' in f and not '_valid' in f]
for zz,tt in enumerate(txt_name):
	if tt == tr+'.txt':
		path_txt = path_sub_train + tt
		path_txt_ctr = path_sub_train + tt.split('.')[0] + '_ctr.txt'

		os.system(cmd_line_train + ' ' + tt.split('.')[0] + ' ' + path_txt + ' ' + path_txt_ctr + ' --list True')
		os.system('mv ' + tt.split('.')[0] + '.yml ' + path_sub_train + tt.split('.')[0] + '.yml')

		id_train_subj = [line.rstrip('\n').split('/')[-1] for line in open(path_txt)]

		run_test(cmd_line_test, '', path_sub_train, valid_subj)
		if dyn:
			run_test(cmd_line_test_dyn, '_dyn', path_sub_train, valid_subj)
		if os.path.isfile(path_sub_train + tt.split('.')[0] + '.yml'):
			os.remove(path_sub_train + tt.split('.')[0] + '.yml')