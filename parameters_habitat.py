import os
import numpy as np 

base_folder = '/home/ggeorgak' #'/home/yimengl/Datasets'

class ParametersMapNet_Habitat(object):
	def __init__(self):
		self.habitat_root = '{}/habitat-api'.format(base_folder)

		self.train_config = "configs/tasks/my_objectnav_mp3d_train.yaml"
		self.test_config = "configs/tasks/my_objectnav_mp3d_test.yaml"

		self.action_list = ['rotate_ccw', 'rotate_cw', 'forward']

		self.orig_res = (64, 64) #(480, 640)
		self.crop_size = (64, 64)
		self.batch_size = 2
		self.seq_len = 5
		#self.with_shortest_path = False

		#=================================================================================================
		self.src_root = base_folder #'{}/mapnet_habitat_navigation'.format(base_folder) #'{}/habitat-api/trained_model'.format(base_folder)
		self.model_id = '37'
		self.model_dir = '{}/{}/{}/'.format(self.src_root, 'output', self.model_id)
		if not os.path.exists(self.model_dir):
			os.makedirs(self.model_dir)

		self.custom_resnet = False

		self.observation_dim = (21, 21) #61, 21
		self.global_map_dim = (29, 29) #151, 29
		self.cell_size = 0.3 #0.1, 0.3
		self.img_embedding = 32
		self.depth_embedding = 16
		self.sseg_embedding = 16
		self.dets_embedding = 16
		self.sseg_labels = 40
		self.dets_nClasses = 40

		self.with_img = True
		self.with_depth = False
		self.with_sseg = True
		self.with_dets = False
		self.use_raw_sseg = False
		self.use_raw_dets = False
		self.update_type = 'lstm'

		if self.use_raw_sseg:
			self.sseg_embedding = self.sseg_labels
		if self.use_raw_dets:
			self.dets_embedding = self.dets_nClasses
		self.grid_channels = self.with_img * self.img_embedding + self.with_depth*self.depth_embedding + self.with_sseg * self.sseg_embedding + self.with_dets * self.dets_embedding
		self.input_flags = (self.with_img, self.with_sseg, self.with_dets, self.use_raw_sseg, self.use_raw_dets, self.with_depth)
		self.map_embedding = self.grid_channels

		self.orientations = 12
		self.pad = int((self.observation_dim[1] - 1) / 2.0)

		#training params
		self.loss_type = 'NLL'
		self.nEpochs = 100
		self.lr_rate = 5e-4
		self.step_size = 10
		self.gamma = 0.5

		self.save_interval = 100
		self.show_interval = 5
		self.plot_interval = 100
		self.test_interval = 500

		#EvaLuation params
		self.test_batch_size = 1


class ParametersIL_Habitat(object):
	def __init__(self):
		self.habitat_root = '{}/habitat-api'.format(base_folder)

		self.train_config = "configs/tasks/my_objectnav_mp3d_train.yaml"
		self.test_config = "configs/tasks/my_objectnav_mp3d_test.yaml"

		self.action_list = ['rotate_ccw', 'rotate_cw', 'forward']

		self.orig_res = (64,64) #(480, 640)
		self.crop_size = (64, 64)
		self.crop_size_obsv = (224, 224)
		self.batch_size = 2
		self.seq_len = 5

		self.sseg_labels = 40
		self.dets_nClasses = 40

		self.mapNet_src_root =  base_folder #'{}/mapnet_habitat_navigation'.format(base_folder) #'{}/habitat-api/trained_model/'.format(base_folder)
		self.mapNet_model_id = '37'
		self.mapNet_model_dir = self.mapNet_src_root + '/output/' + self.mapNet_model_id + '/'
		self.mapNet_iters = 100 #3500
		self.use_p_gt = False
		self.finetune_mapNet = True

		self.model_id = 'IL_38'
		self.model_dir = self.mapNet_src_root + '/output/' + self.model_id + '/'
		if not os.path.exists(self.model_dir):
			os.makedirs(self.model_dir)
		self.test_iters = 20 #500

		self.use_ego_obsv = True
		self.conv_embedding = 8
		self.fc_dim = 128

		self.nTargets = 21

		#self.max_shortest_path = 20
		self.max_steps = 500 #10
		self.stop_on_collision = False
		self.dist_from_goal = 1.0

		#training params
		self.nEpochs = 100
		self.lr_rate = 1e-3
		self.loss_weight = 10
		self.step_size = 2
		self.gamma = 0.5

		# how to select the minibatch params
		self.EPS_START = 0.9
		self.EPS_END = 0.1
		self.EPS_DECAY = 1000

		self.save_interval = 50 #500
		self.show_interval = 5
		self.plot_interval = 100
		self.test_interval = 500