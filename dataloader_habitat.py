from torch.utils.data import Dataset
import math
import numpy as np
import random
import matplotlib.pyplot as plt
from parameters_habitat import ParametersMapNet_Habitat, ParametersIL_Habitat
from PIL import Image 
import data_helper_habitat as dhh
import torch
import gzip
import json

import habitat
from habitat.config.default import get_config
from habitat.core.embodied_task import Episode
from habitat.core.logging import logger
from habitat.datasets import make_dataset
from habitat.datasets.object_nav.object_nav_dataset import ObjectNavDatasetV1
from habitat.tasks.eqa.eqa import AnswerAction
from habitat.tasks.nav.nav import MoveForwardAction
from habitat.utils.test_utils import sample_non_stop_action

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR

from mapNet import MapNet
from IL_Net import ILNet, Encoder



# Tailored to work with mapNet training
class Habitat_MP3D(Dataset):

	def __init__(self, par, seq_len, config_file):
		self.config_file = '{}/{}'.format(par.habitat_root, config_file)

		self.seq_len = seq_len
		self.dets_nClasses = par.dets_nClasses

		config = get_config(self.config_file)
		config.defrost()
		config.SIMULATOR.DEPTH_SENSOR.NORMALIZE_DEPTH = False
		config.freeze()
		self.hfov = float(config.SIMULATOR.DEPTH_SENSOR.HFOV) * np.pi / 180.
		#self.intr = np.zeros((4), dtype=np.float32)
		#self.intr[0] = 1./np.tan(self.hfov/2.)*(config.SIMULATOR.DEPTH_SENSOR.WIDTH/2.) # fx
		#self.intr[1] = 1./np.tan(self.hfov/2.)*(config.SIMULATOR.DEPTH_SENSOR.HEIGHT/2.) # fy
		#self.intr[2] = config.SIMULATOR.DEPTH_SENSOR.WIDTH/2. # cx
		#self.intr[3] = config.SIMULATOR.DEPTH_SENSOR.HEIGHT/2. # cy
		dataset = make_dataset(id_dataset=config.DATASET.TYPE, config=config.DATASET)
		self.env = habitat.Env(config=config, dataset=dataset)

		self.orig_res = (config.SIMULATOR.DEPTH_SENSOR.HEIGHT, config.SIMULATOR.DEPTH_SENSOR.WIDTH)
		self.cropSize = par.crop_size
		self.normalize = True
		self.pixFormat = 'NCHW'


	def get_episode(self):
		imgs = np.zeros((self.seq_len, 3, self.cropSize[1], self.cropSize[0]), dtype=np.float32)
		ssegs = np.zeros((self.seq_len, 1, self.cropSize[1], self.cropSize[0]), dtype=np.float32)
		#dets = np.zeros((self.seq_len, self.dets_nClasses, self.cropSize[1], self.cropSize[0]), dtype=np.float32)
		depths = np.zeros((self.seq_len, 1, self.cropSize[1], self.cropSize[0]), dtype=np.float32)
		points2D, local3D, abs_poses, rel_poses = [], [], [], []

		observations = self.env.reset() # samples an episode
		#print(self.env._current_episode)
		len_shortest_path = len(self.env._current_episode.shortest_paths[0])
		#print('Path length:', len_shortest_path)

		for i in range(self.seq_len):
			# visual and 3d info
			imgData = dhh.preprocess_img(observations['rgb'], self.cropSize, self.pixFormat, self.normalize)
			depthData = dhh.preprocess_depth(observations['depth'], self.cropSize)
			points2D_step, local3D_step = dhh.depth_to_3D(observations, self.hfov, self.orig_res, self.cropSize)
			#ssegData, pred = dhh.get_sseg(img=observations['rgb'])
			ssegData = np.expand_dims(observations['semantic'], 0).astype(float)
			
			#dhh.display_sample2( observations['rgb'], np.squeeze(observations['depth']), observations['semantic'] )
			#dhh.display_sample(  observations['rgb'], 
			#						np.squeeze(observations['depth']), 
			#						dhh.colorize_sseg(ssegData), 
			#						dhh.convert_semantic_output(pred) )#,
									#savepath="grid_examples/"+str(i)+"_obs.png" )
			
			# simulator state sensors
			#gps = np.copy(observations['gps']) # starts from [0,0]
			#heading = np.copy(observations['heading'])
			#state = self.env._sim.get_agent_state()
			#sensor_state = self.env._sim.get_agent_state().sensor_states['depth']
			#print(state.position, sensor_state.position)
			#print("GPS:", gps)
			#print(heading, observations['compass'])
			#agent_pose = np.concatenate((gps, heading))
			agent_pose = dhh.get_sim_location(self.env)
			#print("SIM-LOCATION:", agent_pose)


			imgs[i,:,:,:] = imgData
			ssegs[i,:,:,:] = ssegData
			#dets[i, :, :, :] = detData
			depths[i,:,:,:] = depthData
			abs_poses.append(agent_pose)
			points2D.append(points2D_step)
			local3D.append(local3D_step)

			# get the relative pose with respect to the first pose in the sequence
			rel = dhh.get_rel_pose(pos2=abs_poses[i], pos1=abs_poses[0])
			#print(agent_pose, rel)
			rel_poses.append(rel)

			# explicitly clear observation otherwise they will be kept in memory the whole time
			observations = None

			if i < len_shortest_path-1:
				action = self.env._current_episode.shortest_paths[0][i].action
			else: # when the episode length is smaller than seq_len then select a random next action
				while True:
				    action = self.env.action_space.sample()
				    if action['action'] != 'STOP':
				        break
			#print('Dataloader action:', action)
			#print(action['action'])
			observations = self.env.step(action)


		item = {}
		item['images'] = torch.from_numpy(imgs).float()
		item['points2D'] = points2D
		item['local3D'] = local3D
		item['pose'] = rel_poses
		item['abs_pose'] = abs_poses
		item['sseg'] = torch.from_numpy(ssegs).float()
		item['depths'] = torch.from_numpy(depths).float()
		#item['dets'] = torch.from_numpy(dets).float() # dummy detections
		return item



	'''
	# Return an episode
	def __getitem__(self, index):
		imgs = np.zeros((self.seq_len, 3, self.cropSize[1], self.cropSize[0]), dtype=np.float32)
		ssegs = np.zeros((self.seq_len, 1, self.cropSize[1], self.cropSize[0]), dtype=np.float32)
		dets = np.zeros((self.seq_len, self.dets_nClasses, self.cropSize[1], self.cropSize[0]), dtype=np.float32)
		points2D, local3D, poses_epi = [], [], []

		observations = self.env.reset()
		# compare len_shortest_path with seq_len
		while True:
			len_shortest_path = len(self.env._current_episode.shortest_paths[0])
			if len_shortest_path >= self.seq_len:
				break
			else:
				print('shortest path is shorter than seq_len. Reset environment ...')
				observations = self.env.reset()

		imgData, ssegData, detData, points2D_step, local3D_step = dhh.getImageData(observations, 
			self.dets_nClasses, self.cropSize, self.orig_res, self.pixFormat, self.normalize)

		imgs[0, :, :, :] = imgData
		points2D.append(points2D_step)
		local3D.append(local3D_step)

		ssegs[0, :, :, :] = ssegData
		dets[0, :, :, :] = detData

		gps = np.copy(observations['gps'])
		heading = np.copy(observations['heading'])
		poses_epi.append(np.concatenate((gps, heading)))

		# explicitly clear observation otherwise they will be kept in memory the whole time
		observations = None

		for i in range(1, self.seq_len):
			
			#while True:
			#	action = self.env.action_space.sample()
			#	if action['action'] != 'STOP':
			#		break
			
			action = self.env._current_episode.shortest_paths[0][i-1].action

			observations = self.env.step(action)
			imgData, ssegData, detData, points2D_step, local3D_step = dhh.getImageData(observations, 
				self.dets_nClasses, self.cropSize, self.orig_res, self.pixFormat, self.normalize)

			imgs[i, :, :, :] = imgData
			points2D.append(points2D_step)
			local3D.append(local3D_step)

			ssegs[i, :, :, :] = ssegData
			dets[i, :, :, :] = detData

			gps = np.copy(observations['gps'])
			heading = np.copy(observations['heading'])
			poses_epi.append(np.concatenate((gps, heading)))

			observations = None

		# Need to get the relative poses (towards the first frame) for the ground-truth
		poses_epi = np.asarray(poses_epi)
		rel_poses = dhh.relative_poses(poses=poses_epi)

		item = {}
		item['images'] = torch.from_numpy(imgs).float()
		item['points2D'] = points2D
		item['local3D'] = local3D
		item['pose'] = rel_poses
		item['abs_pose'] = poses_epi
		item['scale'] = 1.0
		item['sseg'] = torch.from_numpy(ssegs).float()
		item['dets'] = torch.from_numpy(dets).float()
		return item
	'''

#===========================================================================================================

from habitat.tasks.nav.shortest_path_follower import ShortestPathFollower

class myDistanceToGoal():
	def __init__(self, sim, config):
		self._previous_position = None
		self._sim = sim
		self._config = config
		self._episode_view_points = None

		self.goal_radius = self._config.TASK.SUCCESS.SUCCESS_DISTANCE
		self._follower = ShortestPathFollower(self._sim, self.goal_radius, False)

	# called when a new episode start
	def reset_metric(self, episode):
		self._previous_position = None
		self._metric = None
		self._episode_view_points = [
			view_point.agent_state.position 
			for goal in episode.goals
			for view_point in goal.view_points 
			]
		self.update_metric(episode)

	def update_metric(self, episode):
		current_position = self._sim.get_agent_state().position
		distance_to_target = self.my_geodesic_distance(current_position, self._episode_view_points, episode)
		self._previous_position = current_position
		self._metric = distance_to_target

	# adapted from geodesic_distance() in habitat_simulator.py
	def my_geodesic_distance(self, position_a, position_b, episode):
		path = episode._shortest_path_cache
		path.requested_start = np.array(position_a, dtype=np.float32)

		self._sim._sim.pathfinder.find_path(path)
		return path.geodesic_distance

	# a target object has multiple viewpoints. find the closest one.
	def get_nearest_goal_viewpoint(self, episode):
		current_position = self._sim.get_agent_state().position
		path = episode._shortest_path_cache
		path.requested_start = np.array(current_position, dtype=np.float32)

		self._sim._sim.pathfinder.find_path(path)
		return path.points[-1]	

	# run the shortest_path_follower to generate the next action to go to the nearest viewpoint
	def get_next_action(self, episode):
		goal_viewpoint = self.get_nearest_goal_viewpoint(episode)
		best_action = self._follower.get_next_action(goal_viewpoint)
		return best_action

#===========================================================================================================
# used for IL test
class Habitat_MP3D_online(Dataset):

	def __init__(self, par, seq_len, config_file, action_list):
		self.config_file = '{}/{}'.format(par.habitat_root, config_file)
		
		self.seq_len = seq_len
		self.actions = action_list
		self.dets_nClasses = par.dets_nClasses

		config = get_config(self.config_file)
		config.defrost()
		config.SIMULATOR.DEPTH_SENSOR.NORMALIZE_DEPTH = False
		config.TASK.MEASUREMENTS.append('COLLISIONS')
		config.freeze()
		self.hfov = float(config.SIMULATOR.DEPTH_SENSOR.HFOV) * np.pi / 180.
		dataset = make_dataset(id_dataset=config.DATASET.TYPE, config=config.DATASET)
		self.env = habitat.Env(config=config, dataset=dataset)

		self.cropSize = par.crop_size
		self.cropSizeObsv = par.crop_size_obsv
		self.orig_res = (config.SIMULATOR.DEPTH_SENSOR.HEIGHT, config.SIMULATOR.DEPTH_SENSOR.WIDTH)
		self.normalize = True
		self.pixFormat = 'NCHW'

		self.dg = myDistanceToGoal(self.env.sim, config)

#=============================================================================================================
class Habitat_MP3D_IL(Dataset):

	def __init__(self, par, seq_len, config_file, action_list):
		self.config_file = '{}/{}'.format(par.habitat_root, config_file)

		self.seq_len = seq_len
		self.actions = action_list
		self.dets_nClasses = par.dets_nClasses

		config = get_config(self.config_file)
		config.defrost()
		config.SIMULATOR.DEPTH_SENSOR.NORMALIZE_DEPTH = False
		config.TASK.MEASUREMENTS.append('COLLISIONS')
		config.freeze()
		self.hfov = float(config.SIMULATOR.DEPTH_SENSOR.HFOV) * np.pi / 180.
		dataset = make_dataset(id_dataset=config.DATASET.TYPE, config=config.DATASET)
		self.env = habitat.Env(config=config, dataset=dataset)

		self.cropSize = par.crop_size
		self.cropSizeObsv = par.crop_size_obsv
		self.orig_res = (config.SIMULATOR.DEPTH_SENSOR.HEIGHT, config.SIMULATOR.DEPTH_SENSOR.WIDTH)
		self.normalize = True
		self.pixFormat = 'NCHW'

		self.dg = myDistanceToGoal(self.env.sim, config)


	# Return an episode
	#def __getitem__(self, index):
	def get_episode(self):
		imgs = np.zeros((self.seq_len, 3, self.cropSize[1], self.cropSize[0]), dtype=np.float32)
		imgs_obsv = np.zeros((self.seq_len, 3, self.cropSizeObsv[1], self.cropSizeObsv[0]), dtype=np.float32)
		ssegs = np.zeros((self.seq_len, 1, self.cropSize[1], self.cropSize[0]), dtype=np.float32)
		depths = np.zeros((self.seq_len, 1, self.cropSize[1], self.cropSize[0]), dtype=np.float32)
		depths_obsv = np.zeros((self.seq_len, 1, self.cropSizeObsv[1], self.cropSizeObsv[0]), dtype=np.float32)
		#dets = np.zeros((self.seq_len, self.dets_nClasses, self.cropSize[1], self.cropSize[0]), dtype=np.float32)
		#dets_obsv = np.zeros((self.seq_len, 1, self.cropSizeObsv[1], self.cropSizeObsv[0]), dtype=np.float32)
		points2D, local3D, abs_poses, rel_poses, action_seq, cost_seq, collision_seq = [], [], [], [], [], [], []

		# Choose whether the episode's observations are going to be decided by the
		# teacher (best action) or randomly
		choice = np.random.randint(2, size=1)[0] # if 1 then do teacher

		observations = self.env.reset()
		self.dg.reset_metric(self.env._current_episode)
		# compare len_shortest_path with seq_len
		while True:
			len_shortest_path = len(self.env._current_episode.shortest_paths[0])
			if len_shortest_path > self.seq_len and observations['objectgoal'][0] in [0, 5, 6, 8, 10, 13]:
				break
			else:
				#print('shortest path', len_shortest_path,'is shorter than seq_len. Reset environment ...')
				observations = self.env.reset()

		target_lbl = observations['objectgoal'][0]

		for i in range(self.seq_len):
			imgData = dhh.preprocess_img(observations['rgb'], self.cropSize, self.pixFormat, self.normalize)
			img_obsv = dhh.preprocess_img(observations['rgb'], self.cropSizeObsv, self.pixFormat, self.normalize)
			depthData = dhh.preprocess_depth(observations['depth'], self.cropSize)
			depth_obsv = dhh.preprocess_depth(observations['depth'], self.cropSizeObsv)
			points2D_step, local3D_step = dhh.depth_to_3D(observations, self.hfov, self.orig_res, self.cropSize)
			ssegData = np.expand_dims(observations['semantic'], 0).astype(float)

			imgs[i, :, :, :] = imgData
			depths[i, :, :, :] = depthData
			points2D.append(points2D_step)
			local3D.append(local3D_step)
			ssegs[i, :, :, :] = ssegData
			imgs_obsv[i, :, :, :] = img_obsv
			depths_obsv[i, :, :, :] = depth_obsv

			agent_pose = dhh.get_sim_location(self.env)
			abs_poses.append(agent_pose)
			# get the relative pose with respect to the first pose in the sequence
			rel = dhh.get_rel_pose(pos2=abs_poses[i], pos1=abs_poses[0])
			#print(agent_pose, rel)
			rel_poses.append(rel)

			# use shortest path to find the best_action
			best_action = self.dg.get_next_action(self.env._current_episode)
			cost_seq.append(dhh.configAction_to_costOfParamAction(best_action))

			# either select the best action or ...
			if best_action == 0: # meaning the best action is stop
				action = np.random.randint(1, 4)
			elif choice:
				action = best_action
			# ... randomly choose the next action between [1, 2, 3] to move in the episode
			else:
				action = np.random.randint(1, 4)

			param_action = dhh.configAction_to_paramAction(action)
			action_seq.append(param_action)

			observations = self.env.step(action)

			if i==0:
				collision_seq.append(0)
			else:
				metrics = self.env.get_metrics()
				collision_seq.append( metrics['collisions']['is_collision'] )

		observations = None

		collision_seq = np.asarray(collision_seq, dtype=np.float32)
		action_seq = np.asarray(action_seq)
		cost_seq = np.asarray(cost_seq, dtype=np.float32)

		item = {}
		item['images'] = torch.from_numpy(imgs).float()
		item['points2D'] = points2D
		item['local3D'] = local3D
		item['actions'] = action_seq
		item['costs'] = torch.from_numpy(cost_seq).float()
		item['target_lbl'] = target_lbl
		item['pose'] = rel_poses
		item['abs_pose'] = abs_poses
		item['collisions'] = torch.from_numpy(collision_seq).float()
		item['scale'] = 1.0
		item['sseg'] = torch.from_numpy(ssegs).float()
		#item['dets'] = torch.from_numpy(dets).float() # dummy detections
		item['depths'] = torch.from_numpy(depths).float()
		item['images_obsv'] = torch.from_numpy(imgs_obsv).float()
		#item['dets_obsv'] = torch.from_numpy(dets_obsv).float() # dummy detections
		item['depths_obsv'] = torch.from_numpy(depths_obsv).float()
		return item


	def get_item_policy(self, parIL, parMapNet, policy_net, mapNet, ego_encoder):
		with torch.no_grad():
			imgs = np.zeros((self.seq_len, 3, self.cropSize[1], self.cropSize[0]), dtype=np.float32)
			imgs_obsv = np.zeros((self.seq_len, 3, self.cropSizeObsv[1], self.cropSizeObsv[0]), dtype=np.float32)
			ssegs = np.zeros((self.seq_len, 1, self.cropSize[1], self.cropSize[0]), dtype=np.float32)
			depths = np.zeros((self.seq_len, 1, self.cropSize[1], self.cropSize[0]), dtype=np.float32)
			depths_obsv = np.zeros((self.seq_len, 1, self.cropSizeObsv[1], self.cropSizeObsv[0]), dtype=np.float32)		
			#dets = np.zeros((self.seq_len, self.dets_nClasses, self.cropSize[1], self.cropSize[0]), dtype=np.float32)
			#dets_obsv = np.zeros((self.seq_len, 1, self.cropSizeObsv[1], self.cropSizeObsv[0]), dtype=np.float32)
			points2D, local3D, abs_poses, rel_poses, action_seq, cost_seq, collision_seq = [], [], [], [], [], [], []

			observations = self.env.reset()
			self.dg.reset_metric(self.env._current_episode)
			# compare len_shortest_path with seq_len
			while True:
				len_shortest_path = len(self.env._current_episode.shortest_paths[0])
				if len_shortest_path > self.seq_len and observations['objectgoal'][0] in [0, 5, 6, 8, 10, 13]:
					break
				else:
					#print('shortest path is shorter than seq_len. Reset environment ...')
					observations = self.env.reset()

			#imgData, ssegData, detData, points2D_step, local3D_step = dhh.getImageData(observations, 
			#	self.dets_nClasses, self.cropSize, self.orig_res, self.pixFormat, self.normalize)
			#img_obsv, det_obsv = dhh.getImageData(observations, 1, self.cropSizeObsv, self.orig_res, self.pixFormat, 
			#	self.normalize, get3d=False)
			imgData = dhh.preprocess_img(observations['rgb'], self.cropSize, self.pixFormat, self.normalize)
			img_obsv = dhh.preprocess_img(observations['rgb'], self.cropSizeObsv, self.pixFormat, self.normalize)
			points2D_step, local3D_step = dhh.depth_to_3D(observations, self.hfov, self.orig_res, self.cropSize)
			ssegData = np.expand_dims(observations['semantic'], 0).astype(float)
			depthData = dhh.preprocess_depth(observations['depth'], self.cropSize)
			depth_obsv = dhh.preprocess_depth(observations['depth'], self.cropSizeObsv)	
			#detData = np.zeros((self.dets_nClasses, self.cropSize[1], self.cropSize[0]), dtype=np.float32) # dummy detections
			#det_obsv = np.zeros((1, self.cropSizeObsv[1], self.cropSizeObsv[0]), dtype=np.float32) # dummy detection observation

			imgs[0, :, :, :] = imgData
			depths[0, :, :, :] = depthData
			points2D.append(points2D_step)
			local3D.append(local3D_step)
			ssegs[0, :, :, :] = ssegData
			#dets[0, :, :, :] = detData
			imgs_obsv[0, :, :, :] = img_obsv
			#dets_obsv[0, :, :, :] = det_obsv
			depths_obsv[0, :, :, :] = depth_obsv

			#gps = np.copy(observations['gps'])
			#heading = np.copy(observations['heading'])
			#current_pose = np.concatenate((gps, heading))
			#poses_epi.append(current_pose)
			agent_pose = dhh.get_sim_location(self.env)
			abs_poses.append(agent_pose)
			# get the relative pose with respect to the first pose in the sequence
			rel = dhh.get_rel_pose(pos2=abs_poses[0], pos1=abs_poses[0])
			#print(agent_pose, rel)
			rel_poses.append(rel)			

			target_lbl = observations['objectgoal'][0]
			collision_seq.append(0)
			# use shortest path to find the best_action
			best_action = self.dg.get_next_action(self.env._current_episode)
			cost_seq.append(dhh.configAction_to_costOfParamAction(best_action))

			# explicitly clear observation otherwise they will be kept in memory the whole time
			observations = None

#===============================================================================================================
			# predict next action
			imgs_batch = torch.tensor(np.expand_dims(imgData, axis=0)).float()
			pose_gt_batch = np.expand_dims(rel, axis=0)
			sseg_batch = torch.tensor(np.expand_dims(ssegData, axis=0)).float()
			#dets_batch = torch.tensor(np.expand_dims(detData, axis=0)).float()
			depths_batch = torch.tensor(np.expand_dims(depthData, axis=0)).float()
			points2D_batch, local3D_batch = [], [] # add another dimension for the batch
			points2D_batch.append(points2D_step)
			local3D_batch.append(local3D_step)

			mapNet_input_start = (imgs_batch.cuda(), points2D_batch, local3D_batch, sseg_batch.cuda(), depths_batch.cuda())
			p_, map_ = mapNet.forward_single_step(local_info=mapNet_input_start, t=0, 
				input_flags=parMapNet.input_flags, update_type=parMapNet.update_type)

			tvec = torch.zeros(1, parIL.nTargets).float().cuda()
			tvec[0, target_lbl] = 1
			collision_ = torch.tensor([0], dtype=torch.float32).cuda() # collision indicator is 0 at initial position

			if parIL.use_ego_obsv:
				tensor_img_obsv = torch.tensor(img_obsv).float().cuda()
				tensor_depth_obsv = torch.tensor(depth_obsv).float().cuda()
				#print('tensor_img_obsv.shape = {}'.format(tensor_img_obsv.shape))
				#print('tensor_det_obsv.shape = {}'.format(tensor_det_obsv.shape))
				enc_in = torch.cat((tensor_img_obsv, tensor_depth_obsv), 0).unsqueeze(0)
				#print('enc_in.shape = {}'.format(enc_in.shape))
				ego_obsv_feat = ego_encoder(enc_in) # 1 x 512 x 1 x 1
				state = (map_, p_, tvec, collision_, ego_obsv_feat)
			else:
				state = (map_, p_, tvec, collision_)

			policy_net.hidden = policy_net.init_hidden(batch_size=1, state_items=len(state)-1)

#================================================================================================================
			for i in range(1, self.seq_len):
				pred_costs = policy_net(state, parIL.use_ego_obsv) # apply policy for single step
				pred_costs = pred_costs.view(-1).cpu().numpy()
				# use the deterministic version here
				pred_label = np.argmin(pred_costs)
				pred_action = self.actions[pred_label]
				#print('pred_action = {}'.format(pred_action))
				# pred_action is a param action
				action = dhh.paramAction_to_configAction(pred_action)

				observations = self.env.step(action)
				#imgData, ssegData, detData, points2D_step, local3D_step = dhh.getImageData(observations, 
				#	self.dets_nClasses, self.cropSize, self.orig_res, self.pixFormat, self.normalize)
				#img_obsv, det_obsv = dhh.getImageData(observations, 1, self.cropSizeObsv, self.orig_res, self.pixFormat, 
				#	self.normalize, get3d=False)
				imgData = dhh.preprocess_img(observations['rgb'], self.cropSize, self.pixFormat, self.normalize)
				img_obsv = dhh.preprocess_img(observations['rgb'], self.cropSizeObsv, self.pixFormat, self.normalize)
				points2D_step, local3D_step = dhh.depth_to_3D(observations, self.hfov, self.orig_res, self.cropSize)
				ssegData = np.expand_dims(observations['semantic'], 0).astype(float)
				depthData = dhh.preprocess_depth(observations['depth'], self.cropSize)
				depth_obsv = dhh.preprocess_depth(observations['depth'], self.cropSizeObsv)	
				#detData = np.zeros((self.dets_nClasses, self.cropSize[1], self.cropSize[0]), dtype=np.float32) # dummy detections
				#det_obsv = np.zeros((1, self.cropSizeObsv[1], self.cropSizeObsv[0]), dtype=np.float32)				

				imgs[i, :, :, :] = imgData
				depths[i, :, :, :] = depthData
				points2D.append(points2D_step)
				local3D.append(local3D_step)
				ssegs[i, :, :, :] = ssegData
				#dets[i, :, :, :] = detData
				imgs_obsv[i, :, :, :] = img_obsv
				#dets_obsv[i, :, :, :] = det_obsv
				depths_obsv[i, :, :, :] = depth_obsv

				#gps = np.copy(observations['gps'])
				#heading = np.copy(observations['heading'])
				#current_pose = np.concatenate((gps, heading))
				#poses_epi.append(current_pose)
				agent_pose = dhh.get_sim_location(self.env)
				abs_poses.append(agent_pose)
				# get the relative pose with respect to the first pose in the sequence
				rel = dhh.get_rel_pose(pos2=abs_poses[i], pos1=abs_poses[0])
				#print(agent_pose, rel)
				rel_poses.append(rel)	

				metrics = self.env.get_metrics()
				collision_seq.append( metrics['collisions']['is_collision'] )

				param_action = dhh.configAction_to_paramAction(action)
				action_seq.append(param_action)
				# use shortest path to find the best_action
				best_action = self.dg.get_next_action(self.env._current_episode)
				cost_seq.append(dhh.configAction_to_costOfParamAction(best_action))

				observations = None

#===============================================================================================================
				# get next state from mapNet
				imgs_batch = torch.tensor(np.expand_dims(imgData, axis=0)).float()
				pose_gt_batch = np.expand_dims(rel, axis=0)
				sseg_batch = torch.tensor(np.expand_dims(ssegData, axis=0)).float()
				#dets_batch = torch.tensor(np.expand_dims(detData, axis=0)).float()
				depths_batch = torch.tensor(np.expand_dims(depthData, axis=0)).float()
				points2D_batch, local3D_batch = [], [] # add another dimension for the batch
				points2D_batch.append(points2D_step)
				local3D_batch.append(local3D_step)
				batch_next = (imgs_batch.cuda(), points2D_batch, local3D_batch, sseg_batch.cuda(), depths_batch.cuda())

				if parIL.use_p_gt:
					#next_im_rel_pose = dhh.relative_poses(poses=pose_gt_batch)
					#p_gt = dh.build_p_gt(parMapNet, pose_gt_batch=np.expand_dims(next_im_rel_pose, axis=1)).squeeze(1)
					p_gt = dhh.build_p_gt(parMapNet, pose_gt_batch=np.expand_dims(pose_gt_batch, axis=1)).squeeze(1)
					p_next, map_next = mapNet.forward_single_step(local_info=batch_next, t=i, input_flags=parMapNet.input_flags,
						map_previous=state[0], p_given=p_gt, update_type=parMapNet.update_type)
				else:
					p_next, map_next = mapNet.forward_single_step(local_info=batch_next, t=i, 
						input_flags=parMapNet.input_flags, map_previous=state[0], update_type=parMapNet.update_type)

				tvec = torch.zeros(1, parIL.nTargets).float().cuda()
				tvec[0, target_lbl] = 1
				collision_ = torch.tensor([collision_seq[i]], dtype=torch.float32).cuda() # collision indicator is 0

				if parIL.use_ego_obsv:
					tensor_img_obsv = torch.tensor(img_obsv).float().cuda()
					tensor_depth_obsv = torch.tensor(depth_obsv).float().cuda()
					#print('tensor_img_obsv.shape = {}'.format(tensor_img_obsv.shape))
					#print('tensor_dets_obsv.shape = {}'.format(tensor_det_obsv.shape))
					enc_in = torch.cat((tensor_img_obsv, tensor_depth_obsv), 0).unsqueeze(0)
					ego_obsv_feat = ego_encoder(enc_in) # b x 512 x 1 x 1
					state = (map_next, p_next, tvec, collision_, ego_obsv_feat)
				else:
					state = (map_next, p_next, tvec, collision_)

			# Need to get the relative poses (towards the first frame) for the ground-truth
			#poses_epi = np.asarray(poses_epi)
			#rel_poses = dhh.relative_poses(poses=poses_epi)

			collision_seq = np.asarray(collision_seq, dtype=np.float32)
			action_seq = np.asarray(action_seq)
			cost_seq = np.asarray(cost_seq, dtype=np.float32)

			item = {}
			item['images'] = torch.from_numpy(imgs).float()
			item['points2D'] = points2D
			item['local3D'] = local3D
			item['actions'] = action_seq
			item['costs'] = torch.from_numpy(cost_seq).float()
			item['target_lbl'] = target_lbl
			item['pose'] = rel_poses
			item['abs_pose'] = abs_poses
			item['collisions'] = torch.from_numpy(collision_seq).float()
			item['scale'] = 1.0
			item['sseg'] = torch.from_numpy(ssegs).float()
			#item['dets'] = torch.from_numpy(dets).float()
			item['depths'] = torch.from_numpy(depths).float()
			item['images_obsv'] = torch.from_numpy(imgs_obsv).float()
			#item['dets_obsv'] = torch.from_numpy(dets_obsv).float()
			item['depths_obsv'] = torch.from_numpy(depths_obsv).float()
			return item

##=====================================================================================================
		
