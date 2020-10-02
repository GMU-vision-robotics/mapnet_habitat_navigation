import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
import os
import numpy as np
import math
import random
from mapNet import MapNet
from IL_Net import Encoder
from parameters_habitat import ParametersMapNet_Habitat, ParametersIL_Habitat
from dataloader_habitat import Habitat_MP3D_online
import helper as hl
#import data_helper as dh
import data_helper_habitat as dhh
import pickle
#import vis_test as vis

num_test_episodes = 10 #100

#os.environ["CUDA_VISIBLE_DEVICES"]="1"

def softmax(x):
	scoreMatExp = np.exp(np.asarray(x))
	return scoreMatExp / scoreMatExp.sum(0)

def evaluate_ILNet(parIL, parMapNet, mapNet, ego_encoder, test_iter, data):
	print("\nRunning validation on ILNet!")
	with torch.no_grad():
		policy_net = hl.load_model(model_dir=parIL.model_dir, model_name="ILNet", test_iter=test_iter)
		list_dist_to_goal, list_success, list_spl, list_soft_spl = [],[],[],[]
		episode_results = {} # store predictions in order to visualize

		for j in range(num_test_episodes):

			'''
			imgs = np.zeros((data.seq_len, 3, data.cropSize[1], data.cropSize[0]), dtype=np.float32)
			imgs_obsv = np.zeros((data.seq_len, 3, data.cropSizeObsv[1], data.cropSizeObsv[0]), dtype=np.float32)
			ssegs = np.zeros((data.seq_len, 1, data.cropSize[1], data.cropSize[0]), dtype=np.float32)
			dets = np.zeros((data.seq_len, data.dets_nClasses, data.cropSize[1], data.cropSize[0]), dtype=np.float32)
			dets_obsv = np.zeros((data.seq_len, 1, data.cropSizeObsv[1], data.cropSizeObsv[0]), dtype=np.float32)
			points2D, local3D = [], []
			'''
			abs_poses, rel_poses, action_seq, cost_seq, collision_seq = [], [], [], [], []

			observations = data.env.reset()
			data.dg.reset_metric(data.env._current_episode)
			# compare len_shortest_path with seq_len
			while True:
				len_shortest_path = len(data.env._current_episode.shortest_paths[0])
				if len_shortest_path > data.seq_len and observations['objectgoal'][0] in [0, 5, 6, 8, 10, 13]:
					break
				else:
					#print('shortest path is shorter than seq_len. Reset environment ...')
					observations = data.env.reset()

			done = 0

			#imgData, ssegData, detData, points2D_step, local3D_step = dhh.getImageData(observations, 
			#	data.dets_nClasses, data.cropSize, data.orig_res, data.pixFormat, data.normalize)
			#img_obsv, det_obsv = dhh.getImageData(observations, 1, data.cropSizeObsv, data.orig_res, data.pixFormat, 
			#	data.normalize, get3d=False)
			imgData = dhh.preprocess_img(observations['rgb'], data.cropSize, data.pixFormat, data.normalize)
			img_obsv = dhh.preprocess_img(observations['rgb'], data.cropSizeObsv, data.pixFormat, data.normalize)
			points2D_step, local3D_step = dhh.depth_to_3D(observations, data.hfov, data.orig_res, data.cropSize)
			ssegData = np.expand_dims(observations['semantic'], 0).astype(float)
			depthData = dhh.preprocess_depth(observations['depth'], data.cropSize)
			depth_obsv = dhh.preprocess_depth(observations['depth'], data.cropSizeObsv)			
			#detData = np.zeros((data.dets_nClasses, data.cropSize[1], data.cropSize[0]), dtype=np.float32) # dummy detections
			#det_obsv = np.zeros((1, data.cropSizeObsv[1], data.cropSizeObsv[0]), dtype=np.float32) # dummy detection observation


			#gps = np.copy(observations['gps'])
			#heading = np.copy(observations['heading'])
			#current_pose = np.concatenate((gps, heading))
			#poses_epi.append(current_pose)
			agent_pose = dhh.get_sim_location(data.env)
			abs_poses.append(agent_pose)
			rel = dhh.get_rel_pose(pos2=abs_poses[0], pos1=abs_poses[0]) # get the relative pose with respect to the first pose in the sequence
			rel_poses.append(rel)	

			target_lbl = observations['objectgoal'][0]

			# Just ignore collision right now till I figure out how to deal with it
			collision_seq.append(0)
			# use shortest path to find the best_action
			best_action = data.dg.get_next_action(data.env._current_episode)
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
			deterministic = False
	#================================================================================================================
			for t in range(1, parIL.max_steps+1):
				pred_costs = policy_net(state, parIL.use_ego_obsv) # apply policy for single step
				pred_costs = pred_costs.view(-1).cpu().numpy()

				if deterministic:
					pred_label = np.argmin(pred_costs)
					pred_action = data.actions[pred_label]
				else:
					# choose the action with a certain prob
					pred_probs = softmax(-pred_costs)
					pred_label = np.random.choice(len(data.actions), 1, p=pred_probs)[0]
					pred_action = data.actions[pred_label]

				print(t, pred_action)
				# pred_action is a param action
				action = dhh.paramAction_to_configAction(pred_action)

				observations = data.env.step(action)
				#imgData, ssegData, detData, points2D_step, local3D_step = dhh.getImageData(observations, 
				#	data.dets_nClasses, data.cropSize, data.orig_res, data.pixFormat, data.normalize)
				#img_obsv, det_obsv = dhh.getImageData(observations, 1, data.cropSizeObsv, data.orig_res, data.pixFormat, 
				#	data.normalize, get3d=False)
				imgData = dhh.preprocess_img(observations['rgb'], data.cropSize, data.pixFormat, data.normalize)
				img_obsv = dhh.preprocess_img(observations['rgb'], data.cropSizeObsv, data.pixFormat, data.normalize)
				points2D_step, local3D_step = dhh.depth_to_3D(observations, data.hfov, data.orig_res, data.cropSize)
				ssegData = np.expand_dims(observations['semantic'], 0).astype(float)
				depthData = dhh.preprocess_depth(observations['depth'], data.cropSize)
				depth_obsv = dhh.preprocess_depth(observations['depth'], data.cropSizeObsv)					
				#detData = np.zeros((data.dets_nClasses, data.cropSize[1], data.cropSize[0]), dtype=np.float32) # dummy detections
				#det_obsv = np.zeros((1, data.cropSizeObsv[1], data.cropSizeObsv[0]), dtype=np.float32) # dummy detection observation

				#gps = np.copy(observations['gps'])
				#heading = np.copy(observations['heading'])
				#current_pose = np.concatenate((gps, heading))
				#poses_epi.append(current_pose)
				agent_pose = dhh.get_sim_location(data.env)
				abs_poses.append(agent_pose)
				rel = dhh.get_rel_pose(pos2=abs_poses[t], pos1=abs_poses[0]) # get the relative pose with respect to the first pose in the sequence
				rel_poses.append(rel)	

				metrics = self.env.get_metrics()
				collision_seq.append( metrics['collisions']['is_collision'] )

				param_action = dhh.configAction_to_paramAction(action)
				action_seq.append(param_action)
				# use shortest path to find the best_action
				best_action = data.dg.get_next_action(data.env._current_episode)
				cost_seq.append(dhh.configAction_to_costOfParamAction(best_action))

				observations = None

	#===============================================================================================================
				# predict next action
				collision = 0
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
					p_gt = dh.build_p_gt(parMapNet, pose_gt_batch=np.expand_dims(pose_gt_batch, axis=1)).squeeze(1)
					p_next, map_next = mapNet.forward_single_step(local_info=batch_next, t=t, input_flags=parMapNet.input_flags,
						map_previous=state[0], p_given=p_gt, update_type=parMapNet.update_type)
				else:
					p_next, map_next = mapNet.forward_single_step(local_info=batch_next, t=t, 
						input_flags=parMapNet.input_flags, map_previous=state[0], update_type=parMapNet.update_type)

				tvec = torch.zeros(1, parIL.nTargets).float().cuda()
				tvec[0, target_lbl] = 1
				collision_ = torch.tensor([collision_seq[t]], dtype=torch.float32).cuda() # collision indicator is 0

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

				# check if reach the goal
				data.dg.update_metric(data.env._current_episode)
				path_dist = data.dg._metric
				#print('path_dist = {}'.format(path_dist))
				if path_dist <= parIL.dist_from_goal:
					data.env.step(0) # take stop action
					done = 1
					break
				# check if it's successful
				#metrics = data.env.get_metrics()
				#if metrics['success'] > 0:
				#	break
			# save results to list
			metrics = data.env.get_metrics()
			list_dist_to_goal.append(metrics['distance_to_goal'])
			list_success.append(metrics['success'])
			list_spl.append(metrics['spl'])
			list_soft_spl.append(metrics['softspl'])

#=======================================================================================================
			print('*****************done = {}********************'.format(done))
			# Need to get the relative poses (towards the first frame) for the ground-truth
			#poses_epi = np.asarray(poses_epi)
			#rel_poses = dhh.relative_poses(poses=poses_epi)

			collision_seq = np.asarray(collision_seq, dtype=np.float32)
			action_seq = np.asarray(action_seq)
			cost_seq = np.asarray(cost_seq, dtype=np.float32)

			episode_results[j] = (action_seq, target_lbl, done)

#=========================================== analyze result =====================================================
		episode_results_path = parIL.model_dir+'episode_results_'+str(test_iter)+'.pkl'
		#if test:
			#episode_results_path = parIL.model_dir+'episode_results_test_'+str(test_iter)+'.pkl'
		with open(episode_results_path, 'wb') as f:
			pickle.dump(episode_results, f)
		
		res_file = open(parIL.model_dir+"val_"+parIL.model_id+".txt", "a+")
		success_rate = 1.0 * sum(list_success) / len(list_success)
		mean_distance_to_goal = 1.0 * sum(list_dist_to_goal) / len(list_dist_to_goal)
		mean_spl = 1.0 * sum(list_spl) / len(list_spl)
		mean_soft_spl = 1.0 * sum(list_soft_spl) / len(list_soft_spl)
		print("Test iter:", test_iter, "Success rate:", success_rate, "Mean spl:", mean_spl)
		res_file.write("Test iter:" + str(test_iter) + "\n")
		res_file.write("Test set:" + str(num_test_episodes) + "\n")
		res_file.write("Success rate:" + str(success_rate) + "\n")
		res_file.write("Mean distance_to_goal:" + str(mean_distance_to_goal) + "\n")
		res_file.write("Mean spl:" + str(mean_spl) + "\n")
		res_file.write("Mean softspl:" + str(mean_soft_spl) + "\n")
		res_file.write("\n")
		res_file.close()


if __name__ == '__main__':
	parMapNet = ParametersMapNet_Habitat()
	parIL = ParametersIL_Habitat()
	action_list = np.asarray(parMapNet.action_list)

	# Need to load the trained MapNet
	if parIL.finetune_mapNet: # choose whether to use a finetuned mapNet model or not
		mapNet_model = hl.load_model(model_dir=parIL.model_dir, model_name="MapNet", test_iter=parIL.test_iters)
	else:
		mapNet_model = hl.load_model(model_dir=parIL.mapNet_model_dir, model_name="MapNet", test_iter=parIL.mapNet_iters)
	# If we are not using a trained mapNet model then define a new one
	#mapNet_model = MapNet(parMapNet, update_type=parMapNet.update_type, input_flags=parMapNet.input_flags) #Encoder(par)
	#mapNet_model.cuda()
	#mapNet_model.eval()
	
	ego_encoder = Encoder()
	ego_encoder.cuda()
	ego_encoder.eval()

	mp3d = Habitat_MP3D_online(par=parIL, seq_len=parIL.seq_len, config_file=parIL.test_config, 
		action_list=parIL.action_list)

	evaluate_ILNet(parIL, parMapNet, mapNet_model, ego_encoder, test_iter=parIL.test_iters, data=mp3d) 
					 