import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
import os
import numpy as np
import math
import random
import pickle
from dataloader_habitat import Habitat_MP3D_IL
from IL_Net import ILNet, Encoder
from mapNet import MapNet
from parameters_habitat import ParametersIL_Habitat, ParametersMapNet_Habitat
import helper as hl
#import data_helper as dh
import data_helper_habitat as dhh
#from QNet import DQN, DQN_2, Encoder
from itertools import chain
from copy import deepcopy
#from test_IL import evaluate_ILNet

#os.environ["CUDA_VISIBLE_DEVICES"]="0"

def get_minibatch(batch_size, tvec_dim, seq_len, nActions, data):
    imgs_batch = torch.zeros(batch_size, seq_len, 3, data.cropSize[1], data.cropSize[0]).float().cuda()
    sseg_batch = torch.zeros(batch_size, seq_len, 1, data.cropSize[1], data.cropSize[0]).float().cuda()
    #dets_batch = torch.zeros(batch_size, seq_len, data.dets_nClasses, data.cropSize[1], data.cropSize[0]).float().cuda()
    depths_batch = torch.zeros(batch_size, seq_len, 1, data.cropSize[1], data.cropSize[0]).cuda()
    imgs_obsv_batch = torch.zeros(batch_size, seq_len, 3, data.cropSizeObsv[1], data.cropSizeObsv[0]).float().cuda()
    #dets_obsv_batch = torch.zeros(batch_size, seq_len, 1, data.cropSizeObsv[1], data.cropSizeObsv[0]).float().cuda()
    depths_obsv_batch = torch.zeros(batch_size, seq_len, 1, data.cropSizeObsv[1], data.cropSizeObsv[0]).float().cuda()
    tvec_batch = torch.zeros(batch_size, tvec_dim).float().cuda()
    pose_gt_batch = np.zeros((batch_size, seq_len, 3), dtype=np.float32)
    collisions_batch = torch.zeros(batch_size, seq_len).float().cuda()
    costs_batch = torch.zeros(batch_size, seq_len, nActions).float().cuda()
    points2D_batch, local3D_batch = [], []
    image_names, scenes, scales, actions = [], [], [], []
    for k in range(batch_size):
        ex = data.get_episode()
        imgs_batch[k,:,:,:,:] = ex["images"]
        sseg_batch[k,:,:,:,:] = ex["sseg"]
        #dets_batch[k,:,:,:,:] = ex['dets']
        depths_batch[k,:,:,:,:] = ex['depths']
        imgs_obsv_batch[k,:,:,:,:] = ex['images_obsv']
        #dets_obsv_batch[k,:,:,:,:] = ex['dets_obsv']
        depths_obsv_batch[k,:,:,:,:] = ex['depths_obsv']
        points2D_batch.append(ex["points2D"]) # nested list of batch_size x n_points x 2
        local3D_batch.append(ex["local3D"]) # nested list of batch_size x n_points x 3
        # Label of the target object for each episode
        tvec_batch[k,ex["target_lbl"]] = 1
        # We need to keep other info to allow us to do the steps later
        #image_names.append(ex['images_names'])
        #scenes.append(ex['scene'])
        scales.append(ex['scale'])
        pose_gt_batch[k,:,:] = ex["pose"]
        collisions_batch[k,:] = ex['collisions']
        actions.append(ex['actions'])
        costs_batch[k,:,:] = ex['costs']
    mapNet_batch = (imgs_batch, points2D_batch, local3D_batch, sseg_batch, depths_batch, pose_gt_batch)
    IL_batch = (imgs_obsv_batch, depths_obsv_batch, tvec_batch, collisions_batch, actions, costs_batch, scales)
    return mapNet_batch, IL_batch


def unroll_policy(parIL, parMapNet, policy_net, mapNet, ego_encoder, batch_size, tvec_dim, seq_len, nActions, data):
    imgs_batch = torch.zeros(batch_size, seq_len, 3, data.cropSize[1], data.cropSize[0]).float().cuda()
    sseg_batch = torch.zeros(batch_size, seq_len, 1, data.cropSize[1], data.cropSize[0]).float().cuda()
    #dets_batch = torch.zeros(batch_size, seq_len, data.dets_nClasses, data.cropSize[1], data.cropSize[0]).float().cuda()
    depths_batch = torch.zeros(batch_size, seq_len, 1, data.cropSize[1], data.cropSize[0]).cuda()
    imgs_obsv_batch = torch.zeros(batch_size, seq_len, 3, data.cropSizeObsv[1], data.cropSizeObsv[0]).float().cuda()
    #dets_obsv_batch = torch.zeros(batch_size, seq_len, 1, data.cropSizeObsv[1], data.cropSizeObsv[0]).float().cuda()
    depths_obsv_batch = torch.zeros(batch_size, seq_len, 1, data.cropSizeObsv[1], data.cropSizeObsv[0]).float().cuda()
    tvec_batch = torch.zeros(batch_size, tvec_dim).float().cuda()
    pose_gt_batch = np.zeros((batch_size, seq_len, 3), dtype=np.float32)
    collisions_batch = torch.zeros(batch_size, seq_len).float().cuda()
    costs_batch = torch.zeros(batch_size, seq_len, nActions).float().cuda()
    points2D_batch, local3D_batch = [], []
    image_names, scenes, scales, actions = [], [], [], []
    for k in range(batch_size):
        ex = data.get_item_policy(parIL, parMapNet, policy_net, mapNet, ego_encoder)
        imgs_batch[k,:,:,:,:] = ex["images"]
        sseg_batch[k,:,:,:,:] = ex["sseg"]
        #dets_batch[k,:,:,:,:] = ex['dets']
        depths_batch[k,:,:,:,:] = ex['depths']
        imgs_obsv_batch[k,:,:,:,:] = ex['images_obsv']
        #dets_obsv_batch[k,:,:,:,:] = ex['dets_obsv']
        depths_obsv_batch[k,:,:,:,:] = ex['depths_obsv']
        points2D_batch.append(ex["points2D"]) # nested list of batch_size x n_points x 2
        local3D_batch.append(ex["local3D"]) # nested list of batch_size x n_points x 3
        # Label of the target object for each episode
        tvec_batch[k,ex["target_lbl"]] = 1
        # We need to keep other info to allow us to do the steps later
        #image_names.append(ex['images_names'])
        #scenes.append(ex['scene'])
        scales.append(ex['scale'])
        pose_gt_batch[k,:,:] = ex["pose"]
        collisions_batch[k,:] = ex['collisions']
        actions.append(ex['actions'])
        costs_batch[k,:,:] = ex['costs']
    mapNet_batch = (imgs_batch, points2D_batch, local3D_batch, sseg_batch, depths_batch, pose_gt_batch)
    IL_batch = (imgs_obsv_batch, depths_obsv_batch, tvec_batch, collisions_batch, actions, costs_batch, scales)
    return mapNet_batch, IL_batch


# Choose how to sample the next minibatch
def select_minibatch(par, iters_done):
    sample = random.random()
    eps_threshold = par.EPS_END + (par.EPS_START-par.EPS_END) * math.exp(-1. * iters_done / par.EPS_DECAY)    
    if sample > eps_threshold:
        return 0
    else:
        return 1


def run_mapNet(parMapNet, mapNet, start_info, use_p_gt, pose_gt_batch):
    if use_p_gt:
        p_gt_batch = dhh.build_p_gt(parMapNet, pose_gt_batch)
        p_, map_ = mapNet(local_info=start_info, update_type=parMapNet.update_type, 
                                    input_flags=parMapNet.input_flags, p_gt=p_gt_batch)
        p_ = p_gt_batch.clone() # overwrite the predicted with the ground-truth location
    else:
        p_, map_ = mapNet(local_info=start_info, update_type=parMapNet.update_type, input_flags=parMapNet.input_flags)
    return p_, map_



if __name__ == '__main__':
    parMapNet = ParametersMapNet_Habitat()
    parIL = ParametersIL_Habitat()

    action_list = np.asarray(parMapNet.action_list)

    # init the model
    policy_net = ILNet(parIL, parMapNet.map_embedding, parMapNet.orientations, parIL.nTargets, len(action_list), parIL.use_ego_obsv)
    policy_net.train()
    policy_net.cuda()

    # Need to load the trained MapNet
    state_model = hl.load_model(model_dir=parIL.mapNet_model_dir, model_name="MapNet", 
                                    test_iter=parIL.mapNet_iters, eval=not(parIL.finetune_mapNet))
    # If we are not using a trained mapNet model then define a new one
    #state_model = MapNet(parMapNet, update_type=parMapNet.update_type, input_flags=parMapNet.input_flags) #Encoder(par)
    #state_model.cuda()
    #state_model.eval()

    if parIL.finetune_mapNet: # need to chain the parameters of mapNet and policy
        all_params = chain(policy_net.parameters(), state_model.parameters())
    else:
        all_params = policy_net.parameters()
    optimizer = optim.Adam(all_params, lr=parIL.lr_rate)
    scheduler = StepLR(optimizer, step_size=parIL.step_size, gamma=parIL.gamma)

    if parIL.use_ego_obsv:
        ego_encoder = Encoder()
        ego_encoder.cuda()
        ego_encoder.eval()

    # Collect the training episodes
    print("Loading training episodes...")
    mp3d = Habitat_MP3D_IL(par=parIL, seq_len=parIL.seq_len, config_file=parIL.train_config, 
        action_list=parIL.action_list)

    hl.save_params(parIL, parIL.model_dir, name="IL")
    hl.save_params(parMapNet, parIL.model_dir, name="mapNet")
    log = open(parIL.model_dir+"train_log_"+parIL.model_id+".txt", 'w')
    #nData = len(train_ids)
    #iters_per_epoch = int(nData / float(parIL.batch_size))
    iters_per_epoch = 1000
    log.write("Iters_per_epoch:"+str(iters_per_epoch)+"\n")
    print("Iters per epoch:", iters_per_epoch)
    loss_list = []

    #mapNet_batch, IL_batch = get_minibatch(batch_size=parIL.batch_size, tvec_dim=parIL.nTargets, 
    #                seq_len=parIL.seq_len, nActions=len(action_list), data=mp3d) # **** temp

    for ep in range(parIL.nEpochs):
        data_index = 0

        for i in range(iters_per_epoch):
            iters = i + ep*iters_per_epoch # actual number of iterations given how many epochs passed

            
            ch = select_minibatch(par=parIL, iters_done=iters)
            #ch = 0
            if ch:
                # Sample from the pre-selected episodes, which include random and shortest path sequences
                mapNet_batch, IL_batch = get_minibatch(batch_size=parIL.batch_size, tvec_dim=parIL.nTargets, 
                    seq_len=parIL.seq_len, nActions=len(action_list), data=mp3d)
            else:
                # Sample episodes by unrolling the policy to generate the sequence
                mapNet_batch, IL_batch = unroll_policy(parIL, parMapNet, policy_net, state_model, ego_encoder, batch_size=parIL.batch_size, 
                    tvec_dim=parIL.nTargets, seq_len=parIL.seq_len, nActions=len(action_list), data=mp3d)
            
            
            (imgs_batch, points2D_batch, local3D_batch, sseg_batch, depths_batch, pose_gt_batch) = mapNet_batch
            (imgs_obsv_batch, depths_obsv_batch, tvec_batch, collisions_batch, actions, costs_batch, scales) = IL_batch
            data_index += parIL.batch_size

            # get the map for every step from mapNet
            start_info = (imgs_batch, points2D_batch, local3D_batch, sseg_batch, depths_batch)
            if parIL.finetune_mapNet:
                p_, map_ = run_mapNet(parMapNet, state_model, start_info, parIL.use_p_gt, pose_gt_batch)
            else:
                with torch.no_grad():
                    p_, map_ = run_mapNet(parMapNet, state_model, start_info, parIL.use_p_gt, pose_gt_batch)

            if parIL.use_ego_obsv: # Get the encoding of the img/det in case you add it into the state
                with torch.no_grad():
                    enc_in = torch.cat((imgs_obsv_batch, depths_obsv_batch), 2)
                    enc_in = enc_in.view(parIL.batch_size*parIL.seq_len, 4, parIL.crop_size_obsv[1], parIL.crop_size_obsv[0])
                    ego_obsv_feat = ego_encoder(enc_in) # (b*seq) x 512 x 1 x 1
                    ego_obsv_feat = ego_obsv_feat.view(parIL.batch_size, parIL.seq_len, ego_obsv_feat.shape[1])
                state = (map_, p_, tvec_batch, collisions_batch, ego_obsv_feat)
            else: # state that goes in the IL net is: (map, p, tvec, collision)
                state = (map_, p_, tvec_batch, collisions_batch)

            policy_net.hidden = policy_net.init_hidden(parIL.batch_size, state_items=len(state)-1)
            pred_costs = policy_net(state, parIL.use_ego_obsv)

            loss = policy_net.build_loss(cost_pred=pred_costs, cost_gt=costs_batch, loss_weight=parIL.loss_weight)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

            # Show, plot, save, test
            if iters % parIL.show_interval == 0:
                log.write("Epoch:"+str(ep)+" ITER:"+str(iters)+" Loss:"+str(loss.data.item())+"\n")
                print("Epoch:", str(ep), " ITER:", str(iters), " Loss:", str(loss.data.item()))
                log.flush()

            if iters > 0:
                loss_list.append(loss.data.item())
            if iters % parIL.plot_interval == 0 and iters>0:
                hl.plot_loss(loss=loss_list, epoch=ep, iteration=iters, step=1, loss_name="L1", loss_dir=parIL.model_dir)

            if iters % parIL.save_interval == 0:
                hl.save_model(model=policy_net, model_dir=parIL.model_dir, model_name="ILNet", train_iter=iters)
                if parIL.finetune_mapNet:
                    hl.save_model(model=state_model, model_dir=parIL.model_dir, model_name="MapNet", train_iter=iters)

            # We don't do test
            #if iters % parIL.test_interval == 0:
            #    evaluate_ILNet(parIL, parMapNet, mapNet=state_model, ego_encoder=ego_encoder, test_iter=iters, 
            #                                    test_ids=test_ids, test_data=mp3d_test, action_list=action_list)

            