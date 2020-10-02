
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
import os
import numpy as np
import random
import cv2
import math
import pickle
import matplotlib.pyplot as plt
import data_helper_habitat as dhh
import helper as hl
from mapNet import MapNet
#from parameters import Parameters
from parameters_habitat import ParametersMapNet_Habitat
from dataloader_habitat import Habitat_MP3D
#from test_MapNet import evaluate_MapNet
import time

#os.environ["CUDA_VISIBLE_DEVICES"]="0"

def get_minibatch(par, data):
    imgs_batch = torch.zeros(par.batch_size, par.seq_len, 3, par.crop_size[1], par.crop_size[0])
    #pose_gt_batch = torch.zeros(par.batch_size, par.seq_len, 3)
    pose_gt_batch = np.zeros((par.batch_size, par.seq_len, 3), dtype=np.float32)
    sseg_batch = torch.zeros(par.batch_size, par.seq_len, 1, par.crop_size[1], par.crop_size[0])
    #dets_batch = torch.zeros(par.batch_size, par.seq_len, par.dets_nClasses, par.crop_size[1], par.crop_size[0])
    depths_batch = torch.zeros(par.batch_size, par.seq_len, 1, par.crop_size[1], par.crop_size[0])
    points2D_batch, local3D_batch = [], []
    for k in range(par.batch_size):
        ex = data.get_episode() 
        #ex = data[0] # episode
        imgs_seq = ex["images"]
        points2D_seq = ex["points2D"]
        local3D_seq = ex["local3D"]
        pose_gt_seq = ex["pose"]
        sseg_seq = ex["sseg"]
        #dets_seq = ex["dets"]
        depths_seq = ex['depths']
        #print(sseg_seq.shape)
        #sseg_tmp = sseg_seq.view(-1)
        #print(sseg_tmp.shape)
        #print(torch.max(sseg_tmp))
        #print(pose_gt_seq)
        imgs_batch[k,:,:,:,:] = imgs_seq
        pose_gt_batch[k,:,:] = pose_gt_seq
        sseg_batch[k,:,:,:,:] = sseg_seq
        #dets_batch[k,:,:,:,:] = dets_seq
        depths_batch[k,:,:,:,:] = depths_seq
        points2D_batch.append(points2D_seq) # nested list of batch_size x seq_len x n_points x 2
        local3D_batch.append(local3D_seq) # nested list of batch_size x seq_len x n_points x 3
    return (imgs_batch.cuda(), points2D_batch, local3D_batch, pose_gt_batch, sseg_batch.cuda(), depths_batch.cuda())


if __name__ == '__main__':
    par = ParametersMapNet_Habitat()
    # Init the model
    mapNet_model = MapNet(par, update_type=par.update_type, input_flags=par.input_flags) #Encoder(par)
    mapNet_model.cuda()
    mapNet_model.train()
    optimizer = optim.Adam(mapNet_model.parameters(), lr=par.lr_rate)
    scheduler = StepLR(optimizer, step_size=par.step_size, gamma=par.gamma)
    # Load the dataset
    print("Loading the training data...")
    mp3d = Habitat_MP3D(par, seq_len=par.seq_len, config_file=par.train_config)

    '''
    # save sampled data to reproduce validation results
    avd_file = open(par.model_dir+"mp3d_data.pkl", 'wb')
    pickle.dump(mp3d, avd_file)
    '''

    log = open(par.model_dir+"train_log_"+par.model_id+".txt", 'w')
    hl.save_params(par, par.model_dir, name="mapNet")
    loss_list=[]

    #all_ids = list(range(len(mp3d)))
    #test_ids = all_ids[::100] # select a small subset for testing
    #train_ids = list(set(all_ids) - set(test_ids)) # the rest for training
    
    #nData = len(train_ids)
    #iters_per_epoch = int(nData / float(par.batch_size))
    iters_per_epoch = 1000
    log.write("Iters_per_epoch:"+str(iters_per_epoch)+"\n")
    print("Iters per epoch:", iters_per_epoch)

    for ep in range(par.nEpochs):
        #random.shuffle(train_ids)
        for i in range(iters_per_epoch):
            iters = i + ep*iters_per_epoch # actual number of iterations given how many epochs passed

            # Sample the training minibatch
            #start = time.time()
            batch = get_minibatch(par, data=mp3d)
            (imgs_batch, points2D_batch, local3D_batch, pose_gt_batch, sseg_batch, depths_batch) = batch                                                                    
            #log.write("Get minibatch time:"+str(time.time()-start)+"\n")

            p_gt_batch = dhh.build_p_gt(par, pose_gt_batch)
            
            #start = time.time()
            local_info = (imgs_batch, points2D_batch, local3D_batch, sseg_batch, depths_batch)
            p_pred, map_pred = mapNet_model(local_info, update_type=par.update_type, 
                                                        input_flags=par.input_flags, p_gt=None)
            #print("MapNet:", time.time()-start)
            
            # Backprop the loss
            loss = mapNet_model.build_loss(p_pred, p_gt_batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

            # Show, plot, save, test
            if iters % par.show_interval == 0:
                log.write("Epoch:"+str(ep)+" ITER:"+str(iters)+" Loss:"+str(loss.data.item())+"\n")
                print("Epoch:", str(ep), " ITER:", str(iters), " Loss:", str(loss.data.item()))
                log.flush()

            if iters > 0:
                loss_list.append(loss.data.item())
            if iters % par.plot_interval == 0 and iters>0:
                hl.plot_loss(loss=loss_list, epoch=ep, iteration=iters, step=1, loss_name="NLL", loss_dir=par.model_dir)

            if iters % par.save_interval == 0:
                hl.save_model(model=mapNet_model, model_dir=par.model_dir, model_name="MapNet", train_iter=iters)
            '''
            if iters % par.test_interval == 0:
            
                mp3d_test = Habitat_MP3D(par, seq_len=par.seq_len, config_file=par.test_config, action_list=par.action_list, 
                    with_shortest_path=par.with_shortest_path)
                evaluate_MapNet(par, test_iter=iters, test_data=mp3d_test)
            '''

        # ** Do a full-blown debugging, check the values of the tensors









'''
# Define the map at time 0
map_0 = np.ones((par.batch_size, par.map_embedding, par.global_map_dim[0], par.global_map_dim[1]), dtype=np.float32)
map_0[0,:,:,:] = map_0[0,:,:,:] / float(par.map_embedding * par.global_map_dim[0] * par.global_map_dim[1])
map_0 = torch.tensor(map_0, dtype=torch.float32).cuda()
#print(torch.sum(map_0))
#raise Exception("ppp")

# Define the map at time 0
#map_0 = np.ones((par.batch_size, par.map_embedding, par.global_map_dim[0], par.global_map_dim[1]), dtype=np.float32)
#for k in range(par.batch_size):
#    map_0[k,:,:,:] = map_0[k,:,:,:] / float(par.map_embedding * par.global_map_dim[0] * par.global_map_dim[1])
#map_0 = torch.tensor(map_0, dtype=torch.float32).cuda()
#p_, map_next = mapNet_model(local_info, map_previous=map_0) # need to add the map from previous timesteps

p_, map_next = mapNet_model(local_info, map_previous=map_0) # need to add the map from previous timesteps


# Trying to understand how p is reshaped and how to get the map coordinate for the largest value
#print(p_gt_batch.shape)
#print(p_gt_batch[0,0,0,14,14])
#p_tmp = p_gt_batch[0,0,:,:,:].view(-1)
#ind = torch.nonzero(p_tmp)
#print(ind)
#p_gt_ = p_tmp.view(par.orientations, par.global_map_dim[0], par.global_map_dim[1])
#print(p_gt_[0,14,14])

#p_gt_batch = p_gt_batch.view(par.batch_size*par.seq_len, par.orientations*par.global_map_dim[0]*par.global_map_dim[1])
#ind = torch.nonzero(p_gt_batch[0,:])
#print(ind)
#p_gt_batch = p_gt_batch.view(par.batch_size, par.seq_len, par.orientations, par.global_map_dim[0], par.global_map_dim[1])
#print(p_gt_batch[0,0,0,14,14])


# The returned image feat are 1 x 512 x 23 x 40, much smaller than the image. This depends on how shallow the network is.
# Three choices: 
# 1) Do the ground projection given the current size
# 2) Upscale the features to match the img size
# 3) Change all resNet stride parameters to 1

# Resize the features to the image/depth resolution
img_feat_resized = F.interpolate(img_feat, size=(par.crop_size[1], par.crop_size[0]), mode='nearest')
print(img_feat_resized.shape)

# Create the grid and discretize the set of coordinates into the bins
# Points2D holds the image pixel coordinates with valid depth values
# Local3D holds the X,Y,Z coordinates that correspond to the points2D

# For each local3d find which bin it belongs to
valid = []
map_coords = np.zeros((local3D.shape[0], 2), dtype=np.int)
#map_occ = np.zeros((par.map_dim[0], par.map_dim[1]), dtype=np.float32)
for i in range(local3D.shape[0]):
    x, z = local3D[i,0], local3D[i,2]
    xb = int( math.floor(x/par.cell_size) + (par.map_dim[0]-1)/2.0 )
    zb = int( math.floor(z/par.cell_size) + (par.map_dim[1]-1)/2.0 )
    map_coords[i,0] = xb
    zb = (par.map_dim[1]-1)-zb # mirror the z axis so that the origin is at the bottom
    map_coords[i,1] = zb
    # keep bin coords within dimensions
    #if xb<0 or zb<0 or xb>=par.map_dim[0] or zb>=par.map_dim[1]:
    if xb>=0 and zb>=0 and xb<par.map_dim[0] and zb<par.map_dim[1]:
        valid.append(i)
        #map_occ[zb,xb] = 1
        #invalid.append(i)
        #print(x, z, xb, zb)

#plt.imshow(map_occ)
#plt.show()
#raise Exception("gggg")

valid = np.asarray(valid, dtype=np.int)
#print(invalid)
#print(local3D.shape)
points2D = points2D[valid,:]
local3D = local3D[valid,:]
map_coords = map_coords[valid,:]
#print(local3D.shape)


feat_dict = {} # keep track of the feature vectors that project to each map location

# go through each point in points2D and collect the feature from the img_feat_resized
# add it to the right map_coord in the map. if the location is not empty, then do max pooling 
# of the current feature with the existing one (channel wise) 
for i in range(points2D.shape[0]):
    pix_x, pix_y = points2D[i,0], points2D[i,1]
    map_x, map_y = map_coords[i,0], map_coords[i,1]
    #print(pix_x, pix_y, map_x, map_y)
    pix_feat = img_feat_resized[0, :, pix_y, pix_x]
    #print(pix_feat.shape)
    if (map_x,map_y) not in feat_dict.keys():
        feat_dict[(map_x,map_y)] = pix_feat.unsqueeze(0)
    else:
        feat_dict[(map_x,map_y)] = torch.cat(( feat_dict[(map_x,map_y)], pix_feat.unsqueeze(0) ), 0) # cat with the existing features in that bin
        #print(feat_dict[(map_x,map_y)].shape)

    # check whether the location already contains features
    #grid_feat = grid[map_x, map_y, :] # in the beginning all zeroes
    #print(torch.sum(grid_feat))
    #if torch.sum(grid_feat)==0: # if all zeroes, then just add the features as they are
    #    grid[map_x, map_y, :] = pix_feat
    #else:
    #print(torch.sum(grid_feat))
#print(feat_dict[(map_x,map_y)].shape)

grid = np.zeros((par.map_dim[0], par.map_dim[1], img_feat_resized.shape[1]), dtype=np.float32)
grid = torch.tensor(grid, dtype=torch.float32)
# Do max-pooling over the bins. Is the max-pooling being done channel-wise?
for (map_x,map_y) in feat_dict.keys():
    bin_feats = feat_dict[(map_x,map_y)] # [n x d] n:number of feature vectors projected, d:feat_dim
    bin_feat, ind = torch.max(bin_feats,0) # [1 x d]
    grid[map_y, map_x, :] = bin_feat

    raise Exception("gggg")

# Pass the grid to a CNN to get the observation with 32-D embeddings
'''
