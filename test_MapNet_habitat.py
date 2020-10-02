
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import os
import numpy as np
import random
import cv2
import math
import matplotlib.pyplot as plt
#import data_helper as dh
import helper as hl
#from parameters import Parameters
from parameters_habitat import ParametersMapNet_Habitat
from mapNet import MapNet
from dataloader_habitat import Habitat_MP3D
import time
import pickle

#os.environ["CUDA_VISIBLE_DEVICES"]="1"

num_test_episodes = 10

def get_pose(par, p):
    # get the location and orientation of the max value
    # p is r x h x w
    p_tmp = p.view(-1)
    m, _ = torch.max(p_tmp, 0)
    p_tmp = p_tmp.view(par.orientations, par.global_map_dim[0], par.global_map_dim[1])
    p_tmp = p_tmp.detach().cpu().numpy()
    inds = np.where(p_tmp==m.data.item())
    r = inds[0][0] # discretized orientation
    zb = inds[1][0]
    xb = inds[2][0]
    return r, zb, xb


def get_angle_from_ind(par, ind):
    angle = (ind / par.orientations) * (2*np.pi)
    if angle < -np.pi:
        angle += 2 * np.pi
    if angle > np.pi:
        angle -= 2 * np.pi
    return np.degrees(angle)


def undo_discretization(par, zb, xb):
    #zb = (par.global_map_dim[1]-1)-zb
    x = (xb-(par.global_map_dim[0]-1)/2.0) * par.cell_size
    z = (zb-(par.global_map_dim[1]-1)/2.0) * par.cell_size
    return z, x


def evaluate_MapNet(par, test_iter, test_data):
    print("\nRunning validation on MapNet!")
    print("Testing on", par.seq_len, "length episodes.")
    with torch.no_grad():
        # Load the model
        test_model = hl.load_model(model_dir=par.model_dir, model_name="MapNet", test_iter=test_iter)

        episode_results = {} # store predictions and ground-truth in order to visualize
        error_pos_list, error_angle_list = [], []
        #angle_acc = 0
        
        # run 100 episodes
        for i in range(num_test_episodes):

            ex = test_data.get_episode() #test_data[1] # 1 is a random index number
            imgs_seq = ex["images"]
            points2D_seq = ex["points2D"]
            local3D_seq = ex["local3D"]
            pose_gt_seq = ex["pose"]
            abs_pose_gt_seq = ex["abs_pose"]
            sseg_seq = ex["sseg"]
            #dets_seq = ex["dets"]
            depths_seq = ex['depths']
            #scale = ex["scale"]

            # for now assume that test_batch_size=1
            imgs_batch = imgs_seq.unsqueeze(0)
            pose_gt_batch = np.expand_dims(pose_gt_seq, axis=0)
            sseg_batch = sseg_seq.unsqueeze(0)
            #dets_batch = dets_seq.unsqueeze(0)
            depths_batch = depths_seq.unsqueeze(0)
            points2D_batch, local3D_batch = [], [] # add another dimension for the batch
            points2D_batch.append(points2D_seq)
            local3D_batch.append(local3D_seq)
            #p_gt_batch = dh.build_p_gt(par, pose_gt_batch)
            #print("Input time:", time.time()-start)
            #print('scene: {}'.format(scene))
            #print('imgs_name : {}'.format(imgs_name))
            #start = time.time()
            local_info = (imgs_batch.cuda(), points2D_batch, local3D_batch, sseg_batch.cuda(), depths_batch.cuda())
            p_pred, map_pred = test_model(local_info, update_type=par.update_type, input_flags=par.input_flags)
            # remove the tensors from gpu memory
            p_pred = p_pred.cpu().detach()
            map_pred = map_pred.cpu().detach()
            #start = time.time()
            # Remove the first step in any sequence since it is a constant
            p_pred = p_pred[:,1:,:,:,:]
            pose_gt_batch = pose_gt_batch[:,1:,:]
            pred_pose = np.zeros((par.seq_len, 3), dtype=np.float32)
            episode_error_pos, episode_error_angle = [], [] # add the errors of the episode so you can do the median
            for s in range(p_pred.shape[1]): # seq_len-1
                # convert p to coordinates and orientation values
                rb, zb, xb = get_pose(par, p=p_pred[0,s,:,:,:])
                # undiscretize the map coords and get the angle value
                z_pred, x_pred = undo_discretization(par, zb, xb)
                pred_coords = np.array([x_pred, z_pred], dtype=np.float32)
                pred_angle = get_angle_from_ind(par, ind=rb)

                #r_gt = np.floor( np.mod(pose_gt_batch[0,s,2]/(2*np.pi), 1) * par.orientations )
                #print('Pred angle:', angle_pred, ' GT angle:', np.degrees(pose_gt_batch[0,s,2]))
                #print('Pred ind:', rb, ' GT ind:', r_gt)
                
                gt_coords = pose_gt_batch[0,s,:2]
                gt_angle = np.degrees(pose_gt_batch[0,s,2])
                
                error_pos = np.linalg.norm( gt_coords - pred_coords )
                error_angle = min( abs(gt_angle-pred_angle), 360-abs(gt_angle-pred_angle) )
                
                episode_error_pos.append(error_pos)
                episode_error_angle.append(error_angle)


                # store predictions and gt
                pred_pose[s+1, :] = np.array([x_pred, z_pred, np.radians(pred_angle)], dtype=np.float32)

            episode_results[i] = (pose_gt_seq, abs_pose_gt_seq, pred_pose)

            episode_error_pos = np.asarray(episode_error_pos)
            episode_error_angle = np.asarray(episode_error_angle)

            error_pos_list.append( np.median(episode_error_pos) )
            error_angle_list.append( np.median(episode_error_angle) )
            #print("Metrics time:", time.time()-start)

        #with open('examples/MapNet/episode_results.pkl', 'wb') as f:
        with open(par.model_dir+'episode_results_'+str(test_iter)+'.pkl', 'wb') as f:    
            pickle.dump(episode_results, f)
    
        error_pos_list = np.asarray(error_pos_list)
        error_angle_list = np.asarray(error_angle_list)
        #print(error_list)
        error_pos_res = error_pos_list.mean()
        error_angle_res = error_angle_list.mean()

        #angle_acc = angle_acc / float(num_test_episodes) # ** need to change this to number of steps
        print("Test iter:", test_iter, "Position error:", error_pos_res, "Angle error:", error_angle_res)
        res_file = open(par.model_dir+"val_"+par.model_id+".txt", "a+")
        res_file.write("Test iter:" + str(test_iter) + "\n")
        res_file.write("Test set:" + str(num_test_episodes) + "\n")
        res_file.write("Position error:" + str(error_pos_res) + "\n")
        res_file.write("Angle error:" + str(error_angle_res) + "\n")
        res_file.write("\n")
        res_file.close()


if __name__ == '__main__':
    par = ParametersMapNet_Habitat()

    mp3d = Habitat_MP3D(par, seq_len=par.seq_len, config_file=par.test_config)
    #, action_list=par.action_list, 
    #    with_shortest_path=par.with_shortest_path)

    print("Loading the test data...")
    #evaluate_MapNet(par, test_iter=4500, test_data=mp3d)
    evaluate_MapNet(par, test_iter=20, test_data=mp3d)










'''
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
