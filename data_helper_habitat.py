import numpy as np
import os
import cv2
import matplotlib.pyplot as plt
import random
import math
import visualpriors
import torchvision.transforms.functional as TF
from PIL import Image
import torch
from sklearn.decomposition import PCA
from habitat_sim.utils.common import d3_40_colors_rgb
import quaternion
from functools import reduce

'''
def convert_image_by_pixformat_normalize(src_image, pix_format, normalize):
    if pix_format == 'NCHW':
        src_image = src_image.transpose((2, 0, 1))
    if normalize:
        src_image = src_image.astype(np.float32) / 255.0 #* 2.0 - 1.0
    return src_image
'''

'''
hfov = float(79) * np.pi / 180.
# unproject pixels to the 3D camera coordinate frame
def depth_to_3D(depth, orig_res, crop_res):
    non_zero_inds = np.where(depth>0) # get all non-zero points
    #print('sum(depth) = {}'.format(np.sum(depth)))
    points2D = np.zeros((len(non_zero_inds[0]), 2), dtype=np.int)
    points2D[:,0] = non_zero_inds[1] # inds[1] is x (width coordinate)
    points2D[:,1] = non_zero_inds[0]

    # scale the intrinsics based on the new resolution
    fx, fy, cx, cy = 1./np.tan(hfov/2.)*320.0, 1./np.tan(hfov/2.)*240.0, 320.0, 240.0
    fx *= crop_res[0] / float(orig_res[0])
    fy *= crop_res[1] / float(orig_res[1])
    cx *= crop_res[0] / float(orig_res[0])
    cy *= crop_res[1] / float(orig_res[1])
    #print(fx, fy, cx, cy)
    # scale the depth based on the given AVD scale parameter
    #depth = depth/1000.0 #float(scale)
    
    # unproject the points
    z = depth[points2D[:,1], points2D[:,0]]
    local3D = np.zeros((points2D.shape[0], 3), dtype=np.float32)
    a = points2D[:,0]-cx
    b = points2D[:,1]-cy
    q1 = a[:,np.newaxis]*z[:,np.newaxis] / fx
    q2 = b[:,np.newaxis]*z[:,np.newaxis] / fy
    local3D[:,0] = q1.reshape(q1.shape[0])
    local3D[:,1] = q2.reshape(q2.shape[0])
    local3D[:,2] = z
    return points2D, local3D
'''

def depth_to_3D(obs, hfov, orig_res, crop_size):
    K = np.array([
        [1 / np.tan(hfov / 2.), 0., 0., 0.],
        [0., 1 / np.tan(hfov / 2.), 0., 0.],
        [0., 0.,  1, 0],
        [0., 0., 0, 1]])
    xs, ys = np.meshgrid(np.linspace(-1,1,orig_res[0]), np.linspace(1,-1,orig_res[1]))
    depth = obs['depth'][...,0].reshape(1, orig_res[0], orig_res[1])
    xs = xs.reshape(1,orig_res[0],orig_res[1])
    ys = ys.reshape(1,orig_res[0],orig_res[1])
    # Unproject
    # negate depth as the camera looks along -Z
    xys = np.vstack((xs * depth , ys * depth, -depth, np.ones(depth.shape))) # 4 x 128 x 128
    xys = xys.reshape(4, -1)
    xy_c0 = np.matmul(np.linalg.inv(K), xys)
    #print(xy_c0.shape)
    local3D = np.zeros((xy_c0.shape[1],3), dtype=np.float32)
    local3D[:,0] = xy_c0[0,:]
    local3D[:,1] = xy_c0[1,:]
    local3D[:,2] = xy_c0[2,:]
    # create the points2D containing all image coordinates
    # ** not sure if y should be linspace from 127 to 0
    x, y = np.meshgrid(np.linspace(0,crop_size[0]-1,orig_res[0]), np.linspace(0,crop_size[1]-1,orig_res[1]))
    xy_img = np.vstack((x.reshape(1,orig_res[0],orig_res[1]), y.reshape(1,orig_res[0],orig_res[1])))
    points2D = xy_img.reshape(2, -1)
    points2D = np.transpose(points2D) # Npoints x 2
    return points2D, local3D


def build_p_gt(par, pose_gt_batch):
    batch_size = pose_gt_batch.shape[0]
    seq_len = pose_gt_batch.shape[1]
    p_gt = np.zeros((batch_size, seq_len, par.orientations, par.global_map_dim[0], par.global_map_dim[1]), dtype=np.float32) 
    for b in range(batch_size):
        pose_seq = pose_gt_batch[b,:,:] # seq_len x 3
        #print('pose_seq = {}'.format(pose_seq))
        #assert 1==2
        map_coords_gt, valid = discretize_coords(x=pose_seq[:,0], z=pose_seq[:,1], map_dim=par.global_map_dim, cell_size=par.cell_size)
        # use the gt direction to get the orientation index 
        #print(map_coords_gt)
        dir_ind = np.floor( np.mod(pose_seq[:,2]/(2*np.pi), 1) * par.orientations ) 
        #print('Angle:', np.degrees(pose_seq[:,2]), ' Ind:', dir_ind)
        # the indices not included in valid are those outside the map, p_gt is all zeroes in that case
        map_coords_gt = map_coords_gt[valid,:]
        dir_ind = dir_ind[valid]
        # numerical issue: when number is very close to 0 (i.e. -6e-6) dir_ind is given as 12
        dir_ind = np.where(dir_ind == par.orientations, 0, dir_ind)
        p_gt[b, valid, dir_ind.astype(int), map_coords_gt[:,1], map_coords_gt[:,0]] = 1
        #plt.imshow(p_gt[b,2,:,:,1])
        #plt.show()
    return torch.from_numpy(p_gt).cuda().float()  


def discretize_coords(x, z, map_dim, cell_size):
    # x, z are the coordinates of the 3D point (either in camera coordinate frame, or the ground-truth camera position)
    #map_coords = np.zeros((local3D.shape[0], 2), dtype=np.int)
    map_coords = np.zeros((len(x), 2), dtype=np.int)
    #map_occ = np.zeros((1, map_dim[0], map_dim[1]), dtype=np.float32)
    #print(local3D.shape)
    #x, z = local3D[:,0], local3D[:,2]
    #print(x.shape, z.shape)
    xb = np.floor(x[:]/cell_size) + (map_dim[0]-1)/2.0
    zb = np.floor(z[:]/cell_size) + (map_dim[1]-1)/2.0
    xb = xb.astype(int)
    zb = zb.astype(int)
    #zb = (map_dim[1]-1)-zb
    map_coords[:,0] = xb
    map_coords[:,1] = zb
    # keep bin coords within dimensions
    inds_1 = np.where(xb>=0)
    inds_2 = np.where(zb>=0)
    inds_3 = np.where(xb<map_dim[0])
    inds_4 = np.where(zb<map_dim[1])
    valid = reduce(np.intersect1d, (inds_1, inds_2, inds_3, inds_4))
    #print(valid.shape)
    xb = xb[valid]
    zb = zb[valid]
    #map_occ[0,zb,xb] = 1
    #plt.imshow(map_occ[0,:,:])
    #plt.show()
    #raise Exception("444")
    return map_coords, valid #, map_occ

'''
def convertToSSeg(rgb_img, cropSize):
    rgb_img = Image.fromarray(rgb_img)
    o_t = TF.to_tensor(TF.resize(rgb_img, cropSize[0])) * 2 - 1
    o_t = o_t.unsqueeze(0)
    pred = visualpriors.feature_readout(o_t, 'segment_semantic', device='cpu')[0].numpy()
    result = np.expand_dims(np.argmax(pred, axis=0), axis=0).astype('float32')
    return result


def getImageData(observations, dets_nClasses, cropSize, orig_res, pixFormat, normalize, get3d=True):
    rgb_img = np.copy(observations['rgb'])
    rgb_img = cv2.resize(rgb_img, (cropSize[0], cropSize[1]))
    imgData = convert_image_by_pixformat_normalize(rgb_img, pixFormat, normalize)

    if get3d:
        #sseg_img = np.zeros((1, cropSize[0], cropSize[1]), dtype=np.float32)
        sseg_img = convertToSSeg(rgb_img, cropSize)
        det_img = np.zeros((dets_nClasses, cropSize[0], cropSize[1]), dtype=np.float32)

        #get 2d and 3d points
        depth_img = np.copy(observations['depth'])
        depth_img = cv2.resize(depth_img, (cropSize[0], cropSize[1]))
        #print('depth_img = {}'.format(depth_img))
        #assert 1==2
        points2D, local3D = depth_to_3D(depth_img, orig_res, cropSize)

        return imgData, sseg_img, det_img, points2D, local3D
    else:
        det_img = np.zeros((dets_nClasses, cropSize[0], cropSize[1]), dtype=np.float32)
        return imgData, det_img
'''

color_mapping = {
    0:(255,255,255), # white
    1:(210,105,30), # chocolate
    2:(255,0,0), # red
    3:(0,255,0), # green
    4:(0,0,255), # blue
    5:(255,255,0), # yellow
    6:(0,255,255), # cyan
    7:(255,0,255), # magenta
    8:(128,128,128), # gray
    9:(128,0,0), # maroon
    10:(128,128,0), # olive (dark yellow)
    11:(0,128,0), # dark green
    12:(128,0,128), # purple
    13:(0,128,128), # teal
    14:(0,0,128), # navy (dark blue)
    15:(255,165,0), # orange
    16:(255,20,147) # pink
}

def convert_semantic_output(pred):
    pred = np.squeeze(pred)
    pred = np.transpose(pred, (1,2,0))
    #print(pred.shape)
    x = np.zeros((pred.shape[0],pred.shape[1],3), dtype='float')
    #k_embed = 8
    #embedding_flattened = pred.reshape((-1,64)) # for segment_unsup2d, segment_unsup25d
    embedding_flattened = pred.reshape((-1,17)) # for segment_semantic
    pca = PCA(n_components=3)
    pca.fit(np.vstack(embedding_flattened))
    lower_dim = pca.transform(embedding_flattened).reshape((pred.shape[0],pred.shape[1],-1))
    x = (lower_dim-lower_dim.min()) / (lower_dim.max()-lower_dim.min())
    #print(x.shape)
    return x


def colorize_sseg(sseg):
    sseg = sseg.squeeze(0)
    sseg_img = np.zeros((sseg.shape[0], sseg.shape[1], 3), dtype=np.uint8)
    for label in color_mapping.keys():
        sseg_img[ sseg == label ] = color_mapping[label]
    #print(sseg_img)
    return sseg_img


def display_sample(rgb_obs, depth_obs, sseg_img, sseg_pred, savepath=None):
    depth_obs = depth_obs / np.amax(depth_obs) # normalize for visualization
    rgb_img = Image.fromarray(rgb_obs, mode="RGB")
    depth_img = Image.fromarray((depth_obs * 255).astype(np.uint8), mode="L")
    arr = [rgb_img, depth_img, sseg_img, sseg_pred]
    titles = ['rgb', 'depth', 'sseg', 'sseg_pred']

    plt.figure(figsize=(12 ,8))
    for i, data in enumerate(arr):
        ax = plt.subplot(1, 4, i+1)
        ax.axis('off')
        ax.set_title(titles[i])
        plt.imshow(data)
    if savepath is None:
        plt.show()
    else:
        plt.savefig(savepath, bbox_inches='tight', pad_inches=0, dpi=100)
    plt.close()


def display_sample2(rgb_obs, depth_obs, semantic_obs, savepath=None):
    depth_obs = depth_obs / np.amax(depth_obs) # normalize for visualization
    rgb_img = Image.fromarray(rgb_obs, mode="RGB")
    depth_img = Image.fromarray((depth_obs * 255).astype(np.uint8), mode="L")

    semantic_img = Image.new("P", (semantic_obs.shape[1], semantic_obs.shape[0]))
    semantic_img.putpalette(d3_40_colors_rgb.flatten())
    semantic_img.putdata((semantic_obs.flatten() % 40).astype(np.uint8))
    semantic_img = semantic_img.convert("RGBA")

    arr = [rgb_img, depth_img, semantic_img]
    titles = ['rgb', 'depth', 'sseg']

    plt.figure(figsize=(12 ,8))
    for i, data in enumerate(arr):
        ax = plt.subplot(1, 3, i+1)
        ax.axis('off')
        ax.set_title(titles[i])
        plt.imshow(data)
    if savepath is None:
        plt.show()
    else:
        plt.savefig(savepath, bbox_inches='tight', pad_inches=0, dpi=100)
    plt.close()


def get_sseg(img):
    img = img[:,:,::-1]
    img_norm = (img/256) * 2 - 1 # normalization required by visualpriors
    img_norm = img_norm.transpose(2,0,1)
    img_norm = torch.from_numpy(img_norm).float().cuda()
    img_norm = img_norm.unsqueeze(0)
    representation = visualpriors.representation_transform(img_norm, 'segment_semantic')
    pred = visualpriors.feature_readout(img_norm, 'segment_semantic') # 1 x 17 x H x W
    #pred = torch.nn.functional.softmax(input=pred, dim=1)
    #print(pred.shape)
    pred = pred.cpu().numpy()
    sseg = np.argmax(pred, axis=1) # 1 x H x W
    return sseg, pred


def preprocess_depth(depth, cropSize):
    depth = cv2.resize(depth, (cropSize[0], cropSize[1]))
    depth = depth.astype(np.float32) / 10.0 # normalize, assume maximum depth 10m
    return np.expand_dims(depth, 0)


def preprocess_img(img, cropSize, pixFormat, normalize):
    img = cv2.resize(img, (cropSize[0], cropSize[1]))
    if pixFormat == 'NCHW':
        img = img.transpose((2, 0, 1))
    if normalize:
        img = img.astype(np.float32) / 255.0 #* 2.0 - 1.0
    return img


def get_sim_location(env):
    agent_state = env._sim.get_agent_state(0)
    x = -agent_state.position[2]
    y = -agent_state.position[0]
    axis = quaternion.as_euler_angles(agent_state.rotation)[0]
    if (axis%(2*np.pi)) < 0.1 or (axis%(2*np.pi)) > 2*np.pi - 0.1:
        o = quaternion.as_euler_angles(agent_state.rotation)[1]
    else:
        o = 2*np.pi - quaternion.as_euler_angles(agent_state.rotation)[1]
    if o > np.pi:
        o -= 2 * np.pi
    return x, y, o


def get_rel_pose(pos2, pos1):
    x1, y1, o1 = pos1
    x2, y2, o2 = pos2
    dx = x2 - x1
    dy = y2 - y1
    do = o2 - o1
    if do < -math.pi:
        do += 2 * math.pi
    if do > math.pi:
        do -= 2 * math.pi
    # we also need to transform the pos2 coords with respect to pos1 orientation
    pos1_rot_mat = np.array([[np.cos(o1), -np.sin(o1)],[np.sin(o1),np.cos(o1)]])
    rel_coord = np.array([dy, dx])
    rel_coord = rel_coord.reshape((2,1))
    rel_coord = np.matmul(pos1_rot_mat,rel_coord).squeeze(1)
    #return dx, dy, do
    return rel_coord[0], rel_coord[1], do

'''
def relative_poses(poses):
    #print(poses)
    # poses (seq_len x 3) contains the ground-truth camera positions and orientation in the sequence
    # make them relative to the first pose in the sequence
    seq_len, _ = poses.shape
    rel_poses = np.zeros((poses.shape[0], poses.shape[1]), dtype=np.float32)
    x0 = poses[0,0]
    y0 = poses[0,1]
    a0 = poses[0,2]
    # relative translation
    rel_poses[:,0] = poses[:,0] - x0
    rel_poses[:,1] = poses[:,1] - y0
    #print(rel_poses)
    rel_poses[:,2] = poses[:,2] - a0
    # convert the angles to [-pi, pi]
    rows_smaller_pi = rel_poses[:, 2] < -math.pi
    rel_poses[rows_smaller_pi, 2] = rel_poses[rows_smaller_pi, 2] + 2 * math.pi
    rows_larger_pi = rel_poses[:, 2] > math.pi
    rel_poses[rows_larger_pi, 2] = rel_poses[rows_larger_pi, 2] - 2 * math.pi
    rel_poses[:, 2] = np.round(rel_poses[:, 2], 4)
    #print('poses = {}'.format(poses))
    #print('rel_poses = {}'.format(rel_poses))
    return rel_poses
'''

# configAction is Habitat env's action. paramAction is the action in parameter.py files.
# In habitat env, 1 is forward, 2 is left, 3 is right
# in parameter.py, 2 is forward, 0 is left, 1 is right
def configAction_to_paramAction(config_action):
	if config_action == 1: 
		# move forward
		return 2
	elif config_action == 2: 
		# turn left
		return 0
	else:
		return 1

def paramAction_to_configAction(param_action):
    if param_action == 'forward': 
        # move forward
        return 1
    elif param_action == 'rotate_ccw': 
        # turn left
        return 2
    else:
        return 3

def configAction_to_costOfParamAction(config_action):
	if config_action == 1: 
		# move forward
		return [1, 1, -1]
	elif config_action == 2: 
		# turn left
		return [-1, 1, 1]
	else:
		return [1, -1, 1]