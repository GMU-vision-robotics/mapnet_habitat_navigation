import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
import os
import numpy as np
import data_helper_habitat as dhh

class MapNet(nn.Module): 
    # Implementation of MapNet and all its core components following the paper:
    # Henriques and Vedaldi, MapNet: An Allocentric Spatial Memory for Mapping Environments, CVPR 2018
    def __init__(self, par, update_type, input_flags):
        super(MapNet, self).__init__()
        (with_feat, with_sseg, with_dets, use_raw_sseg, use_raw_dets, with_depth) = input_flags
        self.crop_size = par.crop_size
        self.global_map_dim = par.global_map_dim
        self.observation_dim  = par.observation_dim
        self.cell_size = par.cell_size
        self.grid_channels = par.grid_channels
        self.map_embedding = par.map_embedding
        self.sseg_labels = par.sseg_labels
        self.dets_nClasses = par.dets_nClasses
        self.orientations = par.orientations
        self.pad = par.pad
        self.loss_type = par.loss_type
        self.update_type = par.update_type

        if with_feat:
            self.resnet_feat_dim = 256 #512
            fnet = models.resnet50(pretrained=True)
            self.ResNet50Truncated = nn.Sequential(*list(fnet.children())[:-5]) # -4 up to layer2 (conv3_x), -5 up to layer1 (conv2_x)
            # Define the small network that outputs the final grid with the embedding
            self.small_cnn_img = nn.Sequential(
                nn.Conv2d(in_channels=self.resnet_feat_dim, out_channels=64, kernel_size=3, stride=1, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_channels=64, out_channels=par.img_embedding, kernel_size=3, stride=1, padding=1),
                nn.ReLU(inplace=True)
            )
        if with_depth:
            # depth feature extractor
            self.resnet_feat_dim = 256 #512
            fnet_depth = models.resnet50(pretrained=True)
            fnet_depth.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3,bias=False)
            self.ResNet50Truncated_depth = nn.Sequential(*list(fnet_depth.children())[:-5])
            self.small_cnn_depth = nn.Sequential(
                nn.Conv2d(in_channels=self.resnet_feat_dim, out_channels=64, kernel_size=3, stride=1, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_channels=64, out_channels=par.depth_embedding, kernel_size=3, stride=1, padding=1),
                nn.ReLU(inplace=True)
            )

        if with_sseg and not(use_raw_sseg):
            # Define the small network that outputs the semantic label grid
            self.small_cnn_sseg = nn.Sequential( # 40 classes
                nn.Conv2d(in_channels=self.sseg_labels, out_channels=64, kernel_size=3, stride=1, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_channels=64, out_channels=par.sseg_embedding, kernel_size=3, stride=1, padding=1),
                nn.ReLU(inplace=True)
            )
        '''    
        if with_dets and not(use_raw_dets):
            # Define the small network that outputs the detection grid embedding
            self.small_cnn_det = nn.Sequential( # 91 classes
                nn.Conv2d(in_channels=self.dets_nClasses, out_channels=64, kernel_size=3, stride=1, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_channels=64, out_channels=par.dets_embedding, kernel_size=3, stride=1, padding=1),
                nn.ReLU(inplace=True)
            )
        '''

        # After the grids are concatenated to a single grid, pass it through a couple of convolutions to extract common features between the embeddings  
        '''
        self.embedding_net = nn.Sequential(
                nn.Conv2d(in_channels=self.grid_channels, out_channels=64, kernel_size=3, stride=1, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_channels=64, out_channels=self.map_embedding, kernel_size=3, stride=1, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_channels=self.map_embedding, out_channels=self.map_embedding, kernel_size=3, stride=1, padding=1),
                nn.ReLU(inplace=True)
            )
        '''

        # Choose how to update the map
        if update_type=="lstm":
            # LSTM to update the map with the current observation
            self.lstm = nn.LSTM(input_size=self.map_embedding, hidden_size=self.map_embedding, num_layers=1)
        elif update_type=="fc":
            # Use a fully connected layer to update the embedding at every grid by combining the current
            # embedding and the previous embedding at every grid location
            self.update_fc = nn.Linear(self.map_embedding*2, self.map_embedding)
        else: # case 'avg'
            # Do average pooling over the embeddings
            self.update_avg = nn.AvgPool1d(kernel_size=2)

        # Choose loss
        if self.loss_type=="BCE":
            self.loss_BCE = nn.BCELoss(reduction="sum") # Binary cross entropy loss
        else:
            self.loss_CEL = nn.CrossEntropyLoss(ignore_index=-1)


    def build_loss(self, p_pred, p_gt):
        batch_size = p_pred.shape[0]
        seq_len = p_pred.shape[1]
        # Remove the first frame in the sequence, since it is always constant
        p_pred = p_pred[:,1:,:,:,:]
        p_gt = p_gt[:,1:,:,:,:]
        if self.loss_type=="BCE":
            p_pred = p_pred.contiguous().view(batch_size*(seq_len-1), self.orientations, self.global_map_dim[0], self.global_map_dim[1])
            p_gt = p_gt.contiguous().view(batch_size*(seq_len-1), self.orientations, self.global_map_dim[0], self.global_map_dim[1])
            loss = self.loss_BCE(p_pred, p_gt)
            loss /= p_pred.shape[0] # assuming reduction="sum"
        else:
            # Both p_pred and p_gt are b x q x h x w x r
            # Need to convert p_pred to N x C, p_gt to N
            p_pred = p_pred.contiguous().view(batch_size*(seq_len-1), self.orientations*self.global_map_dim[0]*self.global_map_dim[1]) # N x C
            p_gt = p_gt.contiguous().view(batch_size*(seq_len-1), self.orientations*self.global_map_dim[0]*self.global_map_dim[1])
            # For each example, get the index of p_gt for which it is 1. It should be unique.
            # When the gt is outside of the map, the p_gt example is all zeroes
            lbls = torch.zeros(p_gt.shape[0]).long().cuda()
            for i in range(p_gt.shape[0]):
                ind = torch.nonzero(p_gt[i,:], as_tuple=False) # label that signifies the ground-truth position and orienation
                if (ind.nelement()==0): # if ind is empty (p_gt is empty), then ignore this example's index in t
                    lbls[i] = -1
                else:
                    lbls[i] = ind[0][0]
            loss = self.loss_CEL(p_pred, lbls) # this does the log_softmax inside the loss
        return loss


    def extract_img_feat(self, img_data):
        img_feat = self.ResNet50Truncated(img_data)
        # Resize the features to the image/depth resolution
        img_feat = F.interpolate(img_feat, size=(self.crop_size[1], self.crop_size[0]), mode='nearest')           
        return img_feat

    def extract_depth_feat(self, depth_data):
        depth_feat = self.ResNet50Truncated_depth(depth_data)
        depth_feat = F.interpolate(depth_feat, size=(self.crop_size[1], self.crop_size[0]), mode='nearest')
        return depth_feat


    def init_p(self, batch_size):
        # Initialize the position (p0) in the center of the map at angle 0 (i.e. orientation index 0).
        p0 = np.zeros((batch_size, self.orientations, self.global_map_dim[0], self.global_map_dim[1]), dtype=np.float32)
        p0[:, 0, int(self.global_map_dim[0]/2.0), int(self.global_map_dim[1]/2.0)] = 1 # 14,14
        return torch.tensor(p0, dtype=torch.float32).cuda()


    def forward_single_step(self, local_info, t, input_flags, map_previous=None, p_given=None, update_type="lstm"):
        # Forward pass of mapNet when the episode is not known before hand (i.e. online training)
        # Runs MapNet for a single step
        (img_data, points2D, local3D, sseg, depth_data) = local_info
        batch_size = img_data.shape[0]

        with_feat, with_depth = input_flags[0], input_flags[5]
        if with_feat:
            img_feat = self.extract_img_feat(img_data)
        else:
            # placeholder, not used later
            img_feat = torch.zeros(batch_size, 1, self.crop_size[1], self.crop_size[0])
        if with_depth:
            depth_feat = self.extract_depth_feat(depth_data)
        else:
            depth_feat = torch.zeros(batch_size, 1, self.crop_size[1], self.crop_size[0])

        # Follow the groundProjection() but now do only for batch_size
        grid = torch.zeros((batch_size, self.grid_channels, self.observation_dim[0], self.observation_dim[1]), dtype=torch.float32).cuda()
        for b in range(batch_size):
            points2D_step = points2D[b]
            local3D_step = local3D[b]
            img_feat_step = img_feat[b,:,:,:].unsqueeze(0)
            sseg_step = sseg[b,:,:,:].unsqueeze(0)
            depth_feat_step = depth_feat[b,:,:,:].unsqueeze(0)
            grid_step = self.groundProjectionStep(img_feat=img_feat_step, points2D=points2D_step, 
                                                local3D=local3D_step, sseg=sseg_step, depth_feat=depth_feat_step, input_flags=input_flags)
            grid[b,:,:,:] = grid_step

        #grid = self.embedding_net(grid)

        rotation_stack = self.rotational_sampler(grid=grid)
        if t==0:
            # Case of first step
            p_ = self.init_p(batch_size=batch_size).clone()
            map_next = self.register_observation(rotation_stack=rotation_stack, p=p_, batch_size=batch_size)
        else:
            if p_given is None:
                p_ = self.position_prediction(rotation_stack=rotation_stack, map_previous=map_previous, batch_size=batch_size)
            else:
                p_ = p_given
            reg_obsv = self.register_observation(rotation_stack=rotation_stack, p=p_, batch_size=batch_size)
            map_next = self.update_map(reg_obsv, map_previous, batch_size=batch_size, update_type=update_type)
        return p_, map_next
        

    def forward(self, local_info, update_type, input_flags, p_gt=None):
        # Runs MapNet for an entire episode
        (img_data, points2D, local3D, sseg, depth_data) = local_info
        batch_size = img_data.shape[0]
        seq_len = img_data.shape[1]

        # Extract the img features
        with_feat, with_depth = input_flags[0], input_flags[5]
        if with_feat:
            img_data = img_data.view(batch_size*seq_len, 3, self.crop_size[1], self.crop_size[0])
            img_feat = self.extract_img_feat(img_data)
            img_feat = img_feat.view(batch_size, seq_len, self.resnet_feat_dim, self.crop_size[1], self.crop_size[0])
        else:
            # placeholder, not used later
            img_feat = torch.zeros(batch_size, seq_len, 1, self.crop_size[1], self.crop_size[0]) 

        if with_depth:
            # depth features
            depth_data = depth_data.view(batch_size*seq_len, 1, self.crop_size[1], self.crop_size[0])
            depth_feat = self.extract_depth_feat(depth_data)
            depth_feat = depth_feat.view(batch_size, seq_len, self.resnet_feat_dim, self.crop_size[1], self.crop_size[0])
        else:
            depth_feat = torch.zeros(batch_size, seq_len, 1, self.crop_size[1], self.crop_size[0]) 


        # Project the img features on a ground grid
        grid = self.groundProjection(img_feat_all=img_feat, points2D_all=points2D, local3D_all=local3D, 
                                        sseg_all=sseg, depths_all=depth_feat, batch_size=batch_size, seq_len=seq_len, input_flags=input_flags)

        # Pass the combined grid through a small network to extract common embedding features
        #grid = self.embedding_net(grid.view(batch_size*seq_len, self.grid_channels, self.observation_dim[0], self.observation_dim[1]))
        #grid = grid.view(batch_size, seq_len, self.map_embedding, self.observation_dim[0], self.observation_dim[1])

        # Rotate the grid's feature channels to obtain the rotational stack
        grid_packed = grid.view(batch_size*seq_len, self.map_embedding, self.observation_dim[0], self.observation_dim[1])
        rotation_stack_packed = self.rotational_sampler(grid=grid_packed)
        rotation_stack = rotation_stack_packed.view(batch_size, seq_len, self.map_embedding, self.orientations, self.observation_dim[0], self.observation_dim[1])
        
        # The next steps need to be carried out in sequence as p_ depends on each step's previous map.
        p_pred = torch.zeros((batch_size, seq_len, self.orientations, self.global_map_dim[0], self.global_map_dim[1]), dtype=torch.float32).cuda()
        #p_pred = torch.tensor(p_pred, dtype=torch.float32).cuda()
        map_pred = torch.zeros((batch_size, seq_len, self.map_embedding, self.global_map_dim[0], self.global_map_dim[1]), dtype=torch.float32).cuda()
        #map_pred = torch.tensor(map_pred, dtype=torch.float32).cuda()
        for q in range(seq_len):
            rotation_stack_step = rotation_stack[:,q,:,:,:,:]
            if q==0:
                # For first observation we assume p_=p0 and we just need to register the observation.
                # In this case, the registered observation is actually the first map so we do not 
                # need to update it (so no need for the LSTM).
                p_ = self.init_p(batch_size=batch_size).clone() #self.p0.clone()
                map_next = self.register_observation(rotation_stack=rotation_stack_step, p=p_, batch_size=batch_size)
            else:
                # Here we need to predict p, register the obsv based on p, and update the map
                map_previous = map_next.clone()
                # Do the cross correlation with the existing map and pass through a softmax
                p_ = self.position_prediction(rotation_stack=rotation_stack_step, map_previous=map_previous, batch_size=batch_size)
                # Convolve the localization prediction (Pt) with the rotational stack using a deconvolution
                # in order to register the observations in the map
                if p_gt is not None: 
                    # if p_gt is given, then register the observation with it.
                    # We still use p_ for the loss, but we use the ground-truth to register the map
                    reg_obsv = self.register_observation(rotation_stack=rotation_stack_step, p=p_gt[:,q,:,:,:], batch_size=batch_size)
                else:
                    reg_obsv = self.register_observation(rotation_stack=rotation_stack_step, p=p_, batch_size=batch_size)

                # Update the map using LSTM - hidden state: map_previous, input: reg_obsv
                # Each spatial location is passed independently in the LSTM
                map_next = self.update_map(reg_obsv, map_previous, batch_size=batch_size, update_type=update_type)

            # Store the p_ predictions and the map for each timestep
            p_pred[:,q,:,:,:] = p_
            map_pred[:,q,:,:,:] = map_next

        return p_pred, map_pred



    def groundProjection(self, img_feat_all, points2D_all, local3D_all, sseg_all, depths_all, batch_size, seq_len, input_flags):
        # A wrapper over groundProjectionStep to do the projection for batch_size x seq_len
        grid = np.zeros((batch_size, seq_len, self.grid_channels, self.observation_dim[0], self.observation_dim[1]), dtype=np.float32)
        grid = torch.tensor(grid, dtype=torch.float32).cuda()
        #map_occ = np.zeros((batch_size, seq_len, 1, self.observation_dim[0], self.observation_dim[1]), dtype=np.float32)
        #map_occ = torch.tensor(map_occ, dtype=torch.float32).cuda()
        for b in range(batch_size):
            points2D_seq = points2D_all[b]
            local3D_seq = local3D_all[b]
            for q in range(seq_len):
                points2D_step = points2D_seq[q] # n_points x 2
                local3D_step = local3D_seq[q] # n_points x 3
                img_feat_step = img_feat_all[b,q,:,:,:].unsqueeze(0) # 1 x resNet_feat_dim x crop_size(1) x crop_size(0)
                sseg_step = sseg_all[b,q,:,:,:].unsqueeze(0) # 1 x 1 x crop_size(1) x crop_size(0)
                depths_step = depths_all[b,q,:,:,:].unsqueeze(0) # 1 x 1 x crop_size(1) x crop_size(0)
                grid_step = self.groundProjectionStep(img_feat=img_feat_step, points2D=points2D_step, 
                                                        local3D=local3D_step, sseg=sseg_step, depth_feat=depths_step, input_flags=input_flags)
                grid[b,q,:,:,:] = grid_step.squeeze(0)
                #map_occ[b,q,:,:,:] = map_occ_step.squeeze(0)
        return grid #, map_occ


    def bin_pooling(self, img_feat, points2D, map_coords):
        # Bin pooling during ground projection of the features
        grid = np.zeros((img_feat.shape[1], self.observation_dim[0], self.observation_dim[1]), dtype=np.float32)
        grid = torch.tensor(grid, dtype=torch.float32).cuda()        
        pix_x, pix_y = points2D[:,0], points2D[:,1]
        pix_feat = img_feat[0, :, pix_y, pix_x]
        uniq_rows = np.unique(map_coords, axis=0)
        for i in range(uniq_rows.shape[0]):
            ucoord = uniq_rows[i,:]
            ind = np.where( (map_coords==ucoord).all(axis=1) )[0] # indices of where ucoord can be found in map_coords
            # Features indices in the ind array belong to the same bin and have to be max-pooled
            bin_feats = pix_feat[:,ind] # [d x n] n:number of feature vectors projected, d:feat_dim
            bin_feat, _ = torch.max(bin_feats,1) # [d]
            grid[:, ucoord[1], ucoord[0]] = bin_feat
        return grid


    def label_pooling(self, sseg, points2D, map_coords):
        # Similar to bin_pooling() but instead features we pool the semantic labels
        # For each bin get the frequencies of the class labels based on the labels projected
        # Each grid location will hold a probability distribution over the semantic labels
        grid = np.zeros((self.sseg_labels, self.observation_dim[0], self.observation_dim[1]), dtype=np.float32)
        pix_x, pix_y = points2D[:,0], points2D[:,1]
        sseg = sseg.cpu()
        pix_lbl = sseg[0, 0, pix_y, pix_x]
        uniq_rows = np.unique(map_coords, axis=0)
        for i in range(uniq_rows.shape[0]):
            ucoord = uniq_rows[i,:]
            ind = np.where( (map_coords==ucoord).all(axis=1) )[0] # indices of where ucoord can be found in map_coords
            bin_lbls = pix_lbl[ind]
            # Labels are from 0-39 where 0:wall ... 39:other_prop
            hist, bins = np.histogram(bin_lbls, bins=list(range(self.sseg_labels+1)))
            hist = hist / float(bin_lbls.shape[0])
            grid[:, ucoord[1], ucoord[0]] = hist
        grid = torch.tensor(grid, dtype=torch.float32).cuda()
        return grid

    '''
    def dets_pooling(self, dets, points2D, map_coords):
        # Bin pooling of the detection masks. Detection scores in the same bin are averaged.
        grid = np.zeros((self.dets_nClasses, self.observation_dim[0], self.observation_dim[1]), dtype=np.float32)
        pix_x, pix_y = points2D[:,0], points2D[:,1]
        # pix_dets holds a vector of binary values which indicate the presence of each category
        # multiple values can be 1 due to overlapping bounding boxes
        # pix_dets that end up in the same grid location are averaged
        pix_dets = dets[0,:,pix_y, pix_x]
        uniq_rows = np.unique(map_coords, axis=0)
        for i in range(uniq_rows.shape[0]):
            ucoord = uniq_rows[i,:]
            ind = np.where( (map_coords==ucoord).all(axis=1) )[0] # indices of where ucoord can be found in map_coords
            bin_dets = pix_dets[:, ind] # all detections in the same bin
            bin_det = bin_dets.mean(1)
            grid[:, ucoord[1], ucoord[0]] = bin_det
        grid = torch.tensor(grid, dtype=torch.float32).cuda()
        return grid
    '''

    # Performs the ground projection for a single image
    def groundProjectionStep(self, img_feat, points2D, local3D, sseg, depth_feat, input_flags):
        (with_feat, with_sseg, with_dets, use_raw_sseg, use_raw_dets, with_depth) = input_flags
        # Create the grid and discretize the set of coordinates into the bins
        # Points2D holds the image pixel coordinates with valid depth values
        # Local3D holds the X,Y,Z coordinates that correspond to the points2D
        # For each local3d find which bin it belongs to
        map_coords, valid = dhh.discretize_coords(x=local3D[:,0], z=local3D[:,2], map_dim=self.observation_dim, cell_size=self.cell_size)
        #map_occ = torch.tensor(map_occ, dtype=torch.float32).cuda()

        points2D = points2D[valid,:]
        map_coords = map_coords[valid,:]
        grids = []
        if with_feat:
            # Max-pool the features for each bin and extract the img embedding to get the img grid 
            grid_img = self.bin_pooling(img_feat, points2D, map_coords)
            # Pass the grid to a CNN to get the observation with 32-D embeddings
            grid_img_in = grid_img.unsqueeze(0)
            grids.append(self.small_cnn_img(grid_img_in.cuda())) # 1 x 32 x 21 x 21
        if with_depth:
            # do the depth grid
            grid_depth = self.bin_pooling(depth_feat, points2D, map_coords)
            grid_depth_in = grid_depth.unsqueeze(0)
            grids.append(self.small_cnn_depth(grid_depth_in.cuda()))
        if with_sseg:
            # Get the probabilities of the labels on the grid and extract an embedding
            grid_sseg = self.label_pooling(sseg, points2D, map_coords)
            grid_sseg_in = grid_sseg.unsqueeze(0)
            if use_raw_sseg:
                grids.append(grid_sseg_in)
            else:
                grids.append(self.small_cnn_sseg(grid_sseg_in))
        '''
        if with_dets:
            grid_det = self.dets_pooling(dets, points2D, map_coords)
            grid_det_in = grid_det.unsqueeze(0)
            if use_raw_dets:
                grids.append(grid_det_in)
            else:
                grids.append(self.small_cnn_det(grid_det_in))
        '''
        if len(grids) == 0:
            raise Exception("No input grids!")
        # Stack the grids 
        grid_out = torch.cat(grids, 1).cuda()
        return grid_out #, map_occ


    def rotational_sampler(self, grid, rot_init=True):
        # The grid (after the groundProjection) is facing up
        # We need to rotate it so as to face to the right (angle 0)
        if rot_init:
            grid = self.do_rotation(grid, angle=-np.pi/2.0)
        # Rotate the grid's feature channels to obtain the rotational stack
        rotation_stack = np.zeros((grid.shape[0], self.map_embedding, self.orientations, self.observation_dim[0], self.observation_dim[1]), dtype=np.float32 )
        rotation_stack = torch.tensor(rotation_stack, dtype=torch.float32).cuda()
        for i in range(self.orientations):
            angle = 2*np.pi*(i/self.orientations)
            rotation_stack[:,:,i,:,:] = self.do_rotation(grid, angle) #grid_trans
        return rotation_stack


    def do_rotation(self, grid, angle):
        # apply a single rotation on a grid (N x n x s x s)
        theta = torch.tensor( [ [np.cos(angle), -1.0*np.sin(angle), 0], [np.sin(angle), np.cos(angle), 0] ] ).cuda() # 2 x 3 affine transform
        theta = theta.unsqueeze(0).repeat(grid.shape[0], 1, 1)
        grid_affine = F.affine_grid(theta, grid.size(), align_corners=True)
        return F.grid_sample(grid, grid_affine.float(), align_corners=True)


    def position_prediction(self, rotation_stack, map_previous, batch_size):
        # Do the cross correlation with the existing map and pass through a softmax
        corr_map = np.zeros((batch_size, self.orientations, self.global_map_dim[0], self.global_map_dim[1]), dtype=np.float32)
        corr_map = torch.tensor(corr_map, dtype=torch.float32).cuda()
        for b in range(batch_size):
            map_ = map_previous[b,:,:,:].unsqueeze(0) # 1 x n x h x w
            for r in range(self.orientations):
                # convolve each filter with the previous map
                filt = rotation_stack[b,:,r,:,:].unsqueeze(0) # 1 x n x s x s
                corr_tmp = F.conv2d(input=map_, weight=filt, padding=self.pad)
                corr_map[b,r,:,:] = corr_tmp
        p_ = np.zeros((batch_size, self.orientations, self.global_map_dim[0], self.global_map_dim[1]), dtype=np.float32)
        p_ = torch.tensor(p_, dtype=torch.float32).cuda()
        for i in range(batch_size):
            p_tmp = corr_map[i,:,:,:].view(-1) # # if we do the CEL loss then we do not use softmax (it is included in the loss layer)
            p_tmp = p_tmp.view(self.orientations, self.global_map_dim[0], self.global_map_dim[1]) # reshape the tensor back to map
            p_[i,:,:,:] = p_tmp
        return p_


    def register_observation(self, rotation_stack, p, batch_size):
        reg_obsv = np.zeros((batch_size, self.map_embedding, self.global_map_dim[0], self.global_map_dim[1]), dtype=np.float32)
        reg_obsv = torch.tensor(reg_obsv, dtype=torch.float32).cuda()
        for i in range(batch_size):
            filt = rotation_stack[i,:,:,:,:] # n x r x s x s
            filt = filt.permute(1,0,2,3) #permute(3,0,1,2) # r x n x s x s
            p_in = p[i,:,:,:].unsqueeze(0) # 1 x r x h x w # take one example from the batch
            reg = F.conv_transpose2d(input=p_in, weight=filt, padding=self.pad) # 1 x n x h x w
            reg_obsv[i,:,:,:] = reg
        return reg_obsv


    def update_map(self, reg_obsv, map_previous, batch_size, update_type):
        # Update the map using LSTM - hidden state: map_previous, input: reg_obsv
        # Each spatial location is passed independently in the LSTM
        if update_type=="lstm":
            map_next = np.zeros((batch_size, self.map_embedding, self.global_map_dim[0], self.global_map_dim[1]), dtype=np.float32)
            map_next = torch.tensor(map_next, dtype=torch.float32).cuda()
            # ** LSTM input requires sequence length dimension, in our case is 1
            for i in range(self.global_map_dim[0]):
                for j in range(self.global_map_dim[1]):
                    emb_in = reg_obsv[:,:,i,j] # b x n
                    emb_hidden = map_previous[:,:,i,j] # b x n
                    emb_in = emb_in.unsqueeze(0) # add the dimension for the sequence length
                    emb_hidden = emb_hidden.unsqueeze(0) # add the dimension for the nLstm layers
                    hidden = (emb_hidden.contiguous().cuda(), emb_hidden.contiguous().cuda()) # LSTM expects two hidden inputs
                    lstm_out, hidden_out = self.lstm(emb_in, hidden)
                    map_next[:,:,i,j] = lstm_out
        elif update_type=="fc": # using a fully connected layer
            map2 = torch.cat((map_previous, reg_obsv), 1)
            map_next = self.update_fc(map2.permute(0,2,3,1))
            map_next = torch.tanh(map_next)
            map_next = map_next.permute(0,3,1,2) # b x n x h x w
        else: # case 'avg', using AvgPool1d layer
            map_next = np.zeros((batch_size, self.map_embedding, self.global_map_dim[0], self.global_map_dim[1]), dtype=np.float32)
            map_next = torch.tensor(map_next, dtype=torch.float32).cuda()
            for i in range(self.global_map_dim[0]):
                for j in range(self.global_map_dim[1]):
                    vec1 = reg_obsv[:,:, i, j].unsqueeze(2)
                    vec2 = map_previous[:,:,i,j].unsqueeze(2)
                    vec = torch.cat((vec1, vec2), 2)
                    avg_out = self.update_avg(vec).squeeze(2)
                    map_next[:,:,i,j] = avg_out
            map_next = torch.tanh(map_next)
        return map_next