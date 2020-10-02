import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
import os
import math
import numpy as np

# Define the model to be used for imitation learning that receives a state from mapNet

class ILNet(nn.Module):
    def __init__(self, par, map_in_embedding, map_orient, tvec_dim, nActions, use_ego_obsv, drop_rate=0.2):
        super(ILNet, self).__init__()
        self.map_in_embedding = map_in_embedding
        self.orientations = map_orient
        self.fc_dim = par.fc_dim
        # Small cnn to take the map and extract an embedding
        self.map_cnn = nn.Sequential(
            nn.Conv2d(in_channels=map_in_embedding, out_channels=par.conv_embedding, kernel_size=3, stride=1, padding=0),
            nn.BatchNorm2d(num_features=par.conv_embedding),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2)
        )
        self.fc_map = nn.Linear(13*13*8, par.fc_dim)
        self.relu_map = nn.ReLU(inplace=True)
        self.drop_map = nn.Dropout(p=drop_rate)
        # Small cnn to take the position prediction and extract an embedding
        self.p_cnn = nn.Sequential(
            nn.Conv2d(in_channels=map_orient, out_channels=par.conv_embedding, kernel_size=3, stride=1, padding=0),
            nn.BatchNorm2d(num_features=par.conv_embedding),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.fc_p = nn.Linear(13*13*8, par.fc_dim)
        self.relu_p = nn.ReLU(inplace=True)
        self.drop_p = nn.Dropout(p=drop_rate)
        # Extract an embedding from the target one-hot vector
        self.fc_t = nn.Linear(tvec_dim, par.fc_dim)
        self.relu_t = nn.ReLU(inplace=True)
        self.drop_t = nn.Dropout(p=drop_rate)
        state_items = 3

        # if we are using egocentric observations
        if use_ego_obsv:
            self.fc_ego = nn.Linear(512, par.fc_dim)
            self.relu_ego = nn.ReLU(inplace=True)
            self.drop_ego = nn.Dropout(p=drop_rate)
            state_items = 4

        # Pass the concatenated inputs into an LSTM
        self.lstm = nn.LSTM(input_size=par.fc_dim*state_items+1, hidden_size=par.fc_dim*state_items+1, num_layers=1)
        # The fc layer that maps from hidden state space to action space 
        self.fc_action = nn.Linear(par.fc_dim*state_items+1, nActions)
        self.hidden = None

        # Use L1 or MSE loss?
        self.cost_loss = nn.MSELoss() #nn.L1Loss() #nn.MSELoss()


    def init_hidden(self, batch_size, state_items):
        return (torch.zeros(1, batch_size, self.fc_dim*state_items+1).cuda(), 
                torch.zeros(1, batch_size, self.fc_dim*state_items+1).cuda())

    def build_loss(self, cost_pred, cost_gt, loss_weight):
        loss = loss_weight * self.cost_loss(cost_pred, cost_gt)
        return loss



    def forward(self, state, use_ego_obsv):
        if use_ego_obsv:
            (map_pred, p_pred, t_vec, collision, ego_obsv) = state
        else:
            (map_pred, p_pred, t_vec, collision) = state
        #print(map_pred.shape)
        #print(p_pred.shape)
        #print(t_vec.shape)
        #print(collision.shape)
        if len(map_pred.size()) == 4: # add the sequence dimension
            map_pred = map_pred.unsqueeze(1)
            p_pred = p_pred.unsqueeze(1)

        batch_size = map_pred.shape[0]
        seq_len = map_pred.shape[1]
        global_map_dim = (map_pred.shape[3], map_pred.shape[4])
        # get map embedding
        map_pred = map_pred.view(batch_size*seq_len, self.map_in_embedding, global_map_dim[0], global_map_dim[1])
        map_cnn_out = self.map_cnn(map_pred)
        #print(map_cnn_out.shape)
        map_out = self.fc_map(map_cnn_out.view(map_cnn_out.shape[0], -1))
        map_out = self.relu_map(map_out)
        map_out = self.drop_map(map_out)
        map_out = map_out.view(batch_size, seq_len, -1)
        #print(map_out.shape)
        # get position embedding
        p_pred = p_pred.view(batch_size*seq_len, self.orientations, global_map_dim[0], global_map_dim[1])
        p_cnn_out = self.p_cnn(p_pred)
        #print(p_cnn_out.shape)
        p_out = self.fc_p(p_cnn_out.view(p_cnn_out.shape[0], -1))
        p_out = self.relu_p(p_out)
        p_out = self.drop_p(p_out)
        p_out = p_out.view(batch_size, seq_len, -1)
        #print(p_out.shape)
        # get target vector embedding
        t_out = self.fc_t(t_vec)
        t_out = self.relu_t(t_out)
        t_out = self.drop_t(t_out)
        t_out = t_out.unsqueeze(1).repeat(1, seq_len, 1) # replicate the target vec for each step in the sequence
        #print(t_out.shape)

        collision = collision.view(batch_size, seq_len, 1)
        #print(collision.shape)

        if use_ego_obsv:
            ego_obsv = ego_obsv.view(batch_size*seq_len, -1)
            ego_out = self.fc_ego(ego_obsv)
            ego_out = self.relu_ego(ego_out)
            ego_out = self.drop_ego(ego_out)
            ego_out = ego_out.view(batch_size, seq_len, -1)
            #print(ego_out.shape)
            x = torch.cat((map_out, p_out, t_out, collision, ego_out), 2)
        else:
            x = torch.cat((map_out, p_out, t_out, collision), 2)
        x = x.permute(1,0,2)
        lstm_out, self.hidden = self.lstm(x, self.hidden)
        #print(lstm_out.shape)
        lstm_out = lstm_out.permute(1,0,2)
        #lstm_out = lstm_out.view(batch_size*seq_len, -1)
        action_scores = self.fc_action(lstm_out) # batch_size x seq_len x nActions
        #print(action_scores.shape)
        return action_scores
        


class Encoder(nn.Module): # This is only the feature extractor from the image
    def __init__(self):
        super(Encoder, self).__init__()
        #os.environ['TORCH_HOME'] = par.root_path # so that torch will look at this folder (src/models) for the pretrained model and not download it again
        fnet = models.resnet18(pretrained=True)
        fnet.conv1 = nn.Conv2d(4, 64, kernel_size=7, stride=2, padding=3,bias=False)
        self.img_features = nn.Sequential(*list(fnet.children())[:-1]) # get resnet18 except the last fc layer
    def forward(self, img_data):
        img_out = self.img_features(img_data) # batch_size x 2048 x 1 x 1
        return img_out