import os
import random

from PIL import Image
from pathlib import Path
import numpy as np
import torch
from torch import optim
from torch import nn
import torchvision as vision
import matplotlib.pyplot as plt

from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.datasets as dst
import torchvision.transforms as transforms

from torchvision import models
import torchvision
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.datasets as dst
import torchvision.transforms as transforms

from torchvision import models
import torchvision

import datetime

MORE_THAN_ONE_OBJECTS_IN_GRID = 1
error_id = 0

import cv2
    
import torchvision.datasets as dst
from transform import *

def idx_image_to_grid_vec(bbox, grid_w_range, grid_h_range, max_w, max_h):
    bbox = bbox.type(torch.FloatTensor)
    img_x, img_y, w, h = torch.chunk(bbox, chunks=4, dim=1)
    c_x, c_y = img_x+w/2., img_y+h/2.
    grid_x = torch.floor(c_x/grid_w_range)
    grid_y = torch.floor(c_y/grid_h_range)
    grid_cx, grid_cy = (grid_x+0.5)*grid_w_range, (grid_y+0.5)*grid_h_range
    new_bbox = torch.cat([(c_x-grid_cx)/grid_w_range, (c_y-grid_cy)/grid_h_range, 
                          torch.exp(w/max_w), torch.exp(h/max_h)], dim=1)
    return new_bbox, grid_x.type(torch.LongTensor), grid_y.type(torch.LongTensor)


def parse_label_to_lstm(label, nodata=False, **method_config):   
    global error_id
    error_id = 0

    class_num = method_config['class_num'] # TODO:81
    grid_height = method_config['grid_height']
    grid_width = method_config['grid_width']
    img_height, img_width = method_config['img_shape']
    grid_depth = method_config['grid_depth']
    
    grid_h_range = int(img_height//grid_height)
    grid_w_range = int(img_width//grid_width)
    max_w = method_config['max_w'] # TODO: FOR NORMALIZE
    max_h = method_config['max_h']
    
    is_mini_mask = method_config['is_mini_mask']
    if is_mini_mask:
        mask_shape = method_config['mini_mask_shape']
        
    max_object_num = method_config['max_object_num']

    # x, y, w, h, class_number
    out_scores = torch.zeros((grid_depth, 1, grid_height, grid_width))
    out_bbox = torch.zeros((grid_depth, 4, grid_height, grid_width))
    
    if nodata:
        # x, y, w, h, class_number
        out_scores = out_scores.reshape(grid_height*grid_width*grid_depth)
        out_bbox = out_bbox.reshape(grid_height*grid_width*grid_depth, 4)
        # train_index = torch.empty(grid_height*grid_width*grid_depth, dtype=torch.uint8).fill_(0)
        positive_index = torch.empty(grid_height*grid_width*grid_depth, dtype=torch.uint8).fill_(0)
        negative_index = torch.empty(grid_height*grid_width, grid_depth, dtype=torch.uint8).fill_(1)
        negative_index[:, 0] = 1
        negative_index = negative_index.reshape(-1)
        train_index = positive_index | negative_index
        if is_mini_mask:
            target = torch.zeros(max_object_num, 1, mask_shape[0], mask_shape[1])
        else:
            target = torch.zeros(max_object_num, 1, img_height, img_width)
        return (out_scores, out_bbox, train_index, positive_index, negative_index, target), 0
    
    class_ids, bbox, target = label
    
    # use grid save idx in object_number
    grid_log_idx = torch.zeros((grid_depth, grid_height, grid_width)).fill_(-1)
    grid_log_num = torch.zeros((grid_height, grid_width), dtype=torch.long)
    
    new_bbox, grid_x, grid_y = idx_image_to_grid_vec(bbox, grid_w_range, grid_h_range, max_w, max_h)
    object_number = class_ids.shape[0]
    # add object to grid
    for idx in range(object_number):
        grid_idx_x, grid_idx_y = grid_x[idx], grid_y[idx]
        
        num = grid_log_num[grid_idx_y, grid_idx_x]
        out_bbox[num, :, grid_idx_y, grid_idx_x] = new_bbox[idx, :]
        out_scores[num, 0, grid_idx_y, grid_idx_x] = class_ids[idx, 0].type(torch.FloatTensor)
        grid_log_idx[num, grid_idx_y, grid_idx_x] = idx
        grid_log_num[grid_idx_y, grid_idx_x] += 1 
    
    max_val = torch.max(grid_log_num).item()
    
    # sort the grid which have mulity objects by area.
    # out taget_idx
    pos = (grid_log_num>1).nonzero()
    for i in range(pos.shape[0]):
        row, col = pos[i, :]
        
        num = grid_log_num[row, col]
        
        # for safety minus 1
        if num>=grid_depth-1:
            error_id = MORE_THAN_ONE_OBJECTS_IN_GRID
            print('there was more than %d(grid_depth) objects in the grid' %(num))
            continue

        choice = grid_log_idx[:num, row, col].reshape(-1, 1).type(torch.long)
        area = torch.sum(target[choice, :, :].reshape(choice.shape[0], -1), 1, True)
        table = torch.cat([area, choice.type(torch.float)], dim=1)
        result, index = torch.sort(table[:, 0], dim=0, descending=True)
        new_choice = table[index, 1].type(torch.long)
        out_bbox[:num, :, row, col] = out_bbox[index, :, row, col]
        out_scores[:num, 0, row, col] = out_scores[index, 0, row, col]
        grid_log_idx[:num, row, col] = new_choice
    
    # mask
    if is_mini_mask:
        out_target = torch.zeros((max_object_num, 1, *mask_shape))
        for idx in range(object_number):
            x, y, w, h = bbox[idx, :]
            try:
                raw_mask = target[idx, y:y+h, x:x+w].unsqueeze(dim=0).unsqueeze(dim=0)
            except:
                print('raw mask wrong:, idx:', idx, x, y, w, h)
            out_target[idx, :, :, :] = torch.nn.functional.upsample(raw_mask, 
                                                size=mask_shape, mode='bilinear', align_corners=True)
    else:
        out_target = target

    # simplify the name
    H, W, T = grid_height, grid_width, grid_depth
        
    # (H*W, T)
    index = torch.argmin(grid_log_idx.permute(1, 2, 0).reshape(-1, T), dim=1)
    negative_index = torch.empty(H*W, T, dtype=torch.uint8).fill_(0)
    negative_index[torch.arange(H*W, dtype=torch.long), index] = 1
    # (H*W*T)
    negative_index = negative_index.reshape(-1)
    
    # (H*W*T)
    positive_index = (grid_log_idx.permute(1, 2, 0)>=0).reshape(-1)
    train_index = positive_index | negative_index
    
    out_scores = out_scores.permute(2, 3, 0, 1).reshape(H*W*T)
    out_bbox = out_bbox.permute(2, 3, 0, 1).reshape(H*W*T, 4)
    return (out_scores, out_bbox, train_index, positive_index, negative_index, out_target), max_val    



def train(model, loss_fn, optimizer, loader_train, log, **config):
    epochs = config.get('epochs', 1)
    is_cuda_type = config.get('is_cuda_type', False)
    print_every = config.get('print_log_every_step', 30)
    learning_rate = config.get('learning_rate', 1e-4)
    decay_lr_time = config.get('decay_learning_rate_at_epcho', [None])
    print(epochs)
    
    for p in optimizer.param_groups:
        p['lr'] = learning_rate
    print('learning_rate: ', learning_rate)
    
    model.train()
    
    step = 0
    for epoch in range(epochs):
        print('Starting epoch %d / %d' % (epoch + 1, epochs))
        for x, y in loader_train:           
            if is_cuda_type:
                x_train = x.type(torch.cuda.FloatTensor)
                out_scores, out_bbox, train_index, positive_index, negative_index, out_target = y
                y_train = out_scores.type(torch.cuda.FloatTensor), \
                            out_bbox.type(torch.cuda.FloatTensor), \
                            train_index.type(torch.cuda.ByteTensor), \
                            positive_index.type(torch.cuda.ByteTensor), \
                            negative_index.type(torch.cuda.ByteTensor), \
                            out_target.type(torch.cuda.FloatTensor)
            else:
                x_train = x.type(torch.FloatTensor)
                y_train = y
            
            hidden = model.init_rnn_state(config['batch_size']*config['grid_height']*config['grid_width'])
            if is_cuda_type:
                hx, cx = hidden
                hidden = hx.type(torch.cuda.FloatTensor), cx.type(torch.cuda.FloatTensor)
            
            net_output = model(x_train, hidden)
          
#             class_loss, bbox_loss, score_loss = loss_fn(net_output, y_train)
#             class_loss, bbox_loss, score_loss = 0, 0, 0

#             loss = class_loss + bbox_loss + score_loss
#             loss = score_loss
#             loss = torch.mean(net_output)
            loss, [class_loss, bbox_loss, score_loss] = loss_fn(net_output, y_train)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            log['class_loss'].append(class_loss)
            log['bbox_loss'].append(bbox_loss)
            log['score_loss'].append(score_loss)
            
#             if (step+1) % print_every == 0:
            if True:
                lr = optimizer.param_groups[0]['lr']
                print('lr=%.9f, t = %d, class loss = %.9f, bbox_loss = %.9f,  score loss = %.9f' % (lr, step + 1, 
                                                    class_loss, bbox_loss, score_loss))
#                 print('lr=%.9f, t = %d, class loss = %.9f, bbox_loss = %.9f,  score loss = %.9f' % (lr, step + 1, 
#                                                     0, 0, 0))

            step = step + 1
            if step == 1:
                break
        
        if epoch+1 in decay_lr_time:
            for p in optimizer.param_groups:
                p['lr'] = p['lr'] * 0.1
            print('learning rate', p['lr'])


def parse_network_bbox_out_to_img(network_output, threshold, **method_config):   
    """BSx40x40xtimesx85"""
    BS = method_config['batch_size']
    H = method_config['grid_height']
    W = method_config['grid_width']
    img_height, img_width = method_config['img_shape']
    T = method_config['grid_depth']
    
    grid_h_range = int(img_height//H)
    grid_w_range = int(img_width//W)
    max_w = method_config['max_w'] # TODO: FOR NORMALIZE
    max_h = method_config['max_h']
       
    # network_out (BSxHxWxT)x85    
    network_out = network_output.reshape(-1, network_output.shape[-1])
    
    score = torch.nn.functional.sigmoid(network_out[:, 0]).reshape(-1, 1)
    class_ids = torch.nn.functional.sigmoid(network_out[:, 1:-4])
    bbox_xy = torch.nn.functional.tanh(network_out[:, -4:-2])
    bbox_wh = torch.exp(torch.nn.functional.sigmoid(network_out[:, -2:]))
    
    network_out = torch.cat([score, class_ids, bbox_xy, bbox_wh], dim=1)
    
    print('score max:%f,    min%f' %(torch.max(score).item(), torch.min(network_out[:, 0]).item()))
    
    # score_out (BSxHxWxT)
#     score_out = network_out[:, 0] < threshold
#     # score_out (BSxHxW)xT
#     score_out = score_out.reshape(-1, T)
#     # stop_signal (BSxHxW)
#     stop_signal = torch.argmax(score_out, dim=1)
#     print('num of objects:%d' %(torch.sum(score_out).item()))
#     print(torch.max(stop_signal))
  
    positive_index = network_out[:, 0] > threshold
#     positive_index = positive_index.reshape(BS, -1)

#     # positive_index (BSxHxW)xT
#     positive_index = torch.zeros_like(score_out, dtype=torch.long)
#     idx = torch.arange(0, score_out.shape[0], dtype=torch.long)
#     fill = torch.arange(0, T, dtype=torch.long)
#     if stop_signal.is_cuda:
#         positive_index = positive_index.cuda()
#         idx = idx.cuda()
#         fill = fill.cuda()
#     positive_index[idx, :] = fill
#     # positive_index (BSxHxWxT)
#     positive_index = (positive_index < stop_signal.reshape(-1, 1).repeat(1, T)).reshape(-1)
#     print('num of objects:%d' %(torch.sum(positive_index).item()))
    
    w = np.arange(0, img_width, grid_w_range, dtype=np.float32)
    h = np.arange(0, img_height, grid_h_range, dtype=np.float32)
    ww, hh = np.meshgrid(w, h)
    cx, cy = ww + grid_w_range/2., hh + grid_h_range/2.
    cx, cy = torch.from_numpy(cx).unsqueeze(-1).repeat(1, 1, T), torch.from_numpy(cy).unsqueeze(-1).repeat(1, 1, T)
    cx, cy = cx.reshape(-1), cy.reshape(-1)
    if positive_index.is_cuda:
        cx, cy = cx.cuda(), cy.cuda()
    
    # network_out bbox BSx(HxWxT)x4
    network_out = network_out.reshape(BS, -1, network_out.shape[-1])
    
    positive_index = positive_index.reshape(BS, -1)
    bbox_out = []
    for i in range(BS):
        # network_out bbox (HxWxT)x4
        x = network_out[i, positive_index[i, :], -4]*grid_w_range + cx[positive_index[i, :]]
        y = network_out[i, positive_index[i, :], -3]*grid_h_range + cy[positive_index[i, :]]
        w = torch.log(network_out[i, positive_index[i, :], -2])*max_w
        h = torch.log(network_out[i, positive_index[i, :], -1])*max_h
        x = x-w/2
        y = y-h/2
        bbox_out.append(torch.cat([t.reshape(-1, 1) for t in [x, y, w, h]], dim=1))
    return bbox_out




class AlbuNet_LSTM_Loss:
    def __init__(self, k, pos_neg_prop, no_ground_class_nums):
#     def __init__(self, no_ground_class_nums):
        self.pos_neg_prop = pos_neg_prop
        self.k = k
        self.class_nums = no_ground_class_nums
#         self.score_loss = nn.BCEWithLogitsLoss()
        self.class_loss = nn.BCEWithLogitsLoss()
        self.bbox_loss = nn.MSELoss()
        self.mask_loss = nn.BCEWithLogitsLoss()
        
    def __call__(self, net_output, label):
        """
        net_output: BSx(40x40xT)x85
        TODO: NOTE that the loss have an another dim 'bs'.
        """

        BS, H, W, T, C = net_output.shape
        net_output = net_output.reshape(BS, H*W*T, -1)
        out_scores, out_bbox, train_index, positive_index, negative_index, out_target = label
        
        net_output = net_output.reshape(-1, net_output.shape[-1])
        out_scores = out_scores.reshape(-1, 1)
        out_bbox = out_bbox.reshape(-1, 4)
        train_index = train_index.reshape(-1)
        positive_index = positive_index.reshape(-1)
        negative_index = negative_index.reshape(-1)
        
        negative_num = torch.sum(negative_index)
        positive_num = torch.sum(positive_index)
        negative_index_copy = negative_index.clone()
        choice = torch.randperm(negative_num)
        if net_output.is_cuda:
            choice = choice.cuda()
        negative_index_copy[negative_index] = choice < positive_num
        negative_index = negative_index_copy
            
        print(torch.sum(positive_index), torch.sum(negative_index))
        
        # scores
        scores_label = (out_scores[train_index, 0] > 0).type(torch.float)
        weight = self.k*self.pos_neg_prop*positive_index.type(torch.float) + \
                    self.k*(1-self.pos_neg_prop)*negative_index.type(torch.float)
        if net_output.is_cuda:
            weight = weight.cuda()
#         self.score_loss = nn.BCEWithLogitsLoss(weight[train_index])
        self.score_loss = nn.BCEWithLogitsLoss()
        score_loss = self.score_loss(net_output[train_index, 0], scores_label)
        
        # bbox
        if torch.sum(positive_index) == 0:
            class_loss = torch.zeros([1])
            bbox_loss = torch.zeros([1])
            if net_output.is_cuda:
                class_loss = class_loss.type(torch.cuda.FloatTensor)
                bbox_loss = bbox_loss.type(torch.cuda.FloatTensor)
                return class_loss + bbox_loss + score_loss, [class_loss.item(), bbox_loss.item(), score_loss.item()]
        else:
            bbox_xy = torch.nn.functional.tanh(net_output[positive_index, -4:-2])
            bbox_wh = torch.exp(torch.nn.functional.sigmoid(net_output[positive_index, -2:]))
            network_out_bbox = torch.cat([bbox_xy, bbox_wh], dim=1)
            bbox_loss = self.bbox_loss(network_out_bbox, out_bbox[positive_index, :])

            # classify
            samples = out_scores[positive_index, 0].shape[0]

            if net_output.is_cuda:
                class_one_hot_label = torch.zeros(samples, self.class_nums+1).type(torch.cuda.FloatTensor)
                one_hot_index = out_scores[positive_index, 0].type(torch.cuda.LongTensor).reshape(-1, 1)
            else:
                class_one_hot_label = torch.zeros(samples, self.class_nums+1)
                one_hot_index = out_scores[positive_index, 0].type(torch.LongTensor).reshape(-1, 1)

            class_one_hot_label = class_one_hot_label.scatter_(1, one_hot_index, 1)
            class_loss = self.class_loss(net_output[positive_index, 1:-4], class_one_hot_label[:, 1:])
        # return class_loss, bbox_loss, score_loss
        return class_loss + bbox_loss + score_loss, [class_loss.item(), bbox_loss.item(), score_loss.item()]
