import torch
import torchvision as vision
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from matplot_color import random_color, get_color

MORE_THAN_ONE_OBJECTS_IN_GRID = 1

def get_bbox_from_mask(mask):
    """mask.shape=N*H*W or H*W"""
    if isinstance(mask, torch.Tensor):
        mask = mask.cpu().type(torch.IntTensor).numpy()
    if len(mask.shape) == 3:
        out = []
        for i in range(mask.shape[0]):
            try:
                bbox = np.array(list(cv2.boundingRect(mask[i, :, :].astype(np.uint8))))
            except:
                print('get_bbox_from_mask_error')
                bbox = np.array([0, 0, 0, 0])
                continue
            out.append(bbox)
        return np.stack(out, axis=0)
    elif len(mask.shape) == 2:
        # return np.array(list(cv2.boundingRect(mask[i, :, :])))
        return np.array(list(cv2.boundingRect(mask[:, :]))).reshape(1, -1)
    else:
        raise 'error for mask.shape'

def idx_image_to_grid_vec(bbox, grid_w_range, grid_h_range, max_w, max_h):
    bbox = bbox.type(torch.FloatTensor)
    img_x, img_y, w, h = torch.chunk(bbox, chunks=4, dim=1)
    c_x, c_y = img_x+w/2., img_y+h/2.
    grid_x = torch.floor(c_x/grid_w_range)
    grid_y = torch.floor(c_y/grid_h_range)
    grid_cx, grid_cy = (grid_x+0.5)*grid_w_range, (grid_y+0.5)*grid_h_range
    new_bbox = torch.cat([(c_x-grid_cx)/grid_w_range, (c_y-grid_cy)/grid_h_range, 
                          w/max_w, h/max_h], dim=1)
    return new_bbox, grid_x.type(torch.LongTensor), grid_y.type(torch.LongTensor)

def parse_label_to_lstm(label, nodata=False, **method_config):   
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
#         train_index = torch.empty(grid_height*grid_width*grid_depth, dtype=torch.uint8).fill_(0)
        positive_index = torch.empty(grid_height*grid_width*grid_depth, dtype=torch.uint8).fill_(0)
        negative_index = torch.empty(grid_height*grid_width, grid_depth, dtype=torch.uint8).fill_(1)
        negative_index[:, 0] = 1
        negative_index = negative_index.reshape(-1)
        train_index = positive_index | negative_index
        if is_mini_mask:
            target = torch.zeros(max_object_num, 1, mask_shape[0], mask_shape[1])
        else:
            target = torch.zeros(max_object_num, 1, img_height, img_width)
        return (out_scores, out_bbox, train_index, positive_index, negative_index, target), 0, error_id
    
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
    fac = torch.arange(1-1/(T+1), 0, -1/(T+1)).reshape(1, -1)
    index = torch.argmin(grid_log_idx.permute(1, 2, 0).reshape(-1, T)*fac, dim=1)
    negative_index = torch.empty(H*W, T, dtype=torch.uint8).fill_(0)
    negative_index[torch.arange(H*W, dtype=torch.long), index] = 1
    # (H*W*T)
    negative_index = negative_index.reshape(-1)
    
    # (H*W*T)
    positive_index = (grid_log_idx.permute(1, 2, 0)>=0).reshape(-1)
    train_index = positive_index | negative_index
    
    out_scores = out_scores.permute(2, 3, 0, 1).reshape(H*W*T)
    out_bbox = out_bbox.permute(2, 3, 0, 1).reshape(H*W*T, 4)
    return (out_scores, out_bbox, train_index, positive_index, negative_index, out_target), max_val, error_id

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
    
    # score = torch.nn.functional.sigmoid(network_out[:, 0]).reshape(-1, 1)
    # class_ids = torch.nn.functional.sigmoid(network_out[:, 1:-4])
    # bbox_xy = torch.nn.functional.tanh(network_out[:, -4:-2])
    # bbox_wh = torch.nn.functional.sigmoid(network_out[:, -2:])
    score = torch.sigmoid(network_out[:, 0]).reshape(-1, 1)
    class_ids = torch.sigmoid(network_out[:, 1:-4])
    bbox_xy = torch.tanh(network_out[:, -4:-2])
    bbox_wh = torch.sigmoid(network_out[:, -2:])
    # print(torch.min(network_out[:, -2:]).item(), torch.max(network_out[:, -2:]).item())
    
    network_out = torch.cat([score, class_ids, bbox_xy, bbox_wh], dim=1)

    # stop_choice (BSxHxWxT)
    stop_choice = network_out[:, 0] < threshold
    # stop_choice (BSxHxW)xT
    stop_choice = stop_choice.reshape(-1, T)
    
    # stop_signal (BSxHxW)
    fac = torch.arange(1, 1/(T+1), -1/(T+1)).reshape(1, -1)
    if network_output.is_cuda:
        fac = fac.cuda()
    stop_signal = torch.argmax((stop_choice).type(torch.float)*fac, dim=1) 
    
    # just hist
    output = network_out[:, 0].cpu().detach().numpy().reshape(-1)
    plt.figure(figsize=[5, 5])
    plt.hist(output)
    plt.show()

    # positive_index (BSxHxW)xT
    positive_index = torch.arange(0, T, dtype=torch.long).reshape(1, T).repeat(BS*H*W, 1)
    if network_out.is_cuda:
        positive_index = positive_index.cuda()
    # positive_index (BSxHxWxT)
    positive_index = (positive_index < stop_signal.reshape(-1, 1).repeat(1, T)).reshape(-1)
    print('num of objects:%d' %(torch.sum(positive_index).item()))

    # cx, cy H*W*T
    cx, cy = get_cx_cy(img_height, img_width, H, W, T, is_cuda=network_output.is_cuda)
    
    # network_out bbox BSx(HxWxT)x85
    network_out = network_out.reshape(BS, -1, network_out.shape[-1])
    
    # positive_index bbox BSx(HxWxT)
    positive_index = positive_index.reshape(BS, -1)
    bbox_out = []
    score_out = []
    class_ids_out = []

    for b in range(BS):
        if torch.sum(positive_index[b, :]) == 0:
            score_out.append(None)
            class_ids_out.append(None)
            bbox_out.append(None)
            continue
        # score
        score_out.append(network_out[b, positive_index[b, :], 0])

        # start from zero
        class_ids = torch.argmax(network_out[b, positive_index[b, :], 1:-4], dim=1)
        class_ids_out.append(class_ids)

        # network_out bbox (HxWxT)x4
        x = network_out[b, positive_index[b, :], -4]*grid_w_range + cx[positive_index[b, :]]
        y = network_out[b, positive_index[b, :], -3]*grid_h_range + cy[positive_index[b, :]]
        w = network_out[b, positive_index[b, :], -2]*max_w
        h = network_out[b, positive_index[b, :], -1]*max_h
        x = x-w/2
        y = y-h/2
        bbox_out.append(torch.cat([t.reshape(-1, 1) for t in [x, y, w, h]], dim=1))

    return score_out, class_ids_out, bbox_out

def get_cx_cy(img_height, img_width, H, W, T, is_cuda):
    grid_w_range = img_width/W
    grid_h_range = img_height/H
    w = np.arange(0, img_width, grid_w_range, dtype=np.float32)
    h = np.arange(0, img_height, grid_h_range, dtype=np.float32)
    ww, hh = np.meshgrid(w, h)
    cx, cy = ww + grid_w_range/2., hh + grid_h_range/2.
    cx, cy = torch.from_numpy(cx).unsqueeze(-1).repeat(1, 1, T), torch.from_numpy(cy).unsqueeze(-1).repeat(1, 1, T)
    cx, cy = cx.reshape(-1), cy.reshape(-1)
    if is_cuda:
        cx, cy = cx.cuda(), cy.cuda()
    return cx, cy

def get_grid(tensor, nrow=2):
    return vision.utils.make_grid(tensor, nrow, padding=2,
              normalize=False, range=None, scale_each=False, pad_value=0)

def show_score_classid_bbox(x, scores, class_ids, bboxs, table=None):
    # x.shape (BS, C, H, W) or (C, H, W)
    batch_size = x.shape[0] if len(x.shape)==4 else 1
    x = x.unsqueeze(0) if len(x.shape)==3 else x
    for b in range(batch_size):
        grid = get_grid(x[b])
        trans = vision.transforms.ToPILImage()
        img = trans(grid)
        fig = plt.figure(figsize=[10, 10])
        plt.imshow(img)
        # plt.axis('off')
        ax=plt.gca()
        
        score = scores[b]
        if score is None:
            continue
        class_id = class_ids[b]
        bbox = bboxs[b]
        print(bbox)
        for idx in range(0, bbox.shape[0]):
            if idx > 100:
                break
            rect = bbox[idx, :].cpu().detach().numpy()
            color = random_color()
            rect_ax = patches.Rectangle((rect[0], rect[1]), rect[2], rect[3], \
                                    linewidth=2, edgecolor=color, facecolor='none')
            ax.add_patch(rect_ax)
            
            if table is not None:
                label = table[class_id[idx].item() + 1]
                
                caption = "{}{:.2f}".format(label, score[idx])
                color = get_color(class_id[idx].item())
                ax.text(rect[0]+2, rect[1]+10, caption, color=color)

        plt.show()
