import torch
from torch import nn

class AlbuNet_LSTM_Loss:
    def __init__(self, k, pos_neg_prop, no_ground_class_nums):
#     def __init__(self, no_ground_class_nums):
        self.pos_neg_prop = pos_neg_prop
        self.k = k
        self.class_nums = no_ground_class_nums
        self.score_loss = nn.BCEWithLogitsLoss()
        self.class_loss = nn.BCEWithLogitsLoss()
        # self.bbox_loss = nn.MSELoss()
        self.bbox_loss = nn.SmoothL1Loss()
        self.mask_loss = nn.BCEWithLogitsLoss()
        
    def __call__(self, net_output, label):
        """
        net_output: BSx(40x40xT)x85
        TODO: NOTE that the loss have an another dim 'bs'.
        torch.Size([1, 8000, 85])
        -----------
        torch.Size([1, 8000])
        torch.Size([1, 8000, 4])
        torch.Size([1, 8000])
        torch.Size([1, 8000])
        torch.Size([1, 8000])
        torch.Size([1, 128, 1, 28, 28])
        """

        BS, H, W, T, C = net_output.shape
        net_output = net_output.reshape(BS, H*W*T, -1)
        out_scores, out_bbox, train_index, positive_index, negative_index, out_target = label
        
        net_output = net_output.reshape(-1, net_output.shape[-1])
        out_scores = out_scores.reshape(-1, 1)
        out_bbox = out_bbox.reshape(-1, 4)
        train_index = train_index.reshape(-1)
        negative_index = negative_index.reshape(-1)
        positive_index = positive_index.reshape(-1)
        
        negative_index_copy1 = negative_index.reshape(BS, H, W, T).clone()
        negative_index_copy1[:, :, :, 0] = 0
        
        negative_num = torch.sum(negative_index)
        positive_num = torch.sum(positive_index)
        negative_index_copy = negative_index.clone()
        choice = torch.randperm(negative_num)
        if net_output.is_cuda:
            choice = choice.cuda()
        negative_index_copy[negative_index] = choice < max(positive_num * 10, 10)
        negative_index = negative_index_copy | negative_index_copy1.reshape(-1)
                    
        train_index = positive_index |  negative_index
        
        # scores
        scores_label = (out_scores[train_index, 0] > 0).type(torch.float)
        # weight = self.k*self.pos_neg_prop*positive_index.type(torch.float) + \
        #             self.k*(1-self.pos_neg_prop)*negative_index.type(torch.float)
        # if net_output.is_cuda:
        #     weight = weight.cuda()
        # self.score_loss = nn.BCEWithLogitsLoss(weight[train_index])
        score_loss = self.score_loss(net_output[train_index, 0], scores_label)
        
        # bbox
        if torch.sum(positive_index) == 0:
            class_loss = torch.zeros([1], dtype=torch.float)
            bbox_loss = torch.zeros([1], dtype=torch.float)
            if net_output.is_cuda:
                class_loss = class_loss.cuda()
                bbox_loss = bbox_loss.cuda()
                return class_loss + bbox_loss + score_loss, [class_loss.item(), bbox_loss.item(), score_loss.item()]
        else:
            # bbox_xy = torch.nn.functional.tanh(net_output[positive_index, -4:-2])
            bbox_xy = torch.tanh(net_output[positive_index, -4:-2])
            # bbox_wh = torch.exp(torch.nn.functional.sigmoid(net_output[positive_index, -2:]))
            # bbox_wh = torch.log(torch.nn.functional.sigmoid(net_output[positive_index, -2:]))
            # bbox_wh = torch.nn.functional.sigmoid(net_output[positive_index, -2:])
            bbox_wh = torch.sigmoid(net_output[positive_index, -2:])
            # network_out_bbox = torch.cat([bbox_xy, bbox_wh], dim=1)
            # bbox_loss = self.bbox_loss(network_out_bbox, out_bbox[positive_index, :])*4
            # bbox_loss = self.bbox_loss(network_out_bbox, torch.log(out_bbox[positive_index, :]))*4
            bbox_xy_loss = self.bbox_loss(bbox_xy, out_bbox[positive_index, -4:-2])
            # print(torch.max(net_output[positive_index, -2:]), torch.min(net_output[positive_index, -2:]))
            # print(torch.max(out_bbox[positive_index, -2:]), torch.min(out_bbox[positive_index, -2:]))
            bbox_wh_loss = self.bbox_loss(torch.exp(bbox_wh), torch.exp(out_bbox[positive_index, -2:]))
            bbox_loss = bbox_xy_loss + bbox_wh_loss

            # classify
            # samples = out_scores[positive_index, 0].shape[0]
            samples = torch.sum(positive_index).item()
            if net_output.is_cuda:
                class_one_hot_label = torch.zeros(samples, self.class_nums+1).type(torch.cuda.FloatTensor)
                one_hot_index = out_scores[positive_index, 0].type(torch.cuda.LongTensor).reshape(-1, 1)
            else:
                class_one_hot_label = torch.zeros(samples, self.class_nums+1)
                one_hot_index = out_scores[positive_index, 0].type(torch.LongTensor).reshape(-1, 1)
            class_one_hot_label = class_one_hot_label.scatter_(1, one_hot_index, 1)
            class_loss = self.class_loss(net_output[positive_index, 1:-4], class_one_hot_label[:, 1:])
        return class_loss + bbox_loss + score_loss, [class_loss.item(), bbox_loss.item(), score_loss.item()]

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
          
            loss, [class_loss, bbox_loss, score_loss] = loss_fn(net_output, y_train)
            
            optimizer.zero_grad()
            loss.backward()
            # torch.nn.utils.clip_grad_norm_(model.parameters(), 10)
            optimizer.step()

            log['class_loss'].append(class_loss)
            log['bbox_loss'].append(bbox_loss)
            log['score_loss'].append(score_loss)
            
            if (step+1) % print_every == 1:
                lr = optimizer.param_groups[0]['lr']
                print('lr=%.9f, t = %d, class loss = %.9f, bbox_loss = %.9f,  score loss = %.9f' % (lr, step + 1, 
                                                    class_loss, bbox_loss, score_loss))
#                 print('lr=%.9f, t = %d, class loss = %.9f, bbox_loss = %.9f,  score loss = %.9f' % (lr, step + 1, 
#                                                     0, 0, 0))

            step = step + 1
            # if step == epochs:
            #     return
            
        if epoch in decay_lr_time:
            for p in optimizer.param_groups:
                p['lr'] = p['lr'] * 0.8
            print('learning rate', p['lr'])
