import torch
import torchvision as vision

from utils import parse_network_bbox_out_to_img
from utils import show_score_classid_bbox

import datetime

def inference(model, loader, table=None, **config):
    is_cuda_type = config.get('is_cuda_type', False)
    max_step = config.get('max_step', 1)
    is_show_image = config.get('is_show_image', True)
    show_use_plot = config.get('show_use_matplotlib', True)
    save_results = config.get('is_save_results', False)
    threshold = config.get('threshold', 0.5)
    batch_size = config.get('batch_size')
    print('threshold: ', threshold)
    model.eval()
    
    step = 0
    for x, y in loader:
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

        start  = datetime.datetime.now()
        net_output = model(x_train, hidden)
        scores, class_ids, bboxs = parse_network_bbox_out_to_img(net_output, **config)
        
        if show_use_plot:
            show_score_classid_bbox(x, scores, class_ids, bboxs, table)
        else:
            # use opencv
            pass
        
        step = step + 1
        if step == max_step:
            break

    if save_results:
        pass
