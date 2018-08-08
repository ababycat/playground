import matplotlib.pyplot as plt
import matplotlib.patches as patches

def get_grid(tensor, nrow=2):
    return vision.utils.make_grid(tensor, nrow, padding=2,
              normalize=False, range=None, scale_each=False, pad_value=0)

def inference(model, loader, **config):
    is_cuda_type = config.get('is_cuda_type', False)
    max_step = config.get('max_step', 1)
    is_show_image = config.get('is_show_image', True)
    show_use_plot = config.get('show_use_matplotlib', True)
    save_results = config.get('is_save_results', False)
    threshold = config.get('threshold', 0.5)
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

        net_output = model(x_train, hidden)
        bboxs = parse_network_bbox_out_to_img(net_output.reshape(net_output.shape[0], -1, net_output.shape[-1]), **config)
        
        if is_show_image:
            if show_use_plot:
                for i in range(0, x.shape[0]):
                    grid = get_grid(x[i])
                    trans = vision.transforms.ToPILImage()
                    img = trans(grid)
                    fig = plt.figure(figsize=[10, 10])
                    plt.imshow(img)
                    currentAxis=plt.gca()
                    bbox = bboxs[i]

                    for b in range(0, bbox.shape[0]):
                        if b > 100:
                            break
                        rect = bbox[b, :].cpu().detach().numpy()
                        rect=patches.Rectangle((rect[0], rect[1]), rect[2], rect[3], \
                                               linewidth=2, edgecolor='r', facecolor='none')
                        currentAxis.add_patch(rect)

        step = step + 1
        if step == max_step:
            break


root = '/home/malong/data/coco/images/val2017'
annFile = '/home/malong/data/coco/annotations/instances_val2017.json'

inference_config = {
    'class_num': 81,
    'grid_height': 40,
    'grid_width': 40,
    'img_shape': (640, 640),
    'grid_depth': 9,
    'max_w': 640,
    'max_h': 640,
    'is_mini_mask': True,
    'mini_mask_shape': (28, 28),
    'max_object_num':128,
    
    'is_cuda_type': True,
    'max_step':1,
    'is_show_image':True,
    'show_use_matplotlib':True,
    'is_save_results':False,
    'class_num': 81,
    'grid_height': 40,
    'grid_width': 40,
    'img_shape': (640, 640),
    'grid_depth': 9,
    'max_w': 640,
    'max_h': 640,
    'is_mini_mask': True,
    'mini_mask_shape': (28, 28),
    'max_object_num':128,
    'batch_size':2,
    'threshold':0.8,
}

transform_inference = _Compose([
                        _Pad(inference_config['img_shape'][0]),
#                         _RandomApply([
# #                             _Resize(scale_range=(1.72, 2), raw=RAW_IMAGE_SIZE),
#                             _Resize(scale_range=(0.8, 1.2), raw=RAW_IMAGE_SIZE),
#                         ], p=0.99),
# #                         _RandomCrop(MODEL_INPUT_SIZE, MODEL_INPUT_SIZE, fill=0, padding_mode='constant'),
#                         _RandomRotation(0.999, 3),
#                         _ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
#                         _RandomHorizontalFlip(0.5),
# #                         _RandomVerticalFlip(0.5),
#                         _RandomGrayscale(0.01),
                        _Lambda(lambda x, y: (_ToTensor()(x), _ToTensor()(y)))
                ])

inference_torch = COCO_Detection_Dataset(root, annFile, transform=transform_inference, 
                                     method=parse_label_to_lstm, **inference_config)

inference_loader = torch.utils.data.DataLoader(inference_torch,
                                          batch_size=inference_config['batch_size'],
                                          shuffle=False,
                                          num_workers=8)

                                

inference(model, inference_loader, **inference_config)
