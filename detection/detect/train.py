rpn = AlbuNet(num_filters=32, pretrained=True, is_deconv=True)
detect = RNN_Decoder(samples_size=\
                     model_config['batch_size']*model_config['grid_height']*model_config['grid_width'], 
                     input_size=256, hidden_size=256, linear_output_size=85, decode_times=9)
model = DetectNet(rpn, detect).cuda()


log = {
    "class_loss": [],
    "bbox_loss": [], 
    "score_loss": []
}


dataset = 'val'
root = '/home/malong/data/coco/images/'+dataset+'2017'
annFile = '/home/malong/data/coco/annotations/instances_'+dataset+'2017.json'

model_config = {
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
    'epochs': 1,
    'is_cuda_type': True,
    'print_log_every_step': 100,
    'learning_rate': 0.001,
    'decay_learning_rate_at_epcho': [50]
}

transform_train = _Compose([
                        _Pad(model_config['img_shape'][0]),
#                         _RandomApply([
# #                             _Resize(scale_range=(1.72, 2), raw=RAW_IMAGE_SIZE),
#                             _Resize(scale_range=(0.8, 1.2), raw=RAW_IMAGE_SIZE),
#                         ], p=0.99),
# #                         _RandomCrop(MODEL_INPUT_SIZE, MODEL_INPUT_SIZE, fill=0, padding_mode='constant'),
#                         _RandomRotation(0.999, 3),
                        _ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
#                         _RandomHorizontalFlip(0.5),
# #                         _RandomVerticalFlip(0.5),
#                         _RandomGrayscale(0.01),
                        _Lambda(lambda x, y: (_ToTensor()(x), _ToTensor()(y)))
                ])

train_torch = COCO_Detection_Dataset(root, annFile, transform=transform_train, 
                                     method=parse_label_to_lstm, **model_config)

train_loader = torch.utils.data.DataLoader(train_torch,
                                          batch_size=model_config['batch_size'],
                                          shuffle=False,
                                          num_workers=8)

# loss_fn = AlbuNet_LSTM_Loss(80)
loss_fn = AlbuNet_LSTM_Loss(50, 0.96, 80)
optimizer = optim.Adam(params=model.parameters(), lr=model_config['learning_rate'])

for i in range(100):
    train(model, loss_fn, optimizer, train_loader, log, **model_config)
    print('----------')
    print(torch.min(model.detect_model.linear1.weight.grad).item(), torch.max(model.detect_model.linear1.weight.grad).item())
