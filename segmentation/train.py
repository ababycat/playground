from pathlib import Path

from dataset import *
from transform import *
from utils import *

MODEL_INPUT_SIZE = 512
RAW_IMAGE_SIZE = 1080

data_read, label_read = get_data(Path('./undersea'))
samples = len(data_read)
choice = list(np.random.permutation(samples))
data_random = [data_read[x] for x in choice]
label_random = [label_read[x] for x in choice]

k_split = 0.8
train_samples = int(np.floor(k_split*samples))

train_set = (data_random[0:train_samples], label_random[0:train_samples])
test_set = (data_random[train_samples:], label_random[train_samples:])

transform_train = _Compose([
                        _RandomApply([
                            _Resize(scale_range=(0.5, 1.5), raw=RAW_IMAGE_SIZE),
                        ], p=0.5),
                        _RandomCrop(MODEL_INPUT_SIZE, MODEL_INPUT_SIZE),
                        _RandomRotation(0.2, 360),
                        _ColorJitter(brightness=0, contrast=0, saturation=0, hue=0),
                        _RandomHorizontalFlip(0.5),
                        _RandomVerticalFlip(0.5),
                        _RandomGrayscale(0.01),
                        _Lambda(lambda x, y: (_ToTensor()(x), _ToTensor()(y)))
                ])

train_torch = SegData(train_set, train=True, transform=transform_train)
valid_torch = SegData(test_set, train=False, transform=None)

train_loader = torch.utils.data.DataLoader(train_torch,
                                          batch_size=4,
                                          shuffle=True,
                                          num_workers=8)

valid_loader = torch.utils.data.DataLoader(valid_torch,
                                          batch_size=4,
                                          shuffle=False,
                                          num_workers=1)

epchos = 800
for epcho in range(epchos):
    for idx, (x, y) in enumerate(train_loader):
        len(x)
    print(epcho)
