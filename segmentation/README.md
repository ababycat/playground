# use unet for segmentation

The model file `model_torch.py` is from [TernausNetV2](https://github.com/ternaus/TernausNetV2.git)


The repository includes:
* Modified torchvision/transorm for transforming images and labels simultaneous. See more details in `transforms.py`
* generate the dataset
* train the u-net network

Wait for implement:
* log the train process
* transform the image off-line for checking train or valid images easily.