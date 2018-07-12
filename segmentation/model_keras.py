import numpy as np
import matplotlib.pyplot as plt
import cv2

from keras.preprocessing import image

MODEL = "VGG16"

if MODEL == "VGG16":
    from keras.applications.vgg16 import VGG16 as Model_pre
    from keras.applications.vgg16 import preprocess_input
elif MODEL == "ResNet50":
    from keras.applications.resnet50 import ResNet50 as Model_pre
    from keras.applications.resnet50 import preprocess_input
elif MODEL == "MobileNet":
    from keras.applications.mobilenet import MobileNet as Model_pre
    from keras.applications.mobilenet import preprocess_input
elif MODEL == "DenseNet121":
    from keras.applications.densenet import DenseNet121 as Model_pre
    from keras.applications.densenet import preprocess_input

# construct the model
# --use u-net model
# ----get the vgg encoder
# ----construct the decoder
def model():
    model = VGG16(weights='imagenet', include_top=False)
    for layer in model.layers:
        

# train method
# --load dataset
# ----train
# ----save checkpoint model
def train():


def predict():


def main():
    train()
    or predict()
