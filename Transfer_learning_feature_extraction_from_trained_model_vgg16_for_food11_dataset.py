import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt; 
 
# Importing sklearn libraries
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score
 
# Importing hypopt library for grid search
from hypopt import GridSearch
 
# Importing Keras libraries
from keras.utils import np_utils
from keras.models import Sequential
from keras.applications import VGG16
from keras.applications import imagenet_utils
from keras.callbacks import ModelCheckpoint
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.layers import Dense, Conv2D, MaxPooling2D
from keras.layers import Dropout, Flatten, GlobalAveragePooling2D
 
import warnings
warnings.filterwarnings('ignore')

import argparse

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", default="../dataset/Food-11",
                help="path to input dataset")
ap.add_argument("-o", "--output", default="output",
                help="path to output .png loss/acc plot")
ap.add_argument("-m", "--model", default="model",
                help="path to output model")
args = vars(ap.parse_args())


# grab the list of images that we'll be describing, then extract
# the class label names from the image paths
print("[INFO] loading images...")
p = [args["dataset"],"training" ]
train = [os.path.join(os.path.sep.join(p),img) for img in os.listdir(
  os.path.sep.join(p))]
p = [args["dataset"],"validation" ]
val = [os.path.join(os.path.sep.join(p),img) for img in os.listdir(
  os.path.sep.join(p))]
p = [args["dataset"],"evaluation" ]
test = [os.path.join(os.path.sep.join(p),img) for img in os.listdir(
  os.path.sep.join(p))]

from imutils import paths
imagePaths = list(paths.list_images(args["dataset"]))
imageFiles = [pt.split(os.path.sep)[-1] for pt in imagePaths]
classNames = [pt.split("_")[0] for pt in imageFiles]
classNames = [str(x) for x in np.unique(classNames)]


#
# train_y = [int(img.split("/")[-1].split("_")[0]) for img in train]
# val_y = [int(img.split("/")[-1].split("_")[0]) for img in val]
# test_y = [int(img.split("/")[-1].split("_")[0]) for img in test]
# num_classes = 11
#
# # Convert class labels in one hot encoded vector
# y_train = np_utils.to_categorical(train_y, num_classes)
# y_val = np_utils.to_categorical(val_y, num_classes)
# y_test = np_utils.to_categorical(test_y, num_classes)
#
# print("Training data available in 11 classes")
# print([train_y.count(i) for i in range(0,11)])
#
# food_classes = ('Bread','Dairy product','Dessert','Egg','Fried food','Meat',
#            'Noodles/Pasta','Rice','Seafood', 'Soup', 'Vegetable/Fruit')
#
# y_pos = np.arange(len(food_classes))
# counts = [train_y.count(i) for i in range(0,11)]
#
# plt.barh(y_pos, counts, align='center', alpha=0.5)
# plt.yticks(y_pos, food_classes)
# plt.xlabel('Counts')
# plt.title('Train Data Class Distribution')
# plt.show()