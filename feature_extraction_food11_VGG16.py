import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt;

# Importing sklearn libraries
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression

# Importing hypopt library for grid search
from hypopt import GridSearch

# Importing Keras libraries
from keras.utils import np_utils
from keras.models import Sequential
# from keras.applications import VGG16
from keras.applications import imagenet_utils
from keras.callbacks import ModelCheckpoint
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.layers import Dense, Conv2D, MaxPooling2D
from keras.layers import Dropout, Flatten, GlobalAveragePooling2D

# import warnings
# warnings.filterwarnings('ignore')

import argparse

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", default="../../PB/datasets/Food11",
                help="path to input dataset")
ap.add_argument("-o", "--output", default="../output",
                help="path to output .png loss/acc plot")
ap.add_argument("-m", "--model", default="../model",
                help="path to model .hdf5")
ap.add_argument("-n", "--network", type=str, default="vgg16",
                help="name of pre-trained network to use")
ap.add_argument("-e", "--epochs", type=int, default=5,
                help="epochs to train")
ap.add_argument("-b", "--batch", type=int, default=32,
                help="batch size to train")
ap.add_argument("-op", "--optimizer", type=str, default="adam",
                help="Learning Rate")
ap.add_argument("-lr", "--lr", type=float, default=1e-2,
                help="Learning Rate")
ap.add_argument("-mm", "--momentum", type=float, default=0.9,
                help="Momentum")
ap.add_argument("-nv", "--nesterov", type=bool, default=False,
                help="Nesterov")
args = vars(ap.parse_args())

from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications import InceptionV3
from tensorflow.keras.applications import Xception  # TensorFlow ONLY
from tensorflow.keras.applications import VGG16
from tensorflow.keras.applications import VGG19
# define a dictionary that maps model names to their classes
# inside Keras
NETWORKS = {
  "vgg16": VGG16,
  "vgg19": VGG19,
  "inception": InceptionV3,
  "xception": Xception,  # TensorFlow ONLY
  "resnet": ResNet50
}

# ensure a valid model name was supplied via command line argument
if args["network"] not in NETWORKS.keys():
  raise AssertionError("The --network command line argument should "
                       "be a key in the `NETWORKS` dictionary(vgg16,vgg19,"
                       "inception,xception,resnet)")

from imutils import paths
import random
import os
import numpy as np
# grab the list of images that we'll be describing
print("[INFO] loading images...")
imagePaths = list(paths.list_images(args["dataset"]))
# randomly shuffle the image paths and then extract the class
# labels from the file paths
random.shuffle(imagePaths)
classNames = [pt.split(os.path.sep)[-2] for pt in imagePaths]
classNames = [str(x) for x in np.unique(classNames)]
print("[INFO] Names of classes {}...".format(classNames))

from pyimagesearch.preprocessing import ImageToArrayPreprocessor
from pyimagesearch.preprocessing import AspectAwarePreprocessor
from pyimagesearch.preprocessing import PatchPreprocessor
from pyimagesearch.preprocessing import SimplePreprocessor
from pyimagesearch.preprocessing import CropPreprocessor
from pyimagesearch.preprocessing import AddChannelPreprocessor

# initialize the image preprocessors
aap = AspectAwarePreprocessor(224, 224)
iap = ImageToArrayPreprocessor()
pp = PatchPreprocessor(224, 224)
sp = SimplePreprocessor(224, 224)
cp = CropPreprocessor(224, 224)
acp = AddChannelPreprocessor(10, 20)

if args["model"] in ("inception", "xception"):
  aap = AspectAwarePreprocessor(299, 299)
  iap = ImageToArrayPreprocessor()
  pp = PatchPreprocessor(299, 299)
  sp = SimplePreprocessor(299, 299)
  cp = CropPreprocessor(299, 299)
  acp = AddChannelPreprocessor(10, 20)

from pyimagesearch.datasets import SimpleDatasetLoader

# load the image (data) and extract the class label assuming
# that our path has the following format:
# /path/to/dataset/{class}/{image}.jpg
# print("[INFO] loading {}".format(imagePath))
# load the dataset from disk then scale the raw pixel intensities to the
# range [0, 1]
spreproc = "sp"
sdl = SimpleDatasetLoader(preprocessors = [sp])
(data, labels) = sdl.load(imagePaths, verbose = 500)
data = data.astype("float") / 255.0

# data = []
# labels = []
# import cv2
# # Creating Data (X) y labels (y) from imagepaths
# # loop over the image paths
# for imagePath in imagePaths:
#   # extract the class label from the filename
#   label = imagePath.split(os.path.sep)[-2]
#
#   # load the image, swap color channels, and resize it to be a fixed
#   # 224x224 pixels while ignoring aspect ratio
#   image = cv2.imread(imagePath)
#   image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#   image = cv2.resize(image, (224, 224))
#   # if we are using the InceptionV3 or Xception networks, then we
#   # need to set the input shape to (299x299) [rather than (224x224)]
#   # and use a different image processing function
#   if args["model"] in ("inception", "xception"):
#     image = cv2.resize(image, (299, 299))
#
#   # update the data and labels lists, respectively
#   data.append(image)
#   labels.append(label)

## convert the data and labels to NumPy arrays while scaling the pixel
## intensities to the range [0, 255]
# data = np.array(data) / 255.0
# labels = np.array(labels)

from sklearn.model_selection import train_test_split

# # partition the data into training and testing splits using 60% of
# # the data for training and the remaining 20% for testing and 20% validation
# (trainX, testX, trainYa, testYa) = train_test_split(data, labels,
#                                                     test_size=0.2,
#                                                     random_state=42)
#
# (trainX, valX, trainYa, valYa) = train_test_split(trainX, trainYa,
#                                                     test_size=0.25,
#                                                     random_state=42)


# partition the data into training and testing splits using 75% of
# the data for training and the remaining 20% for testing and 20% validation
(trainX, testX, trainYa, testYa) = train_test_split(data, labels,
                                                    test_size = 0.25,
                                                    random_state = 42)

from sklearn.preprocessing import LabelBinarizer # [A A B B C D] = [[0 0 0
# 1] [0 0 0 1] [0 0 1 0] [0 0 1 0][0 1 0 0] [1 0 0 0]]
from sklearn.preprocessing import LabelEncoder # [A A B B C D] = [1 1 2 2 3 4]

# # convert the labels from integers to vectors
trainY = LabelBinarizer().fit_transform(trainYa)
trainYa = LabelEncoder().fit_transform(trainYa)  # for plotting
testY = LabelBinarizer().fit_transform(testYa)
testYa = LabelEncoder().fit_transform(testYa)  # for plotting

from sklearn.preprocessing import LabelEncoder  # [A A B B C D] = [1 1 2 2 3 4]
from tensorflow.keras.utils import to_categorical  # [1 1 2 2 3 4] = [[0 0 0
# 1] [0 0 0 1] [0 0 1 0][0 0 1 0][0 1 0 0] [1 0 0 0]]

# convert the labels from integers to vectors
# trainYa = LabelEncoder().fit_transform(trainYa)
# trainY = to_categorical(trainYa)
# testYa = LabelEncoder().fit_transform(testYa)
# testY = to_categorical(testYa)
# valY = LabelEncoder().fit_transform(valYa)

from tensorflow.keras.preprocessing.image import ImageDataGenerator

# construct the image generator for data augmentation
aug = ImageDataGenerator(rotation_range = 15, width_shift_range = 0.1,
                         height_shift_range = 0.1, shear_range = 0.2,
                         zoom_range = 0.2,
                         horizontal_flip = True, fill_mode = "nearest")

from tensorflow.keras.layers import AveragePooling2D
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model

# load the MODELS[args["model"]] network, ensuring the head FC layer sets are left
# off
baseModel = NETWORKS[args["network"]](weights = "imagenet", include_top = False,
                                      input_tensor = Input(
                                        shape = (224, 224, 3)))
if args["network"] in ("inception", "xception"):
  baseModel = NETWORKS[args["network"]](weights = "imagenet",
                                        include_top = False,
                                        input_tensor = Input(
                                          shape = (299, 299, 3)))

baseModel.summary()

#### Extract features from baseModel
#%%
trainX_features = baseModel.predict(trainX, batch_size = args["batch"])
trainX_features_flatten = trainX_features.reshape((trainX_features.shape[0],
                                                 7 * 7 * 512))
testX_features = baseModel.predict(testX, batch_size = args["batch"])
testX_features_flatten = testX_features.reshape((testX_features.shape[0],
                                                 7 * 7 * 512))

### Logistic Regression ###
# we used flattened extracted features from last maxpooling layer of VGG16 as
# an input to Logistic regression (LR) classifier. The validation set is used
# for fine tuning the hyper parameters of the logistic regression model. It
# was found that regurlarized logistic model performs better than default LR
# model. Here, we use hypopt pypi package for performing grid search on
# hyper-parameters.

# param_grid = [{'C': [0.1, 1, 10], 'solver': ['newton-cg', 'lbfgs']}]
#
# # Grid-search all parameter combinations using a validation set.
# opt = GridSearch(
#   model = LogisticRegression(
#     class_weight = 'balanced', multi_class = "auto",
#     max_iter = 1000, random_state = 1),
#   param_grid = param_grid)
#
# from sklearn import preprocessing
# opt.fit(preprocessing.scale(trainX_features_flatten), trainYa, scoring =
# 'accuracy')
# print(opt.get_best_params())


#%%
# train the model
solver = "newton-cg"
C = 1
from sklearn.linear_model import LogisticRegression
print("[INFO] training model...")
model = LogisticRegression(solver = solver , multi_class = "auto", max_iter =
10000, C = C)
model.fit(trainX_features_flatten, trainYa)

print("[INFO] saving model to file...")
import pickle

# save the model to disk
p = [args["output"],'logistic_regression_model.pickle']
f = open(os.path.sep.join(p), 'wb')
pickle.dump(model, f, pickle.HIGHEST_PROTOCOL)
f.close()

# some time later...

print("[INFO] loading model from file...")
# load the model from disk
p = [args["output"],'logistic_regression_model.pickle']
model = pickle.load(open(os.path.sep.join(p), 'rb'))

## %%
##### EVALUATING #####

print("[INFO] evaluating after initialization (making predictions)...")
from sklearn.metrics import classification_report, confusion_matrix, \
  accuracy_score

# 1. Evaluate the network after initialization
# make predictions on the testing set

model.score(testX_features_flatten, testYa)
predictions = model.predict(testX_features_flatten)

print("[INFO] Saving Predictions to file...")
import pickle

p = [args["output"], "predictions_logistic_regression_model.pickle"]
# save the predicitons
f = open(os.path.sep.join(p), 'wb')
pickle.dump(predictions, f, pickle.HIGHEST_PROTOCOL)
f.close()

# 2. Classification report
# for each image in the testing set we need to find the index of the
# label with corresponding largest predicted probability

#
# Loading training history file
print("[INFO] Loading Predictions from file...")
import pickle

p = [args["output"],"predictions_logistic_regression_model.pickle"]
f = open(os.path.sep.join(p), 'rb')
predictions = pickle.load(f)
f.close()

report = classification_report(testYa,predictions, target_names = classNames)
# display Classification Report
print(report, "\n")
print("")

# 3. Confusion Matrix
confmatrix = confusion_matrix(testYa, predictions, labels=range(0, 11))
# display Confusion Matrix
print(confmatrix)

# 4. Saving the classification report and confussion matrix to file
p = [args["output"], "clasification_report_logistic_regression_model.txt"]

f = open(os.path.sep.join(p), "w")
print("[INFO] Saving Classification Report and Confusion Matrix to file...")
f.write(report)
f.write("")
f.write(str(confmatrix))
f.close()

# 5. Compute the rank-1, rank 3 and rank-5 accuracies
print("[INFO] predicting...")
from pyimagesearch.utils.ranked import rank5_accuracy, rankn_accuracy

(rank1, rank5) = rank5_accuracy(to_categorical(predictions), testY.argmax(axis = 1))
(rank1, rankn) = rankn_accuracy(to_categorical(predictions), testY.argmax(axis = 1), n = 3)

# display the rank-1 rank-3 and rank-5 accuracies
print("rank-1: {:.2f}%".format(rank1 * 100))
print("rank-3: {:.2f}%".format(rankn * 100))
print("rank-5: {:.2f}%".format(rank5 * 100))

# 6. Saving the ranking to file
p = [args["output"], "ranking_report_logistic_regression_model.txt"]
print("[INFO] Saving Ranking information to file...")
f = open(os.path.sep.join(p), "w")
f.write("rank-1: {:.2f}%".format(rank1 * 100))
f.write("rank-3: {:.2f}%".format(rankn * 100))
f.write("rank-5: {:.2f}%".format(rank5 * 100))
f.close()

print("\nAccuracy on Test Data: ", accuracy_score(testYa, predictions))
print("\nNumber of correctly identified images: ",
      accuracy_score(testYa, predictions, normalize=False),"from:",
      len(testYa), "\n")