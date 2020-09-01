# USAGE
# 1. python build_food11_dataset.py
# build a new database well organized using food5k_config
# dataset_name/class_label/example_of_class_label.jpg

# 2. python fine_tuning_food11_VGG16.py
# change the head of VGG16 with the new organized dataset

# 3. python train_model_logisitic_regression_datasetcsv_trans_learn.py
# load the features in csv format, evaluate the model. serialize the model and
# close the database
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications import InceptionV3
from tensorflow.keras.applications import Xception  # TensorFlow ONLY
from tensorflow.keras.applications import VGG16
from tensorflow.keras.applications import VGG19
import argparse

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", default = "../../PB/datasets/Food11",
                help = "path to input dataset")
ap.add_argument("-o", "--output", default = "../output",
                help = "path to output .png loss/acc plot")
ap.add_argument("-m", "--model", default = "../model",
                help = "path to model .hdf5")
ap.add_argument("-n", "--network", type = str, default = "vgg16",
                help = "name of pre-trained network to use")
ap.add_argument("-e", "--epochs", type = int, default = 50,
                help = "epochs to train")
ap.add_argument("-b", "--batch", type = int, default = 40,
                help = "batch size to train")
ap.add_argument("-op", "--optimizer", type = str, default = "sgd",
                help = "Learning Rate")
ap.add_argument("-lr", "--lr", type = float, default = 2.4e-3,
                help = "Learning Rate")
ap.add_argument("-mm", "--momentum", type = float, default = 0.8,
                help = "Momentum")
ap.add_argument("-nv", "--nesterov", type = bool, default = True,
                help = "Nesterov")
args = vars(ap.parse_args())

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
import os
import numpy as np

# grab the list of images that we'll be describing, then extract
# the class label names from the image paths
print("[INFO] loading images...")
imagePaths = list(paths.list_images(args["dataset"]))
classNames = [p.split(os.path.sep)[-2] for p in imagePaths]
classNames = [str(x) for x in np.unique(classNames)]
print("[INFO] Names of classes {}...".format(classNames))

from pyimagesearch.preprocessing import ImageToArrayPreprocessor
# apply the Keras utility function that correctly rearranges the dimensions
# of the image
from pyimagesearch.preprocessing import AspectAwarePreprocessor
# # resize the image to a fixed size, ignoring the aspect ratio
from pyimagesearch.preprocessing import PatchPreprocessor
# grab the width and height of the image then use these
# dimensions to define the corners of the image based
from pyimagesearch.preprocessing import SimplePreprocessor
# # resize the image to a fixed size, ignoring the aspect ratio
from pyimagesearch.preprocessing import CropPreprocessor
# extract a random crop from the image with the target width
# and height
from pyimagesearch.preprocessing import AddChannelPreprocessor
# subtitutes G and B channel for medianblur and Canny edge map
from pyimagesearch.preprocessing import MeanPreprocessor
# subtract the means for each channel

# initialize the image preprocessors
aap = AspectAwarePreprocessor(224, 224)
iap = ImageToArrayPreprocessor()
pp = PatchPreprocessor(224, 224)
sp = SimplePreprocessor(224, 224)
cp = CropPreprocessor(224, 224)
acp = AddChannelPreprocessor(10, 20)
mp = MeanPreprocessor(1,1,1)

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
spreproc = "aap"
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

# construct the head of the model that will be placed on top of the
# the base model
# headModel = baseModel.output
# headModel = AveragePooling2D(pool_size=(4, 4))(headModel)
# headModel = Flatten(name="flatten")(headModel)
# headModel = Dense(64, activation="relu")(headModel)
# headModel = Dropout(0.5)(headModel)
# headModel = Dense(len(classNames), activation="softmax")(headModel)

# headModel = baseModel.output
# headModel = Flatten(name="flatten")(headModel)
# headModel = Dense(512, activation="relu")(headModel)
# headModel = Dropout(0.5)(headModel)
# headModel = Dense(128, activation="relu")(headModel)
# headModel = Dropout(0.5)(headModel)
# headModel = Dense(len(classNames), activation="softmax")(headModel)

from pyimagesearch.nn.conv import FCHeadNet, FCHeadNet2

# initialize the new head of the network, a set of FC layers
# followed by a softmax classifier
headModel = FCHeadNet2.build(baseModel, len(classNames), 128)

# place the head FC model on top of the base model (this will become
# the actual model we will train)
model = Model(inputs = baseModel.input, outputs = headModel)
model.summary()

# loop over all layers in the base model and freeze them so they
# will *not* be updated during the training process
for layer in baseModel.layers:
  layer.trainable = False

# compile our model (this needs to be done after our setting our
# layers to being non-trainable
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.optimizers import Adam

# initialize the optimizer and model
epochs = args["epochs"]
batch_size = args["batch"]
lr = args["lr"]  # 0.05
momentum = args["momentum"]
nesterov = args["nesterov"]

# define a dictionary that maps model names to their classes
# inside Keras
OPTIMIZERS = {
  "sgd": SGD,
  "rmsprop": RMSprop,
  "adam": Adam
}

# ensure a valid optimizer name was supplied via command line argument
if args["optimizer"] not in OPTIMIZERS.keys():
  raise AssertionError("The --optimizer command line argument should "
                       "be a key in the `OPTIMIZERS` dictionary(SGD,RMSprop,"
                       "Adam)")
if OPTIMIZERS[args["optimizer"]] == SGD:
  opt = SGD(lr = lr, decay = lr / epochs, momentum = momentum,
            nesterov = nesterov)

if OPTIMIZERS[args["optimizer"]] == RMSprop:
  opt = RMSprop(lr = lr)

if OPTIMIZERS[args["optimizer"]] == Adam:
  opt = Adam(lr = lr, decay = lr / epochs)

model.compile(loss = "categorical_crossentropy", optimizer = opt,
              metrics = ["accuracy"])

##### ARCHITECTURE #####
print("[INFO] saving architecture model to file...")
p = [args["output"], "{}_fine_tuning_arch_{}_lr:{}_epochs:{}_batch:{}_{"
                     "}.png".format(
  args["network"],
  args["optimizer"],
  args["lr"],
  args["epochs"],
  args["batch"],
  os.getpid())]
from tensorflow.keras.utils import plot_model

plot_model(model, to_file = os.path.sep.join(p), show_shapes = True, dpi = 600)

# train the head of the network for a few epochs (all other
# layers are frozen) -- this will allow the new FC layers to
# start to become initialized with actual "learned" values
# versus pure random

# Set callback functions to early stop training and save the best model so far.

# After every epoch ModelCheckpoint saves a model to the location
# specified by the filepath parameter.

# If we include only a filename (e.g.models.hdf5) that file will be overridden
# with the latest model every epoch.
#
# If we only wanted to save the best model according to the
# performance of some loss function, we can set save_best_only=True and
# monitor='val_loss' to not override a file if the model has a worse test
# loss than the previous model.
#
# Alternatively, we can save every epoch’s model as its own file by including
# the epoch number and test loss score into the filename itself. For example
# if we set filepath to model_{epoch:02d}_{val_loss:.2f}.hdf5, the name of
# the file containing the model saved after the 11th epoch with a test loss
# value of 0.33 would be model_10_0.35.hdf5 (notice that the epoch number if
# 0-indexed).

from tensorflow.keras.callbacks import ModelCheckpoint

print("[INFO] writting best model to hdf5 file...")
# p = [args["model"], "model_{epoch:02d}_{val_loss:.2f}.hdf5"]
p = [args["model"], "{}_{}_lr:{}_epochs:{}_batch:{}_{}.hdf5".format(
  args["network"],
  args["optimizer"],
  args["lr"],
  args["epochs"],
  args["batch"],
  os.getpid())]
checkpoint = ModelCheckpoint(os.path.sep.join(p), monitor = "val_loss",
                             save_best_only = True,
                             verbose = 1)
callbacks = [checkpoint]

# train the head
print("[INFO] training head...")
H = model.fit(
  aug.flow(trainX, trainY, batch_size = batch_size),
  steps_per_epoch = len(trainX) // batch_size,
  validation_data = (testX, testY),
  validation_steps = len(testX) // batch_size,
  epochs = epochs, callbacks = checkpoint)

print("[INFO] saving training history to file...")
import pickle

p = [args["output"],
     "{}_training_history_{}_lr:{}_epochs:{}_batch:{}_{}.pickle".format(
       args["network"],
       args["optimizer"],
       args["lr"],
       args["epochs"],
       args["batch"],
       os.getpid())]
# save the training history
f = open(os.path.sep.join(p), 'wb')
pickle.dump(H.history, f, pickle.HIGHEST_PROTOCOL)
f.close()

# %%
print("[INFO] loading the best model...")
from tensorflow.keras.models import load_model

# load the best model
p = [args["model"], "{}_{}_lr:{}_epochs:{}_batch:{}_{}.hdf5".format(
  args["network"],
  args["optimizer"],
  args["lr"],
  args["epochs"],
  args["batch"],
  os.getpid())]
model = load_model(os.path.sep.join(p))

# %%
##### EVALUATING #####

print("[INFO] evaluating after initialization (making predictions)...")
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# 1. Evaluate the network after initialization
# make predictions on the testing set
predictions = model.predict(testX, batch_size = batch_size)

print("[INFO] Saving Predictions to file...")
import pickle

p = [args["output"],
     "{}_predictions_{}_lr:{}_epochs:{}_batch:{}_{}.pickle".format(
       args["network"],
       args["optimizer"],
       args["lr"],
       args["epochs"],
       args["batch"],
       os.getpid())]
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

p = [args["output"],
     "{}_predictions_{}_lr:{}_epochs:{}_batch:{}_{}.pickle".format(
       args["network"],
       args["optimizer"],
       args["lr"],
       args["epochs"],
       args["batch"],
       os.getpid())]
f = open(os.path.sep.join(p), 'rb')
predictions = pickle.load(f)
f.close()

report = classification_report(testY.argmax(axis = 1),
                               predictions.argmax(axis = 1),
                               target_names = classNames)
# display Classification Report
print("Learning Ratio = {}\n".format(args["lr"]))
print(report)
print("")

# 3. Confusion Matrix
confmatrix = confusion_matrix(testY.argmax(axis = 1),
                              predictions.argmax(axis = 1))
# display Confusion Matrix
print(confmatrix)

# 4. Saving the classification report and confussion matrix to file
p = [args["output"],
     "{}_clasification_report_{}_lr:{}_epochs:{}_batch:{}_{}.txt".format(
       args["network"],
       args["optimizer"],
       args["lr"],
       args["epochs"],
       args["batch"],
       os.getpid())]

f = open(os.path.sep.join(p), "w")
print("[INFO] Saving Classification Report and Confusion Matrix to file...")
f.write("Learning Ratio = {}\n".format(args["lr"]))
f.write(report)
f.write("")
f.write(str(confmatrix))
f.close()

# 5. Compute the rank-1, rank 3 and rank-5 accuracies
print("[INFO] predicting...")
from pyimagesearch.utils.ranked import rank5_accuracy, rankn_accuracy

(rank1, rank5) = rank5_accuracy(predictions, testY.argmax(axis = 1))
(rank1, rankn) = rankn_accuracy(predictions, testY.argmax(axis = 1), n = 3)

# display the rank-1 rank-3 and rank-5 accuracies
print("rank-1: {:.2f}%".format(rank1 * 100))
print("rank-3: {:.2f}%".format(rankn * 100))
print("rank-5: {:.2f}%".format(rank5 * 100))

# 6. Saving the ranking to file
p = [args["output"], "{}_ranking_{}_lr:{}_epochs:{}_batch:{}_{}.txt".format(
  args["network"],
  args["optimizer"],
  args["lr"],
  args["epochs"],
  args["batch"],
  os.getpid())]
print("[INFO] Saving Ranking information to file...")
f = open(os.path.sep.join(p), "w")
f.write("rank-1: {:.2f}%".format(rank1 * 100))
f.write("rank-3: {:.2f}%".format(rankn * 100))
f.write("rank-5: {:.2f}%".format(rank5 * 100))
f.close()

# %%
##### PLOTTING  SECTION ######

import matplotlib.pyplot as plt

# 1. Plot and Save the histogram of food CLASSES DISTRIBUTION
print("[INFO] Plotting Food11 Data Class Distribution...")
print([list(labels).count(i) for i in classNames])
bin = np.arange(len(classNames))
counts = [list(labels).count(i) for i in classNames]
plt.style.use("ggplot")
plt.barh(bin, counts, align = 'center', alpha = 0.5)
plt.yticks(bin, classNames)
plt.xlabel('Counts')
plt.title('Food11 Data Class Distribution')
p = [args["output"],
     "{}_Food11_Data_Class_Distribution_{}_lr:{}_epochs:{}_batch:{}_{}.png".format(
       args["network"],
       args["optimizer"],
       args["lr"],
       args["epochs"],
       args["batch"],
       os.getpid())]
print("[INFO] Saving Food11 Data Class Distribution plot...")
plt.savefig(os.path.sep.join(p))
plt.show()

# 2. Plot and Save the histogram of food classes distribution TRAINING SET
print("[INFO] Plotting Food11 Data Class Distribution for train set...")
print([list(trainYa).count(i) for i in list(bin)])
bin = np.arange(len(classNames))
counts = [list(trainYa).count(i) for i in list(bin)]
plt.style.use("ggplot")
plt.barh(bin, counts, align = 'center', alpha = 0.5)
plt.yticks(bin, classNames)
plt.xlabel('Counts')
plt.title('Food11 Data Class Distribution Train Set')
p = [args["output"],
     "{}_Food11_Data_Class_Distribution_for_train_set_{}_lr:{}_epochs:{}_batch:{}_{}.png".format(
       args["network"],
       args["optimizer"],
       args["lr"],
       args["epochs"],
       args["batch"],
       os.getpid())]
print("[INFO] Saving Food11 Data Class Distribution for train set plot...")
plt.savefig(os.path.sep.join(p))
plt.show()

# 3. Plot and Save the histogram of food classes distribution TESTING SET
print("[INFO] Plotting Food11 Data Class Distribution for test set...")
print([list(testYa).count(i) for i in list(bin)])
bin = np.arange(len(classNames))
counts = [list(testYa).count(i) for i in list(bin)]
plt.style.use("ggplot")
plt.barh(bin, counts, align = 'center', alpha = 0.5)
plt.yticks(bin, classNames)
plt.xlabel('Counts')
plt.title('Food11 Data Class Distribution Test Set')
p = [args["output"],
     "{}_Food11_Data_Class_Distribution_for_test_set_{}_lr:{}_epochs:{}_batch:{}_{}.png".format(
       args["network"],
       args["optimizer"],
       args["lr"],
       args["epochs"],
       args["batch"],
       os.getpid())]
print("[INFO] Saving Food11 Data Class Distribution for test set plot...")
plt.savefig(os.path.sep.join(p))
plt.show()

# # plot the histogram of food classes distribution VALIDATION SET
# print("[INFO] Food11 Data Class Distribution for validation set...")
# print([list(valYa).count(i) for i in list(bin)])
# bin = np.arange(len(classNames))
# counts = [list(valYa).count(i) for i in list(bin)]
# plt.style.use("ggplot")
# plt.barh(bin, counts, align='center', alpha=0.5)
# plt.yticks(bin, classNames)
# plt.xlabel('Counts')
# plt.title('Food11 Data Class Distribution Validation Set')
# p = [args["output"], "Food11_Data_Class_Distribution_for_val_set_{}_{"
#                      "}.png".format(spreproc,
#                                     os.getpid())]
# plt.savefig(os.path.sep.join(p))
# plt.show()


from matplotlib import image as img
from random import randint

# 4. Plot and Save a few sample images from food11 dataset
plt.figure(figsize = (8, 8))
k = 0
for i in range(0, 4):
  for j in range(0, 4):
    image = img.imread(imagePaths[randint(0, len(imagePaths))])
    plt.subplot2grid((4, 4), (i, j))
    plt.imshow(image)
    k = k + 1
# show the plot
plt.title('Food11 Data Class Samples')
p = [args["output"],
     "{}_Food11_Data_Class_Samples_{}_lr:{}_epochs:{}_batch:{}_{}.png".format(
       args["network"],
       args["optimizer"],
       args["lr"],
       args["epochs"],
       args["batch"],
       os.getpid())]
print("[INFO] Saving Sample Images plot...")
plt.savefig(os.path.sep.join(p))
plt.show()

# %%
# 5. PLot and Save the plotting from training history
s_epochs = 0
f_epochs = args["epochs"]  # default value
np_range = np.arange(s_epochs, f_epochs)
import matplotlib.pyplot as plt

print("[INFO] loading training history for plotting...")
# Loading training history file
import pickle

p = [args["output"],
     "{}_training_history_{}_lr:{}_epochs:{}_batch:{}_{}.pickle".format(
       args["network"],
       args["optimizer"],
       args["lr"],
       args["epochs"],
       args["batch"],
       os.getpid())]
f = open(os.path.sep.join(p), 'rb')
history = pickle.load(f)
f.close()

# plot the training loss and accuracy
plt.style.use("ggplot")
plt.figure()
plt.plot(list(np_range), history["loss"][s_epochs:f_epochs],
         label = "train_loss")
plt.plot(list(np_range), history["val_loss"][s_epochs:f_epochs],
         label = "val_loss")
plt.plot(list(np_range), history["accuracy"][s_epochs:f_epochs],
         label = "train_acc")
plt.plot(list(np_range), history["val_accuracy"][s_epochs:f_epochs],
         label = "val_acc")
# plt.plot(history["loss"], label="train_loss")
# plt.plot(history["val_loss"], label="val_loss")
# plt.plot(history["accuracy"], label="train_acc")
# plt.plot(history["val_accuracy"], label="val_acc")
plt.title(
  "{}_Training Loss and Accuracy with_{}_lr:{}_epochs:{}_batch:{}_{}".format(
    args["network"],
    args["optimizer"],
    args["lr"],
    args["epochs"],
    args["batch"],
    os.getpid()))
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend()
p = [args["output"],
     "{}_fine_tuning_history_{}_lr:{}_epochs:{}_batch:{}_{}.png".format(
       args["network"],
       args["optimizer"],
       args["lr"],
       args["epochs"],
       args["batch"],
       os.getpid())]
print("[INFO] Saving Training Loss and Accuracy History plot...")
plt.savefig(os.path.sep.join(p))
plt.show()

print("\nAccuracy on Test Data: ", accuracy_score(testY.argmax(axis = 1), predictions.argmax(axis = 1)))
print("\nNumber of correctly identified images: ", accuracy_score(testY.argmax(axis = 1),
                                                                  predictions.argmax(axis = 1), normalize=False),"from:",len(testY.argmax(axis = 1)), "\n")