# USAGE
# 1. python build_food11_dataset.py
# build a new database well organized using food5k_config
# dataset_name/class_label/example_of_class_label.jpg

# 2. python fine_tuning_food11_VGG16.py
# change the head of VGG16 with the new orgnized dataset

# 3. python train_model_logisitic_regression_datasetcsv_trans_learn.py
# load the features in csv format, evaluate the model. serialize the model and
# close the database


import argparse

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", default = "../dataset/Food11",
                help = "path to input dataset")
ap.add_argument("-o", "--output", default = "output",
                help = "path to output .png loss/acc plot")
ap.add_argument("-m", "--model", default = "model",
                help = "path to output model")
args = vars(ap.parse_args())

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

from pyimagesearch.datasets import SimpleDatasetLoader

# load the image (data) and extract the class label assuming
# that our path has the following format:
# /path/to/dataset/{class}/{image}.jpg
# print("[INFO] loading {}".format(imagePath))
# load the dataset from disk then scale the raw pixel intensities
# to the range [0, 1]
spreproc = "aap_iap"
sdl = SimpleDatasetLoader(preprocessors = [aap, iap])
(data, labels) = sdl.load(imagePaths, verbose = 500)
data = data.astype("float") / 255.0

from sklearn.model_selection import train_test_split
# # partition the data into training and testing splits using 60% of
# # the data for training and the remaining 20% for testing and 20% validation
# (trainX, testX, trainYa, testYa) = train_test_split(data, labels,
#                                                     test_size=0.2, random_state=42)
#
# (trainX, valX, trainYa, valYa) = train_test_split(trainX, trainYa,
#                                                     test_size=0.25, random_state=42)


# partition the data into training and testing splits using 75% of
# the data for training and the remaining 20% for testing and 20% validation
(trainX, testX, trainYa, testYa) = train_test_split(data, labels,
                                                    test_size = 0.25,
                                                    random_state = 42)

from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical

# convert the labels from integers to vectors
trainYa = LabelEncoder().fit_transform(trainYa)
trainY = to_categorical(trainYa)
testYa = LabelEncoder().fit_transform(testYa)
testY = to_categorical(testYa)
# valY = LabelEncoder().fit_transform(valYa)



from tensorflow.keras.preprocessing.image import ImageDataGenerator

# construct the image generator for data augmentation
aug = ImageDataGenerator(rotation_range = 15, width_shift_range = 0.1,
                         height_shift_range = 0.1, shear_range = 0.2,
                         zoom_range = 0.2,
                         horizontal_flip = True, fill_mode = "nearest")

from tensorflow.keras.applications import VGG16
from tensorflow.keras.layers import AveragePooling2D
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model

# load the VGG16 Network and initialize the label encoder
print("[INFO] loading network...")
baseModel = VGG16(weights = "imagenet", include_top = False,
                  input_tensor = Input(shape = (224, 224, 3)))

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

p = [args["output"], "VGG16_fine_tuning_arch_{}_{}.png".format(spreproc,
                                                               os.getpid())]
print("[INFO] writting architecture model...")
from tensorflow.keras.utils import plot_model

plot_model(model, to_file = os.path.sep.join(p), show_shapes = True, dpi = 600)

# compile our model (this needs to be done after our setting our
# layers to being non-trainable
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.optimizers import Adam

print("[INFO] compiling model...")
epochs = 25
batch_size = 128
# opt = SGD(lr=0.0001, decay=0.0001 / epochs, momentum=0.9, nesterov=True)
# opt = RMSprop(lr=0.0001)
opt = Adam(lr = 1e-2, decay = 1e-2 / epochs)
model.compile(loss = "categorical_crossentropy", optimizer = opt,
              metrics = ["accuracy"])
spreproc ="Adam_1e-2_batch_128"

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

# p = [args["model"], "model_{epoch:02d}_{val_loss:.2f}.hdf5"]
p = [args["model"], "model_{}_{}.hdf5".format(spreproc,
                                              os.getpid())]
checkpoint = ModelCheckpoint(os.path.sep.join(p), monitor = "val_loss",
                             save_best_only = True,
                             verbose = 1)
callbacks = [checkpoint]

print("[INFO] training head...")
H = model.fit(
  aug.flow(trainX, trainY, batch_size = batch_size),
  steps_per_epoch = len(trainX) // batch_size,
  validation_data = (testX, testY),
  validation_steps = len(testX) // batch_size,
  epochs = epochs, callbacks = checkpoint)

print("[INFO] saving traning history to file...")
import pickle
p = [args["output"], "training_history_{}_{}.pickle".format(spreproc,
                                                            os.getpid())]
# save the training history
f = open(os.path.sep.join(p), 'wb')
pickle.dump(H.history, f, pickle.HIGHEST_PROTOCOL)
f.close()

# %%
from tensorflow.keras.models import load_model

# load the best model
p = [args["model"], "model_{}_{}.hdf5".format(spreproc,
                                              os.getpid())]
model = load_model(os.path.sep.join(p))

# %%
##### EVALUATING #####

from sklearn.metrics import classification_report, confusion_matrix

# 1. Evaluate the network after initialization
print("[INFO] evaluating after initialization (making predictions)...")
# make predictions on the testing set
predictions = model.predict(testX, batch_size = batch_size)


# 2. Classification report
# for each image in the testing set we need to find the index of the
# label with corresponding largest predicted probability
report = classification_report(testY.argmax(axis = 1),
                               predictions.argmax(axis = 1),
                               target_names = classNames)
# display Classification Report
print(report)
print("")

# 3. Confusion Matrix
confmatrix = confusion_matrix(testY.argmax(axis = 1),
                              predictions.argmax(axis = 1))
# display Confusion Matrix
print(confmatrix)

# 4. Saving the classification report and confussion matrix to file
p = [args["output"], "clasification_report_{}_{}.txt".format(spreproc,
                                                             os.getpid())]
f = open(os.path.sep.join(p), "w")
print("[INFO] Saving Calssification Report and Confusion Matrix to file...")
f.write(report)
f.write("")
f.write(str(confmatrix))
f.close()


from pyimagesearch.utils.ranked import rank5_accuracy, rankn_accuracy

# 5. Compute the rank-1, rank 3 and rank-5 accuracies
print("[INFO] predicting...")
(rank1, rank5) = rank5_accuracy(predictions, testY.argmax(axis = 1))
(rank1, rankn) = rankn_accuracy(predictions, testY.argmax(axis = 1), n = 3)

# display the rank-1 rank-3 and rank-5 accuracies
print("rank-1: {:.2f}%".format(rank1 * 100))
print("rank-3: {:.2f}%".format(rankn * 100))
print("rank-5: {:.2f}%".format(rank5 * 100))

# 6. Saving the ranking to file
p = [args["output"], "ranking_{}_{}.txt".format(spreproc,
                                                os.getpid())]
print("[INFO] Saving Ranking information to file...")
f = open(os.path.sep.join(p), "w")
f.write("rank-1: {:.2f}%".format(rank1 * 100))
f.write("rank-3: {:.2f}%".format(rankn * 100))
f.write("rank-5: {:.2f}%".format(rank5 * 100))
f.close()


#%%
##### PLOTTING  SECTION ######

import matplotlib.pyplot as plt

# 1. Plot and Save the histogram of food CLASSES DISTRIBUTION
print("[INFO] Food11 Data Class Distribution...")
print([list(labels).count(i) for i in classNames])
bin = np.arange(len(classNames))
counts = [list(labels).count(i) for i in classNames]
plt.style.use("ggplot")
plt.barh(bin, counts, align = 'center', alpha = 0.5)
plt.yticks(bin, classNames)
plt.xlabel('Counts')
plt.title('Food11 Data Class Distribution')
p = [args["output"], "Food11_Data_Class_Distribution_{}_{}.png".format(
  spreproc, os.getpid())]
print("[INFO] Saving Food11 Data Class Distribution plot...")
plt.savefig(os.path.sep.join(p))
plt.show()

# 2. Plot and Save the histogram of food classes distribution TRAINING SET
print("[INFO] Food11 Data Class Distribution for train set...")
print([list(trainYa).count(i) for i in list(bin)])
bin = np.arange(len(classNames))
counts = [list(trainYa).count(i) for i in list(bin)]
plt.style.use("ggplot")
plt.barh(bin, counts, align = 'center', alpha = 0.5)
plt.yticks(bin, classNames)
plt.xlabel('Counts')
plt.title('Food11 Data Class Distribution Train Set')
p = [args["output"],
     "Food11_Data_Class_Distribution_for_train_set_{}_{}.png".format(spreproc,
                                                                     os.getpid())]
print("[INFO] Saving Food11 Data Class Distribution for train set plot...")
plt.savefig(os.path.sep.join(p))
plt.show()

# 3. Plot and Save the histogram of food classes distribution TESTING SET
print("[INFO] Food11 Data Class Distribution for test set...")
print([list(testYa).count(i) for i in list(bin)])
bin = np.arange(len(classNames))
counts = [list(testYa).count(i) for i in list(bin)]
plt.style.use("ggplot")
plt.barh(bin, counts, align = 'center', alpha = 0.5)
plt.yticks(bin, classNames)
plt.xlabel('Counts')
plt.title('Food11 Data Class Distribution Test Set')
p = [args["output"],
     "Food11_Data_Class_Distribution_for_test_set_{}_{}.png".format(spreproc,
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
p = [args["output"], "Food11_Data_Class_Samples_{}_{}.png".format(spreproc,
                                                                  os.getpid())]
print("[INFO] Saving Sample Images plot...")
plt.savefig(os.path.sep.join(p))
plt.show()

# %%
# 5. PLot and Save the plotting from training history
s_epochs = 0
f_epochs = epochs #default value
np_range = np.arange(s_epochs, f_epochs)
import matplotlib.pyplot as plt

print("[INFO] loading training history for plotting...")
# Loading training history file
import pickle
p = [args["output"], "training_history_{}_{}.pickle".format(spreproc,
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
plt.title("Training Loss and Accuracy with_{}_{}".format(spreproc, os.getpid()))
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend()
p = [args["output"], "VGG16_fine_tuning_history_{}_{}.png".format(spreproc,
                                                                  os.getpid())]
print("[INFO] Saving Training Loss and Accuracy History plot...")
plt.savefig(os.path.sep.join(p))
plt.show()
