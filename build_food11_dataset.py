# USAGE
# 1. python build_food11_dataset.py
# build a new database well organized using food5k_config
# dataset_name/class_label/example_of_class_label.jpg

# 2. python fine_tuning_food11_VGG16.py
# extract the features using VGG16 with the new orgnized dataset

# 3. python train_model_logisitic_regression_datasetcsv_trans_learn.py
# load the features in csv format, evaluate the model. serialize the model and
# close the database

# import the necessary packages
#from config import food5k_config as config
from imutils import paths
import shutil
import os
import argparse


# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", default="../dataset/Food-11",
								help="path to input dataset")
ap.add_argument("-b", "--basedataset", default="../dataset/Food11",
								help="path to new build dataset")
# ap.add_argument("-m", "--model", default="model",
# 								help="path to output model")
args = vars(ap.parse_args())

# loop over the data splits
for split in ("training","evaluation", "validation"):
	# grab all image paths in the current split
	print("[INFO] processing '{} split'...".format(split))
	p = os.path.sep.join([args["dataset"], split])
	imagePaths = list(paths.list_images(p))
	# I prefer to have my dataset on disk organized in the format of:
	# dataset_name/class_label/example_of_class_label.jpg

	classes = ['Bread','Dairy product','Dessert','Egg','Fried food','Meat',
														 'Noodles-Pasta','Rice','Seafood', 'Soup',
						 'Vegetable-Fruit']

	# loop over the image paths
	for imagePath in imagePaths:
		# extract class label from the filename
		filename = imagePath.split(os.path.sep)[-1]
		label = classes[int(filename.split("_")[0])]

		# construct the path to the output directory
		#dirPath = os.path.sep.join([args["basedataset"], split, label])
		dirPath = os.path.sep.join([args["basedataset"], label])
		# if the output directory does not exist, create it
		if not os.path.exists(dirPath):
			print("[INFO] Creating directory for new dataset {}".format(label))
			os.makedirs(dirPath)

		# construct the path to the output image file and copy it
		p = os.path.sep.join([dirPath, filename])
		shutil.copy2(imagePath, p)

# $	tree Food11 --filelimit 15
# Food11
# ├── Bread [994 entries exceeds filelimit, not opening dir]
# ├── Dairy\ product [429 entries exceeds filelimit, not opening dir]
# ├── Dessert [1500 entries exceeds filelimit, not opening dir]
# ├── Egg [986 entries exceeds filelimit, not opening dir]
# ├── Fried\ food [848 entries exceeds filelimit, not opening dir]
# ├── Meat [1325 entries exceeds filelimit, not opening dir]
# ├── Noodles
# │		└── Pasta [440 entries exceeds filelimit, not opening dir]
# ├── Rice [280 entries exceeds filelimit, not opening dir]
# ├── Seafood [855 entries exceeds filelimit, not opening dir]
# ├── Soup [1500 entries exceeds filelimit, not opening dir]
# └── Vegetable
# └── Fruit [709 entries exceeds filelimit, not opening dir]
