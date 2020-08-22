# Transfer Learning using Feature Extraction from Trained model: Food Images Classification

The idea of transfer learning comes from a curious phenomenon that many deep neural networks trained on natural images learn similar features. These are texture, corners, edges and color blobs in the initial layers. Such initial-layer features appear not to specific to a particular data-set or task but are general in that they are applicable to many data-sets and tasks. These standard features found on the initial layers seems to occur regardless of the exact cost function and natural image data-set. We call these initial-layer features general and can be transferred for learning specific data-set.

### Introduction

In transfer learning, we first train a base network on a base data-set and task, and then we transfer the learned features, to a second target network to be trained on a target data-set and task. This process will tend to work if the features are general, that is, suitable to both base and target tasks, instead of being specific to the base task.

Earlier, I have penned down couple of blog-post to train entire Convolution
 Network (CNN) model on sufficiently large data-set. You can read posts [here](https://appliedmachinelearning.blog/2018/03/24/achieving-90-accuracy-in-object-recognition-task-on-cifar-10-dataset-with-keras-convolutional-neural-networks/) and [here](https://appliedmachinelearning.blog/2018/11/28/demonstration-of-facial-emotion-recognition-on-real-time-video-using-cnn-python-keras/). In practice, very few people train an entire CNN
  from scratch
  because it is relatively rare to have a data-set of sufficient size. Instead, it is common to pre-train a convolution neural network (CNN) on a very large data-set (e.g. ImageNet data-set, which contains 1.2 million images with 1000 categories), and then use the pre-trained model either as an initialization or a fixed feature extractor for the task of interest.

There are two ways to do transfer learning.

1. Feature Extraction from pre-trained model and then training a
 classifier on
 top of it.
2. Fine tuning the pre-trained model keeping learnt weights as initial
 parameters.
 
This blog-post showcases the implementation of transfer learning using the first way which is “Feature Extraction from pre-trained model and training a classifier using extracted features”.

###Problem Description

In this blog-post, We will use a data-set containing 16643 food images grouped in 11 major food categories for transfer learning demonstration. This is a food image classification task. The 11 categories are:

- Bread
- Dairy product
- Dessert
- Egg
- Fried food
- Meat
- Noodles/Pasta
- Rice
- Seafood
- Soup
- Vegetable/Fruit

The [Food-11 dataset](https://mmspg.epfl.ch/downloads/food-image-datasets/) is divided in three parts: training, validation and
 evaluation. The naming convention is used, where ID 0-10 refers to the 11
  food categories respectively. The data-set can be downloaded from [here](https://www.dropbox.com/s/sh5yt160xzqjkk0/Food-11.zip?dl=0).

Lets start with python codes on transfer learning with feature extraction technique.

1. Import Library
2. Reading Data
3. Create labels
4. Train, Validation and Test Distribution
5. Sample Images
6. Features Extraction
7. CNN Model Training : Baseline
8. Test Evaluation
9. Transfer Learning CNN : VGG16 Features
10. Test Evaluation
11. Logistic Regression: VGG16 Features
12. Test Evaluation
