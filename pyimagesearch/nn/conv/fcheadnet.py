# import the necessary packages
from tensorflow.keras.layers import AveragePooling2D
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import BatchNormalization

class FCHeadNet:
  @staticmethod
  def build(baseModel, classes, D):
    # initialize the head model that will be placed on top of
    # the base, then add a FC layer
    headModel = baseModel.output
    headModel = Flatten(name="flatten")(headModel)
    headModel = Dense(D, activation="relu")(headModel)
    headModel = Dropout(0.5)(headModel)

    # add a softmax layer
    headModel = Dense(classes, activation="softmax")(headModel)

    # return the model
    return headModel

class FCHeadNet2:
  @staticmethod
  def build(baseModel, classes, D):
    # initialize the head model that will be placed on top of
    # the base, then add a FC layer
    headModel = baseModel.output
    #headModel = AveragePooling2D(pool_size = (4, 4))(headModel)
    headModel = Flatten(name="flatten")(headModel)
    headModel = Dense(D, activation="relu")(headModel)
    headModel = BatchNormalization()(headModel)
    headModel = Dropout(0.5)(headModel)
    headModel = Dense(D/4, activation="relu")(headModel)
    headModel = BatchNormalization()(headModel)
    headModel = Dropout(0.5)(headModel)

    # add a softmax layer
    headModel = Dense(classes, activation="softmax")(headModel)

    # return the model
    return headModel