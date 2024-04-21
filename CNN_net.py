# 导入所需模块
from keras.models import Sequential
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.initializers import TruncatedNormal
from keras.layers.core import Activation
from keras.layers.core import Flatten
from keras.layers.core import Dropout
from keras.layers.core import Dense
from keras import backend as K


class SimpleVGGNet:
    @staticmethod
    def build(width, height, depth, classes):
        model = Sequential()
        model1 = Sequential()
        inputShape = (height, width, depth)
        chanDim = -1

        if K.image_data_format() == "channels_first":
            inputShape = (depth, height, width)
            chanDim = 1

        # CONV => RELU => POOL
        # model.add(Conv2D(32, (3, 3), padding="same",
        #     input_shape=inputShape,kernel_initializer=TruncatedNormal(mean=0.0, stddev=0.01)))
        # model.add(Activation("relu"))
        # model.add(BatchNormalization(axis=chanDim))
        # model.add(MaxPooling2D(pool_size=(2, 2)))
        # #model.add(Dropout(0.25))
        #
        # # (CONV => RELU) * 2 => POOL
        # model.add(Conv2D(64, (3, 3), padding="same",kernel_initializer=TruncatedNormal(mean=0.0, stddev=0.01)))
        # model.add(Activation("relu"))
        # model.add(BatchNormalization(axis=chanDim))
        # model.add(Conv2D(64, (3, 3), padding="same",kernel_initializer=TruncatedNormal(mean=0.0, stddev=0.01)))
        # model.add(Activation("relu"))
        # model.add(BatchNormalization(axis=chanDim))
        # model.add(MaxPooling2D(pool_size=(2, 2)))
        # #model.add(Dropout(0.25))
        # # (CONV => RELU) * 3 => POOL
        # model.add(Conv2D(128, (3, 3), padding="same",kernel_initializer=TruncatedNormal(mean=0.0, stddev=0.01)))
        # model.add(Activation("relu"))
        # model.add(BatchNormalization(axis=chanDim))
        # model.add(Conv2D(128, (3, 3), padding="same",kernel_initializer=TruncatedNormal(mean=0.0, stddev=0.01)))
        # model.add(Activation("relu"))
        # model.add(BatchNormalization(axis=chanDim))
        # model.add(Conv2D(128, (3, 3), padding="same",kernel_initializer=TruncatedNormal(mean=0.0, stddev=0.01)))
        # model.add(Activation("relu"))
        # model.add(BatchNormalization(axis=chanDim))
        # model.add(MaxPooling2D(pool_size=(2, 2)))
        # #model.add(Dropout(0.25))
        # # FC层
        # model.add(Flatten())
        # model.add(Dense(512,kernel_initializer=TruncatedNormal(mean=0.0, stddev=0.01)))
        # model.add(Activation("relu"))
        # model.add(BatchNormalization())
        # model.add(Dropout(0.6))
        # # softmax 分类
        # model.add(Dense(classes,kernel_initializer=TruncatedNormal(mean=0.0, stddev=0.01)))
        # model.add(Activation("softmax"))
        model.add(Conv2D(64, (3, 3), strides=(1, 1), input_shape=(256, 256, 3), padding='same', activation='relu',
                         kernel_initializer='uniform'))
        model.add(BatchNormalization(axis=chanDim))
        model.add(Conv2D(64, (3, 3), strides=(1, 1), padding='same', activation='relu', kernel_initializer='uniform'))
        model.add(BatchNormalization(axis=chanDim))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        
        model.add(Flatten())
        model.add(Dense(128, activation='relu'))
        model.add(BatchNormalization(axis=chanDim))
        model.add(Dropout(0.5))
        model.add(Dense(128, activation='relu'))
        model.add(BatchNormalization(axis=chanDim))
        model.add(Dropout(0.5))
        model.add(Dense(21, activation='softmax'))
        model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])
        return model

