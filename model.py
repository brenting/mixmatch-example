import tensorflow as tf
from tensorflow.keras.layers import (Activation, BatchNormalization, Conv2D,
                                     Dense, Flatten, MaxPooling2D)


class CNN(tf.keras.Model):
    def __init__(self, input_shape=(28, 28, 1), **kwargs):
        super().__init__(input_shape, **kwargs)
        self.conv_0 = Conv2D(filters=64, kernel_size=(3, 3))
        self.acti_0 = Activation("relu")
        self.conv_1 = Conv2D(filters=64, kernel_size=(3, 3))
        self.acti_1 = Activation("relu")
        self.pool_0 = MaxPooling2D(pool_size=(2, 2))
        self.btch_0 = BatchNormalization()
        self.conv_2 = Conv2D(filters=128, kernel_size=(3, 3))
        self.acti_2 = Activation("relu")
        self.conv_3 = Conv2D(filters=128, kernel_size=(3, 3))
        self.acti_3 = Activation("relu")
        self.pool_0 = MaxPooling2D(pool_size=(2, 2))
        self.btch_1 = BatchNormalization()
        self.conv_4 = Conv2D(filters=256, kernel_size=(3, 3))
        self.acti_4 = Activation("relu")
        self.pool_0 = MaxPooling2D(pool_size=(2, 2))
        self.flat_0 = Flatten()
        self.btch_2 = BatchNormalization()
        self.dens_0 = Dense(512)
        self.acti_5 = Activation("relu")
        self.dens_1 = Dense(10)
        # self.acti_6 = Activation("softmax")

    @tf.function
    def call(self, inputs, training=True):
        x = self.conv_0(inputs)
        x = self.acti_0(x)
        x = self.conv_1(x)
        x = self.acti_1(x)
        x = self.pool_0(x)
        x = self.btch_0(x)
        x = self.conv_2(x)
        x = self.acti_2(x)
        x = self.conv_3(x)
        x = self.acti_3(x)
        x = self.pool_0(x)
        x = self.btch_1(x)
        x = self.conv_4(x)
        x = self.acti_4(x)
        x = self.pool_0(x)
        x = self.flat_0(x)
        x = self.btch_2(x)
        x = self.dens_0(x)
        x = self.acti_5(x)
        x = self.dens_1(x)
        # x = self.acti_6(x)
        return x
