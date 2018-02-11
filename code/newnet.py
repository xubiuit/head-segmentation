import os
import keras as k

from keras import regularizers, initializers
from keras.models import Model, Sequential
from keras.layers import Input, InputLayer, Dense, Dropout, Flatten, Activation
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization, GlobalAveragePooling2D, Conv2DTranspose
from keras.applications.vgg16 import VGG16
from keras.applications.vgg19 import VGG19
from keras.applications.resnet50 import ResNet50
from keras.applications.inception_v3 import InceptionV3
from keras.layers.merge import add
import numpy as np

import densenet121

# Bilinear interpolation (reference: https://github.com/warmspringwinds/tf-image-segmentation/blob/master/tf_image_segmentation/utils/upsampling.py)
def bilinear_upsample_weights(factor, number_of_classes):
    filter_size = factor*2 - factor%2
    factor = (filter_size + 1) // 2
    if filter_size % 2 == 1:
        center = factor - 1
    else:
        center = factor - 0.5
    og = np.ogrid[:filter_size, :filter_size]
    upsample_kernel = (1 - abs(og[0] - center) / factor) * (1 - abs(og[1] - center) / factor)
    weights = np.zeros((filter_size, filter_size, number_of_classes, number_of_classes),
                       dtype=np.float32)
    for i in xrange(number_of_classes):
        weights[:, :, i, i] = upsample_kernel
    return weights

def model3(input_dim, nb_classes=2):
    model = Sequential()
    # preprocess
    model.add(BatchNormalization(input_shape=(input_dim, input_dim, 3)))
    model.add(Conv2D(16, kernel_size=(1, 1), padding='same', kernel_regularizer=regularizers.L1L2(l2=1E-4), bias_regularizer=regularizers.L1L2(l2=1E-4)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Conv2D(16, (1, 1), padding='same', kernel_regularizer=regularizers.L1L2(l2=1E-4), bias_regularizer=regularizers.L1L2(l2=1E-4)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Conv2D(16, (1, 1), padding='same', kernel_regularizer=regularizers.L1L2(l2=1E-4), bias_regularizer=regularizers.L1L2(l2=1E-4)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))

    model.add(BatchNormalization(input_shape=(input_dim, input_dim, 3)))
    model.add(Conv2D(32, kernel_size=(3, 3), padding='same', kernel_regularizer=regularizers.L1L2(l2=1E-4), bias_regularizer=regularizers.L1L2(l2=1E-4)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Conv2D(32, (3, 3), padding='same', kernel_regularizer=regularizers.L1L2(l2=1E-4), bias_regularizer=regularizers.L1L2(l2=1E-4)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Conv2D(32, (3, 3), padding='same', kernel_regularizer=regularizers.L1L2(l2=1E-4), bias_regularizer=regularizers.L1L2(l2=1E-4)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(64, kernel_size=(3, 3), padding='same', kernel_regularizer=regularizers.L1L2(l2=1E-4), bias_regularizer=regularizers.L1L2(l2=1E-4)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Conv2D(64, (3, 3), padding='same', kernel_regularizer=regularizers.L1L2(l2=1E-4), bias_regularizer=regularizers.L1L2(l2=1E-4)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Conv2D(64, (3, 3), padding='same', kernel_regularizer=regularizers.L1L2(l2=1E-4), bias_regularizer=regularizers.L1L2(l2=1E-4)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(128, kernel_size=(3, 3), padding='same', kernel_regularizer=regularizers.L1L2(l2=1E-4), bias_regularizer=regularizers.L1L2(l2=1E-4)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Conv2D(128, (3, 3), padding='same', kernel_regularizer=regularizers.L1L2(l2=1E-4), bias_regularizer=regularizers.L1L2(l2=1E-4)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Conv2D(128, (3, 3), padding='same', kernel_regularizer=regularizers.L1L2(l2=1E-4), bias_regularizer=regularizers.L1L2(l2=1E-4)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(256, kernel_size=(3, 3), padding='same', kernel_regularizer=regularizers.L1L2(l2=1E-4), bias_regularizer=regularizers.L1L2(l2=1E-4)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Conv2D(256, (3, 3), padding='same', kernel_regularizer=regularizers.L1L2(l2=1E-4), bias_regularizer=regularizers.L1L2(l2=1E-4)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Conv2D(256, (3, 3), padding='same', kernel_regularizer=regularizers.L1L2(l2=1E-4), bias_regularizer=regularizers.L1L2(l2=1E-4)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(filters=nb_classes,
               kernel_size=(1, 1), padding='same'))
    model.add(Conv2DTranspose(filters=nb_classes,
                        kernel_size=(64, 64),
                        strides=(32, 32),
                        padding='same',
                        activation='sigmoid',
                        kernel_initializer=initializers.Constant(bilinear_upsample_weights(32, nb_classes))))

    return model

def model4(input_dim, nb_classes=2):
    # preprocess
    x = InputLayer(input_shape = (input_dim,input_dim,3)).input
    y1 = x
    x = Conv2D(16, kernel_size=(1, 1), padding='same', kernel_regularizer=regularizers.L1L2(l2=1E-4), bias_regularizer=regularizers.L1L2(l2=1E-4))(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    # x = Conv2D(16, kernel_size=(7, 7), strides=(2,2), padding='valid', kernel_regularizer=regularizers.L1L2(l2=1E-4), bias_regularizer=regularizers.L1L2(l2=1E-4))(x)


    y2 = x
    x = Conv2D(32, kernel_size=(3, 3), padding='same', kernel_regularizer=regularizers.L1L2(l2=1E-4), bias_regularizer=regularizers.L1L2(l2=1E-4))(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(32, (3, 3), padding='same', kernel_regularizer=regularizers.L1L2(l2=1E-4), bias_regularizer=regularizers.L1L2(l2=1E-4))(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(32, (3, 3), padding='same', kernel_regularizer=regularizers.L1L2(l2=1E-4), bias_regularizer=regularizers.L1L2(l2=1E-4))(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    y2 = Conv2D(32, (1, 1), padding='same', kernel_regularizer=regularizers.L1L2(l2=1E-4), bias_regularizer=regularizers.L1L2(l2=1E-4))(y2)
    x = add([x, y2])
    x = MaxPooling2D(pool_size=(2, 2))(x)

    y3 = x
    x = Conv2D(64, kernel_size=(3, 3), padding='same', kernel_regularizer=regularizers.L1L2(l2=1E-4), bias_regularizer=regularizers.L1L2(l2=1E-4))(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(64, (3, 3), padding='same', kernel_regularizer=regularizers.L1L2(l2=1E-4), bias_regularizer=regularizers.L1L2(l2=1E-4))(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(64, (3, 3), padding='same', kernel_regularizer=regularizers.L1L2(l2=1E-4), bias_regularizer=regularizers.L1L2(l2=1E-4))(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    y3 = Conv2D(64, (1, 1), padding='same', kernel_regularizer=regularizers.L1L2(l2=1E-4), bias_regularizer=regularizers.L1L2(l2=1E-4))(y3)
    x = add([x, y3])
    x = MaxPooling2D(pool_size=(2, 2))(x)

    y4 = x
    x = Conv2D(128, kernel_size=(3, 3), padding='same', kernel_regularizer=regularizers.L1L2(l2=1E-4), bias_regularizer=regularizers.L1L2(l2=1E-4))(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(128, (3, 3), padding='same', kernel_regularizer=regularizers.L1L2(l2=1E-4), bias_regularizer=regularizers.L1L2(l2=1E-4))(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(128, (3, 3), padding='same', kernel_regularizer=regularizers.L1L2(l2=1E-4), bias_regularizer=regularizers.L1L2(l2=1E-4))(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    y4 = Conv2D(128, (1, 1), padding='same', kernel_regularizer=regularizers.L1L2(l2=1E-4), bias_regularizer=regularizers.L1L2(l2=1E-4))(y4)
    x = add([x, y4])
    x = MaxPooling2D(pool_size=(2, 2))(x)

    y5 = x
    x = Conv2D(256, kernel_size=(3, 3), padding='same', kernel_regularizer=regularizers.L1L2(l2=1E-4), bias_regularizer=regularizers.L1L2(l2=1E-4))(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(256, (3, 3), padding='same', kernel_regularizer=regularizers.L1L2(l2=1E-4), bias_regularizer=regularizers.L1L2(l2=1E-4))(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(256, (3, 3), padding='same', kernel_regularizer=regularizers.L1L2(l2=1E-4), bias_regularizer=regularizers.L1L2(l2=1E-4))(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    y5 = Conv2D(256, (1, 1), padding='same', kernel_regularizer=regularizers.L1L2(l2=1E-4), bias_regularizer=regularizers.L1L2(l2=1E-4))(y5)
    x = add([x, y5])
    x = MaxPooling2D(pool_size=(2, 2))(x)

    x = Conv2D(16, (1, 1), padding='same', kernel_regularizer=regularizers.L1L2(l2=1E-4), bias_regularizer=regularizers.L1L2(l2=1E-4))(x)

    x = Conv2D(filters=nb_classes,
               kernel_size=(1, 1))(x)
    x = Conv2DTranspose(filters=nb_classes,
                        kernel_size=(64, 64),
                        strides=(32, 32),
                        padding='same',
                        activation='sigmoid',
                        kernel_initializer=initializers.Constant(bilinear_upsample_weights(32, nb_classes)))(x)

    return Model(input=y1, output=x)


def fcn_32s(input_dim, nb_classes=2):
    inputs = Input(shape=(input_dim,input_dim,3))
    vgg16 = VGG16(weights=None, include_top=False, input_tensor=inputs)
    pretrain_model_path = "../weights/vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5"
    if not os.path.exists(pretrain_model_path):
        raise RuntimeError("No pretrained model loaded.")
    vgg16.load_weights(pretrain_model_path)
    x = Conv2D(filters=nb_classes,
               kernel_size=(1, 1))(vgg16.output)
    x = Conv2DTranspose(filters=nb_classes,
                        kernel_size=(64, 64),
                        strides=(32, 32),
                        padding='same',
                        activation='sigmoid',
                        kernel_initializer=initializers.Constant(bilinear_upsample_weights(32, nb_classes)))(x)
    model = Model(inputs=inputs, outputs=x)
    for layer in model.layers[:15]:
        layer.trainable = False
    return model

def vgg16():
    base_model = VGG16(weights=None, include_top=False, input_shape = (224,224,3))
    # Classification block
    x = Flatten(name='flatten', input_shape=base_model.output_shape[1:])(base_model.output)
    x = Dense(512, activation='relu', name='fc1')(x)
    x = Dense(512, activation='relu', name='fc2')(x)
    x = Dense(17, activation='softmax', name='predictions')(x)

    model = Model(inputs=base_model.input, outputs=x)
    return model

def vgg19():
    base_model = VGG19(weights=None, include_top=False, input_shape = (224,224,3))
    x = Flatten(name='flatten')(base_model.output)
    x = Dense(512, activation='relu', name='fc1')(x)
    x = Dense(512, activation='relu', name='fc2')(x)
    x = Dense(17, activation='softmax', name='predictions')(x)

    model = Model(inputs=base_model.input, outputs=x)
    return model


def resNet50(input_dim):
    base_model = ResNet50(weights=None, include_top=False, pooling='avg', input_shape = (input_dim,input_dim,3))
    base_model.load_weights("../weights/resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5")
    # x = Flatten()(base_model.output)
    x = base_model.output
    x = Dropout(0.25)(x)
    x = Dense(1024, activation='relu', name='fc', kernel_initializer='he_uniform')(x)
    x = Dropout(0.5)(x)
    x = Dense(17, activation='softmax', name='predictions', kernel_initializer='he_uniform')(x)
    model = Model(inputs=base_model.input, outputs=x)
    # for layer in model.layers[:25]:
    #     layer.trainable = False
    return model

def inceptionV3(input_dim):
    base_model = InceptionV3(weights=None, include_top=False, input_shape = (input_dim,input_dim,3))
    # Classification block
    x = GlobalAveragePooling2D(name='avg_pool')(base_model.output)
    x = Dense(17, activation='softmax', name='predictions')(x)
    model = Model(inputs=base_model.input, outputs=x)
    return model

def denseNet(input_dim):
    base_model = densenet.DenseNet(input_shape=(input_dim, input_dim, 3), classes=17, dropout_rate=0.2, weights=None, include_top=False)
    x = Dense(17, activation='softmax', kernel_regularizer=regularizers.L1L2(l2=1E-4),
              bias_regularizer=regularizers.L1L2(l2=1E-4))(base_model.output)
    model = Model(inputs=base_model.input, outputs=x)

    # Load model
    weights_file = "../weights/DenseNet-40-12CIFAR10-tf.h5"
    if os.path.exists(weights_file):
        model.load_weights(weights_file)
        print("Model loaded.")
    return model


def denseNet121(input_dim):
    weights_path = '../weights/densenet121_weights_tf.h5'

    # Test pretrained model
    # base_model = densenet121.DenseNet(reduction=0.5, classes=1000, weights_path=weights_path)
    model = densenet121.densenet121_model(img_rows=input_dim, img_cols=input_dim, color_type=3, num_classes=17, weights_path = weights_path)
    return model