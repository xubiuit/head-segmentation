## https://github.com/junjungoal/Tiramisu-keras/edit/master/Tiramisu.py
from keras.models import Model
from keras.layers import Input, MaxPooling2D
from keras.layers.convolutional import Conv2D, Conv2DTranspose
from keras.layers.normalization import BatchNormalization
from keras.layers.core import Activation, Dropout
from keras.layers.merge import concatenate
from keras import backend as K
from keras.regularizers import l2
from keras.utils import plot_model


def denseBlock(t, nb_layers):
    for _ in range(nb_layers):
        tmp = t
        t = BatchNormalization()(t)

        t = Activation('relu')(t)
        t = Conv2D(16, kernel_size=(3, 3), padding='same', kernel_initializer='he_uniform')(t)
        # t = Dropout(0.2)(t)
        t = concatenate([t, tmp])
    return t

def transitionDown(t, nb_features):
    t = BatchNormalization()(t)
    t = Activation('relu')(t)
    t = Conv2D(nb_features, kernel_size=(1, 1), padding='same', kernel_initializer='he_uniform')(t)
    # t = Dropout(0.2)(t)
    t = MaxPooling2D(pool_size=(2, 2), strides=2, padding='same')(t)
    return t

def Tiramisu(layer_per_block, input_shape, num_classes, n_pool=6, growth_rate=16):
    input_layer = Input(shape=input_shape)
    t = Conv2D(48, kernel_size=(3, 3), strides=(1, 1), padding='same')(input_layer)

    #dense block
    nb_features = 48
    skip_connections = []
    for i in range(n_pool):
        t = denseBlock(t, layer_per_block[i])
        skip_connections.append(t)
        nb_features += growth_rate * layer_per_block[i]
        t = transitionDown(t, nb_features)

    t = denseBlock(t, layer_per_block[n_pool]) # bottle neck

    skip_connections = skip_connections[::-1] #subvert the array

    for i in range(n_pool):
        keep_nb_features = growth_rate * layer_per_block[n_pool + i]
        t = Conv2DTranspose(keep_nb_features, strides=2, kernel_size=(3, 3), padding='same')(t) # transition Up
        t = concatenate([t, skip_connections[i]])

        t = denseBlock(t, layer_per_block[n_pool+i+1])

    # t = Conv2D(12, kernel_size=(1, 1), padding='same', kernel_initializer='he_uniform', data_format='channels_last')(t)
    # output_layer = Activation('softmax')(t)

    output_layer = Conv2D(num_classes, (1, 1), padding='same', activation='sigmoid')(t)

    return Model(inputs=input_layer, outputs=output_layer)

def get_tiramisunet(input_shape=(224, 224, 3), num_classes=1):
    # layer_per_block = [4, 5, 7, 10, 12, 15, 12, 10, 7, 5, 4]
    layer_per_block = [4, 4, 4, 4, 4, 4, 8, 4, 4, 4, 4, 4, 4]

    tiramisu = Tiramisu(layer_per_block, input_shape=input_shape, num_classes=num_classes)
    #plot_model(tiramisu, to_file='model.pdf')
    #tiramisu.summary()
    return tiramisu