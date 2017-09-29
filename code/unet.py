from keras.models import Model
from keras.layers import Input, concatenate, Conv2D, MaxPooling2D, BatchNormalization, Activation, UpSampling2D
from keras.optimizers import SGD
import keras.backend as K
from keras.layers.advanced_activations import LeakyReLU

def dice_loss(y_true, y_pred):
    smooth = 1.
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)


def get_unet_128(input_shape=(128, 128, 3),
                 num_classes=1):
    inputs = Input(shape=input_shape)
    # 128

    down1 = Conv2D(64, (3, 3), padding='same')(inputs)
    down1 = BatchNormalization()(down1)
    down1 = Activation('relu')(down1)
    down1 = Conv2D(64, (3, 3), padding='same')(down1)
    down1 = BatchNormalization()(down1)
    down1 = Activation('relu')(down1)
    down1_pool = MaxPooling2D((2, 2), strides=(2, 2))(down1)
    # 64

    down2 = Conv2D(128, (3, 3), padding='same')(down1_pool)
    down2 = BatchNormalization()(down2)
    down2 = Activation('relu')(down2)
    down2 = Conv2D(128, (3, 3), padding='same')(down2)
    down2 = BatchNormalization()(down2)
    down2 = Activation('relu')(down2)
    down2_pool = MaxPooling2D((2, 2), strides=(2, 2))(down2)
    # 32

    down3 = Conv2D(256, (3, 3), padding='same')(down2_pool)
    down3 = BatchNormalization()(down3)
    down3 = Activation('relu')(down3)
    down3 = Conv2D(256, (3, 3), padding='same')(down3)
    down3 = BatchNormalization()(down3)
    down3 = Activation('relu')(down3)
    down3_pool = MaxPooling2D((2, 2), strides=(2, 2))(down3)
    # 16

    down4 = Conv2D(512, (3, 3), padding='same')(down3_pool)
    down4 = BatchNormalization()(down4)
    down4 = Activation('relu')(down4)
    down4 = Conv2D(512, (3, 3), padding='same')(down4)
    down4 = BatchNormalization()(down4)
    down4 = Activation('relu')(down4)
    down4_pool = MaxPooling2D((2, 2), strides=(2, 2))(down4)
    # 8

    center = Conv2D(1024, (3, 3), padding='same')(down4_pool)
    center = BatchNormalization()(center)
    center = Activation('relu')(center)
    center = Conv2D(1024, (3, 3), padding='same')(center)
    center = BatchNormalization()(center)
    center = Activation('relu')(center)
    # center

    up4 = UpSampling2D((2, 2))(center)
    up4 = concatenate([down4, up4], axis=3)
    up4 = Conv2D(512, (3, 3), padding='same')(up4)
    up4 = BatchNormalization()(up4)
    up4 = Activation('relu')(up4)
    up4 = Conv2D(512, (3, 3), padding='same')(up4)
    up4 = BatchNormalization()(up4)
    up4 = Activation('relu')(up4)
    up4 = Conv2D(512, (3, 3), padding='same')(up4)
    up4 = BatchNormalization()(up4)
    up4 = Activation('relu')(up4)
    # 16

    up3 = UpSampling2D((2, 2))(up4)
    up3 = concatenate([down3, up3], axis=3)
    up3 = Conv2D(256, (3, 3), padding='same')(up3)
    up3 = BatchNormalization()(up3)
    up3 = Activation('relu')(up3)
    up3 = Conv2D(256, (3, 3), padding='same')(up3)
    up3 = BatchNormalization()(up3)
    up3 = Activation('relu')(up3)
    up3 = Conv2D(256, (3, 3), padding='same')(up3)
    up3 = BatchNormalization()(up3)
    up3 = Activation('relu')(up3)
    # 32

    up2 = UpSampling2D((2, 2))(up3)
    up2 = concatenate([down2, up2], axis=3)
    up2 = Conv2D(128, (3, 3), padding='same')(up2)
    up2 = BatchNormalization()(up2)
    up2 = Activation('relu')(up2)
    up2 = Conv2D(128, (3, 3), padding='same')(up2)
    up2 = BatchNormalization()(up2)
    up2 = Activation('relu')(up2)
    up2 = Conv2D(128, (3, 3), padding='same')(up2)
    up2 = BatchNormalization()(up2)
    up2 = Activation('relu')(up2)
    # 64

    up1 = UpSampling2D((2, 2))(up2)
    up1 = concatenate([down1, up1], axis=3)
    up1 = Conv2D(64, (3, 3), padding='same')(up1)
    up1 = BatchNormalization()(up1)
    up1 = Activation('relu')(up1)
    up1 = Conv2D(64, (3, 3), padding='same')(up1)
    up1 = BatchNormalization()(up1)
    up1 = Activation('relu')(up1)
    up1 = Conv2D(64, (3, 3), padding='same')(up1)
    up1 = BatchNormalization()(up1)
    up1 = Activation('relu')(up1)
    # 128

    classify = Conv2D(num_classes, (1, 1), activation='sigmoid')(up1)

    model = Model(inputs=inputs, outputs=classify)

    model.compile(optimizer=SGD(lr=0.01, momentum=0.9), loss='binary_crossentropy', metrics=[dice_loss])

    return model


def get_unet_256(input_shape=(256, 256, 3),
                 num_classes=1):
    inputs = Input(shape=input_shape)
    # 256

    down0 = Conv2D(32, (3, 3), padding='same')(inputs)
    down0 = BatchNormalization()(down0)
    down0 = Activation('relu')(down0)
    down0 = Conv2D(32, (3, 3), padding='same')(down0)
    down0 = BatchNormalization()(down0)
    down0 = Activation('relu')(down0)
    down0_pool = MaxPooling2D((2, 2), strides=(2, 2))(down0)
    # 128

    down1 = Conv2D(64, (3, 3), padding='same')(down0_pool)
    down1 = BatchNormalization()(down1)
    down1 = Activation('relu')(down1)
    down1 = Conv2D(64, (3, 3), padding='same')(down1)
    down1 = BatchNormalization()(down1)
    down1 = Activation('relu')(down1)
    down1_pool = MaxPooling2D((2, 2), strides=(2, 2))(down1)
    # 64

    down2 = Conv2D(128, (3, 3), padding='same')(down1_pool)
    down2 = BatchNormalization()(down2)
    down2 = Activation('relu')(down2)
    down2 = Conv2D(128, (3, 3), padding='same')(down2)
    down2 = BatchNormalization()(down2)
    down2 = Activation('relu')(down2)
    down2_pool = MaxPooling2D((2, 2), strides=(2, 2))(down2)
    # 32

    down3 = Conv2D(256, (3, 3), padding='same')(down2_pool)
    down3 = BatchNormalization()(down3)
    down3 = Activation('relu')(down3)
    down3 = Conv2D(256, (3, 3), padding='same')(down3)
    down3 = BatchNormalization()(down3)
    down3 = Activation('relu')(down3)
    down3_pool = MaxPooling2D((2, 2), strides=(2, 2))(down3)
    # 16

    down4 = Conv2D(512, (3, 3), padding='same')(down3_pool)
    down4 = BatchNormalization()(down4)
    down4 = Activation('relu')(down4)
    down4 = Conv2D(512, (3, 3), padding='same')(down4)
    down4 = BatchNormalization()(down4)
    down4 = Activation('relu')(down4)
    down4_pool = MaxPooling2D((2, 2), strides=(2, 2))(down4)
    # 8

    center = Conv2D(1024, (3, 3), padding='same')(down4_pool)
    center = BatchNormalization()(center)
    center = Activation('relu')(center)
    center = Conv2D(1024, (3, 3), padding='same')(center)
    center = BatchNormalization()(center)
    center = Activation('relu')(center)
    # center

    up4 = UpSampling2D((2, 2))(center)
    up4 = concatenate([down4, up4], axis=3)
    up4 = Conv2D(512, (3, 3), padding='same')(up4)
    up4 = BatchNormalization()(up4)
    up4 = Activation('relu')(up4)
    up4 = Conv2D(512, (3, 3), padding='same')(up4)
    up4 = BatchNormalization()(up4)
    up4 = Activation('relu')(up4)
    up4 = Conv2D(512, (3, 3), padding='same')(up4)
    up4 = BatchNormalization()(up4)
    up4 = Activation('relu')(up4)
    # 16

    up3 = UpSampling2D((2, 2))(up4)
    up3 = concatenate([down3, up3], axis=3)
    up3 = Conv2D(256, (3, 3), padding='same')(up3)
    up3 = BatchNormalization()(up3)
    up3 = Activation('relu')(up3)
    up3 = Conv2D(256, (3, 3), padding='same')(up3)
    up3 = BatchNormalization()(up3)
    up3 = Activation('relu')(up3)
    up3 = Conv2D(256, (3, 3), padding='same')(up3)
    up3 = BatchNormalization()(up3)
    up3 = Activation('relu')(up3)
    # 32

    up2 = UpSampling2D((2, 2))(up3)
    up2 = concatenate([down2, up2], axis=3)
    up2 = Conv2D(128, (3, 3), padding='same')(up2)
    up2 = BatchNormalization()(up2)
    up2 = Activation('relu')(up2)
    up2 = Conv2D(128, (3, 3), padding='same')(up2)
    up2 = BatchNormalization()(up2)
    up2 = Activation('relu')(up2)
    up2 = Conv2D(128, (3, 3), padding='same')(up2)
    up2 = BatchNormalization()(up2)
    up2 = Activation('relu')(up2)
    # 64

    up1 = UpSampling2D((2, 2))(up2)
    up1 = concatenate([down1, up1], axis=3)
    up1 = Conv2D(64, (3, 3), padding='same')(up1)
    up1 = BatchNormalization()(up1)
    up1 = Activation('relu')(up1)
    up1 = Conv2D(64, (3, 3), padding='same')(up1)
    up1 = BatchNormalization()(up1)
    up1 = Activation('relu')(up1)
    up1 = Conv2D(64, (3, 3), padding='same')(up1)
    up1 = BatchNormalization()(up1)
    up1 = Activation('relu')(up1)
    # 128

    up0 = UpSampling2D((2, 2))(up1)
    up0 = concatenate([down0, up0], axis=3)
    up0 = Conv2D(32, (3, 3), padding='same')(up0)
    up0 = BatchNormalization()(up0)
    up0 = Activation('relu')(up0)
    up0 = Conv2D(32, (3, 3), padding='same')(up0)
    up0 = BatchNormalization()(up0)
    up0 = Activation('relu')(up0)
    up0 = Conv2D(32, (3, 3), padding='same')(up0)
    up0 = BatchNormalization()(up0)
    up0 = Activation('relu')(up0)
    # 256

    classify = Conv2D(num_classes, (1, 1), activation='sigmoid')(up0)

    model = Model(inputs=inputs, outputs=classify)

    # model.compile(optimizer=SGD(lr=0.01, momentum=0.9), loss='binary_crossentropy', metrics=[dice_loss])

    return model




def get_unet_512(input_shape=(512, 512, 3),
                 num_classes=1):
    inputs = Input(shape=input_shape)
    # 512
    down0 = block(inputs, 16)
    down0_pool = MaxPooling2D((2, 2), strides=(2, 2))(down0)

    # 256
    down1 = block(down0_pool, 32)
    down1_pool = MaxPooling2D((2, 2), strides=(2, 2))(down1)

    # 128
    down2 = block(down1_pool, 64)
    down2_pool = MaxPooling2D((2, 2), strides=(2, 2))(down2)

    # 64
    down3 = block(down2_pool, 128)
    down3_pool = MaxPooling2D((2, 2), strides=(2, 2))(down3)

    # 32
    down4 = block(down3_pool, 256)
    down4_pool = MaxPooling2D((2, 2), strides=(2, 2))(down4)

    # 16
    down5 = block(down4_pool, 512)
    down5_pool = MaxPooling2D((2, 2), strides=(2, 2))(down5)

    # 8
    center = block(down5_pool, 1024)

    # center
    up5 = UpSampling2D((2, 2))(center)
    up5 = concatenate([down5, up5], axis=3)
    up5 = block(up5, 512)

    # 16
    up4 = UpSampling2D((2, 2))(up5)
    up4 = concatenate([down4, up4], axis=3)
    up4 = block(up4, 256)

    # 32
    up3 = UpSampling2D((2, 2))(up4)
    up3 = concatenate([down3, up3], axis=3)
    up3 = block(up3, 128)

    # 64
    up2 = UpSampling2D((2, 2))(up3)
    up2 = concatenate([down2, up2], axis=3)
    up2 = block(up2, 64)

    # 128
    up1 = UpSampling2D((2, 2))(up2)
    up1 = concatenate([down1, up1], axis=3)
    up1 = block(up1, 32)

    # 256
    up0 = UpSampling2D((2, 2))(up1)
    up0 = concatenate([down0, up0], axis=3)
    up0 = block(up0, 16)

    # 512

    classify = Conv2D(num_classes, (1, 1), activation='sigmoid')(up0)
    mask = classify
    model = Model(inputs=inputs, outputs=[classify])
    # model = Model(inputs=inputs, outputs=[classify])

    # model.compile(optimizer=SGD(lr=0.01, momentum=0.9), loss='binary_crossentropy', metrics=[dice_loss])

    return model


def block(in_layer, nchan, relu=False):
    b1 = Conv2D(nchan, (3, 3), padding='same', kernel_initializer='he_uniform')(in_layer)
    # b1 = BatchNormalization()(b1)
    if relu:
        b1 = Activation('relu')(b1)
    else:
        b1 = LeakyReLU(0.0001)(b1)

    b2 = Conv2D(nchan, (3, 3), padding='same')(b1)
    # b2 = BatchNormalization()(b2)
    if relu:
        b2 = Activation('relu')(b2)
    else:
        b2 = LeakyReLU(0.0001)(b2)

    b3 = Conv2D(nchan, (3, 3), padding='same')(b2)
    # b3 = BatchNormalization()(b3)
    if relu:
        b3 = Activation('relu')(b3)
    else:
        b3 = LeakyReLU(0.0001)(b3)

    b4 = Conv2D(nchan, (3, 3), padding='same')(b3)
    # b4 = BatchNormalization()(b4)
    if relu:
        b4 = Activation('relu')(b4)
    else:
        b4 = LeakyReLU(0.0001)(b4)

    out_layer = concatenate([b1, b4], axis=3)
    out_layer = Conv2D(nchan, (1, 1), padding='same')(out_layer)
    return out_layer

# def get_unet_512(input_shape=(512, 512, 3),
#                  num_classes=1):
#     inputs = Input(shape=input_shape)
#     # 512
#
#     down0a = Conv2D(16, (3, 3), padding='same')(inputs)
#     down0a = BatchNormalization()(down0a)
#     down0a = Activation('relu')(down0a)
#     down0a = Conv2D(16, (3, 3), padding='same')(down0a)
#     down0a = BatchNormalization()(down0a)
#     down0a = Activation('relu')(down0a)
#     down0a_pool = MaxPooling2D((2, 2), strides=(2, 2))(down0a)
#     # 256
#
#     down0 = Conv2D(32, (3, 3), padding='same')(down0a_pool)
#     down0 = BatchNormalization()(down0)
#     down0 = Activation('relu')(down0)
#     down0 = Conv2D(32, (3, 3), padding='same')(down0)
#     down0 = BatchNormalization()(down0)
#     down0 = Activation('relu')(down0)
#     down0_pool = MaxPooling2D((2, 2), strides=(2, 2))(down0)
#     # 128
#
#     down1 = Conv2D(64, (3, 3), padding='same')(down0_pool)
#     down1 = BatchNormalization()(down1)
#     down1 = Activation('relu')(down1)
#     down1 = Conv2D(64, (3, 3), padding='same')(down1)
#     down1 = BatchNormalization()(down1)
#     down1 = Activation('relu')(down1)
#     down1_pool = MaxPooling2D((2, 2), strides=(2, 2))(down1)
#     # 64
#
#     down2 = Conv2D(128, (3, 3), padding='same')(down1_pool)
#     down2 = BatchNormalization()(down2)
#     down2 = Activation('relu')(down2)
#     down2 = Conv2D(128, (3, 3), padding='same')(down2)
#     down2 = BatchNormalization()(down2)
#     down2 = Activation('relu')(down2)
#     down2_pool = MaxPooling2D((2, 2), strides=(2, 2))(down2)
#     # 32
#
#     down3 = Conv2D(256, (3, 3), padding='same')(down2_pool)
#     down3 = BatchNormalization()(down3)
#     down3 = Activation('relu')(down3)
#     down3 = Conv2D(256, (3, 3), padding='same')(down3)
#     down3 = BatchNormalization()(down3)
#     down3 = Activation('relu')(down3)
#     down3_pool = MaxPooling2D((2, 2), strides=(2, 2))(down3)
#     # 16
#
#     down4 = Conv2D(512, (3, 3), padding='same')(down3_pool)
#     down4 = BatchNormalization()(down4)
#     down4 = Activation('relu')(down4)
#     down4 = Conv2D(512, (3, 3), padding='same')(down4)
#     down4 = BatchNormalization()(down4)
#     down4 = Activation('relu')(down4)
#     down4_pool = MaxPooling2D((2, 2), strides=(2, 2))(down4)
#     # 8
#
#     center = Conv2D(1024, (3, 3), padding='same')(down4_pool)
#     center = BatchNormalization()(center)
#     center = Activation('relu')(center)
#     center = Conv2D(1024, (3, 3), padding='same')(center)
#     center = BatchNormalization()(center)
#     center = Activation('relu')(center)
#     # center
#
#     up4 = UpSampling2D((2, 2))(center)
#     up4 = concatenate([down4, up4], axis=3)
#     up4 = Conv2D(512, (3, 3), padding='same')(up4)
#     up4 = BatchNormalization()(up4)
#     up4 = Activation('relu')(up4)
#     up4 = Conv2D(512, (3, 3), padding='same')(up4)
#     up4 = BatchNormalization()(up4)
#     up4 = Activation('relu')(up4)
#     up4 = Conv2D(512, (3, 3), padding='same')(up4)
#     up4 = BatchNormalization()(up4)
#     up4 = Activation('relu')(up4)
#     # 16
#
#     up3 = UpSampling2D((2, 2))(up4)
#     up3 = concatenate([down3, up3], axis=3)
#     up3 = Conv2D(256, (3, 3), padding='same')(up3)
#     up3 = BatchNormalization()(up3)
#     up3 = Activation('relu')(up3)
#     up3 = Conv2D(256, (3, 3), padding='same')(up3)
#     up3 = BatchNormalization()(up3)
#     up3 = Activation('relu')(up3)
#     up3 = Conv2D(256, (3, 3), padding='same')(up3)
#     up3 = BatchNormalization()(up3)
#     up3 = Activation('relu')(up3)
#     # 32
#
#     up2 = UpSampling2D((2, 2))(up3)
#     up2 = concatenate([down2, up2], axis=3)
#     up2 = Conv2D(128, (3, 3), padding='same')(up2)
#     up2 = BatchNormalization()(up2)
#     up2 = Activation('relu')(up2)
#     up2 = Conv2D(128, (3, 3), padding='same')(up2)
#     up2 = BatchNormalization()(up2)
#     up2 = Activation('relu')(up2)
#     up2 = Conv2D(128, (3, 3), padding='same')(up2)
#     up2 = BatchNormalization()(up2)
#     up2 = Activation('relu')(up2)
#     # 64
#
#     up1 = UpSampling2D((2, 2))(up2)
#     up1 = concatenate([down1, up1], axis=3)
#     up1 = Conv2D(64, (3, 3), padding='same')(up1)
#     up1 = BatchNormalization()(up1)
#     up1 = Activation('relu')(up1)
#     up1 = Conv2D(64, (3, 3), padding='same')(up1)
#     up1 = BatchNormalization()(up1)
#     up1 = Activation('relu')(up1)
#     up1 = Conv2D(64, (3, 3), padding='same')(up1)
#     up1 = BatchNormalization()(up1)
#     up1 = Activation('relu')(up1)
#     # 128
#
#     up0 = UpSampling2D((2, 2))(up1)
#     up0 = concatenate([down0, up0], axis=3)
#     up0 = Conv2D(32, (3, 3), padding='same')(up0)
#     up0 = BatchNormalization()(up0)
#     up0 = Activation('relu')(up0)
#     up0 = Conv2D(32, (3, 3), padding='same')(up0)
#     up0 = BatchNormalization()(up0)
#     up0 = Activation('relu')(up0)
#     up0 = Conv2D(32, (3, 3), padding='same')(up0)
#     up0 = BatchNormalization()(up0)
#     up0 = Activation('relu')(up0)
#     # 256
#
#     up0a = UpSampling2D((2, 2))(up0)
#     up0a = concatenate([down0a, up0a], axis=3)
#     up0a = Conv2D(16, (3, 3), padding='same')(up0a)
#     up0a = BatchNormalization()(up0a)
#     up0a = Activation('relu')(up0a)
#     up0a = Conv2D(16, (3, 3), padding='same')(up0a)
#     up0a = BatchNormalization()(up0a)
#     up0a = Activation('relu')(up0a)
#     up0a = Conv2D(16, (3, 3), padding='same')(up0a)
#     up0a = BatchNormalization()(up0a)
#     up0a = Activation('relu')(up0a)
#     # 512
#
#     classify = Conv2D(num_classes, (1, 1), activation='sigmoid')(up0a)
#     mask = classify
#     model = Model(inputs=inputs, outputs=[classify])
#     # model = Model(inputs=inputs, outputs=[classify])
#
#     # model.compile(optimizer=SGD(lr=0.01, momentum=0.9), loss='binary_crossentropy', metrics=[dice_loss])
#
#     return model



def get_unet_1024(input_shape=(512, 512, 3),
                 num_classes=1):
    inputs = Input(shape=input_shape)
    # 512

    down0a = Conv2D(16, (3, 3), padding='same')(inputs)
    down0a = BatchNormalization()(down0a)
    down0a = Activation('relu')(down0a)
    down0a = Conv2D(16, (3, 3), padding='same')(down0a)
    down0a = BatchNormalization()(down0a)
    down0a = Activation('relu')(down0a)
    down0a_pool = MaxPooling2D((2, 2), strides=(2, 2))(down0a)
    # 256

    down0 = Conv2D(32, (3, 3), padding='same')(down0a_pool)
    down0 = BatchNormalization()(down0)
    down0 = Activation('relu')(down0)
    down0 = Conv2D(32, (3, 3), padding='same')(down0)
    down0 = BatchNormalization()(down0)
    down0 = Activation('relu')(down0)
    down0_pool = MaxPooling2D((2, 2), strides=(2, 2))(down0)
    # 128

    down1 = Conv2D(64, (3, 3), padding='same')(down0_pool)
    down1 = BatchNormalization()(down1)
    down1 = Activation('relu')(down1)
    down1 = Conv2D(64, (3, 3), padding='same')(down1)
    down1 = BatchNormalization()(down1)
    down1 = Activation('relu')(down1)
    down1_pool = MaxPooling2D((2, 2), strides=(2, 2))(down1)
    # 64

    down2 = Conv2D(128, (3, 3), padding='same')(down1_pool)
    down2 = BatchNormalization()(down2)
    down2 = Activation('relu')(down2)
    down2 = Conv2D(128, (3, 3), padding='same')(down2)
    down2 = BatchNormalization()(down2)
    down2 = Activation('relu')(down2)
    down2_pool = MaxPooling2D((2, 2), strides=(2, 2))(down2)
    # 32

    down3 = Conv2D(256, (3, 3), padding='same')(down2_pool)
    down3 = BatchNormalization()(down3)
    down3 = Activation('relu')(down3)
    down3 = Conv2D(256, (3, 3), padding='same')(down3)
    down3 = BatchNormalization()(down3)
    down3 = Activation('relu')(down3)
    down3_pool = MaxPooling2D((2, 2), strides=(2, 2))(down3)
    # 16

    down4 = Conv2D(512, (3, 3), padding='same')(down3_pool)
    down4 = BatchNormalization()(down4)
    down4 = Activation('relu')(down4)
    down4 = Conv2D(512, (3, 3), padding='same')(down4)
    down4 = BatchNormalization()(down4)
    down4 = Activation('relu')(down4)
    down4_pool = MaxPooling2D((2, 2), strides=(2, 2))(down4)
    # 8

    center = Conv2D(1024, (3, 3), padding='same')(down4_pool)
    center = BatchNormalization()(center)
    center = Activation('relu')(center)
    center = Conv2D(1024, (3, 3), padding='same')(center)
    center = BatchNormalization()(center)
    center = Activation('relu')(center)
    # center

    up4 = UpSampling2D((2, 2))(center)
    up4 = concatenate([down4, up4], axis=3)
    up4 = Conv2D(512, (3, 3), padding='same')(up4)
    up4 = BatchNormalization()(up4)
    up4 = Activation('relu')(up4)
    up4 = Conv2D(512, (3, 3), padding='same')(up4)
    up4 = BatchNormalization()(up4)
    up4 = Activation('relu')(up4)
    up4 = Conv2D(512, (3, 3), padding='same')(up4)
    up4 = BatchNormalization()(up4)
    up4 = Activation('relu')(up4)
    # 16

    up3 = UpSampling2D((2, 2))(up4)
    up3 = concatenate([down3, up3], axis=3)
    up3 = Conv2D(256, (3, 3), padding='same')(up3)
    up3 = BatchNormalization()(up3)
    up3 = Activation('relu')(up3)
    up3 = Conv2D(256, (3, 3), padding='same')(up3)
    up3 = BatchNormalization()(up3)
    up3 = Activation('relu')(up3)
    up3 = Conv2D(256, (3, 3), padding='same')(up3)
    up3 = BatchNormalization()(up3)
    up3 = Activation('relu')(up3)
    # 32

    up2 = UpSampling2D((2, 2))(up3)
    up2 = concatenate([down2, up2], axis=3)
    up2 = Conv2D(128, (3, 3), padding='same')(up2)
    up2 = BatchNormalization()(up2)
    up2 = Activation('relu')(up2)
    up2 = Conv2D(128, (3, 3), padding='same')(up2)
    up2 = BatchNormalization()(up2)
    up2 = Activation('relu')(up2)
    up2 = Conv2D(128, (3, 3), padding='same')(up2)
    up2 = BatchNormalization()(up2)
    up2 = Activation('relu')(up2)
    # 64

    up1 = UpSampling2D((2, 2))(up2)
    up1 = concatenate([down1, up1], axis=3)
    up1 = Conv2D(64, (3, 3), padding='same')(up1)
    up1 = BatchNormalization()(up1)
    up1 = Activation('relu')(up1)
    up1 = Conv2D(64, (3, 3), padding='same')(up1)
    up1 = BatchNormalization()(up1)
    up1 = Activation('relu')(up1)
    up1 = Conv2D(64, (3, 3), padding='same')(up1)
    up1 = BatchNormalization()(up1)
    up1 = Activation('relu')(up1)
    # 128

    up0 = UpSampling2D((2, 2))(up1)
    up0 = concatenate([down0, up0], axis=3)
    up0 = Conv2D(32, (3, 3), padding='same')(up0)
    up0 = BatchNormalization()(up0)
    up0 = Activation('relu')(up0)
    up0 = Conv2D(32, (3, 3), padding='same')(up0)
    up0 = BatchNormalization()(up0)
    up0 = Activation('relu')(up0)
    up0 = Conv2D(32, (3, 3), padding='same')(up0)
    up0 = BatchNormalization()(up0)
    up0 = Activation('relu')(up0)
    # 256

    up0a = UpSampling2D((2, 2))(up0)
    up0a = concatenate([down0a, up0a], axis=3)
    up0a = Conv2D(16, (3, 3), padding='same')(up0a)
    up0a = BatchNormalization()(up0a)
    up0a = Activation('relu')(up0a)
    up0a = Conv2D(16, (3, 3), padding='same')(up0a)
    up0a = BatchNormalization()(up0a)
    up0a = Activation('relu')(up0a)
    up0a = Conv2D(16, (3, 3), padding='same')(up0a)
    up0a = BatchNormalization()(up0a)
    up0a = Activation('relu')(up0a)
    # 512

    up0b = concatenate([inputs, up0a], axis=3)
    up0b = UpSampling2D((2, 2))(up0b)
    up0b = Conv2D(16, (3, 3), padding='same')(up0b)
    up0b = BatchNormalization()(up0b)
    up0b = Activation('relu')(up0b)
    up0b = Conv2D(16, (3, 3), padding='same')(up0b)
    up0b = BatchNormalization()(up0b)
    up0b = Activation('relu')(up0b)
    up0b = Conv2D(16, (3, 3), padding='same')(up0b)
    up0b = BatchNormalization()(up0b)
    up0b = Activation('relu')(up0b)
    # 1024

    classify = Conv2D(num_classes, (1, 1), activation='sigmoid')(up0b)

    model = Model(inputs=inputs, outputs=classify)

    # model.compile(optimizer=SGD(lr=0.01, momentum=0.9), loss='binary_crossentropy', metrics=[dice_loss])

    return model