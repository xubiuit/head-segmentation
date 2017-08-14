# -*- coding: utf-8 -*-
__author__ = 'Zhenyuan Shen: https://kaggle.com/szywind'

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os, time, gc, imutils, cv2
from keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint, TensorBoard
from keras.preprocessing.image import ImageDataGenerator
from tqdm import tqdm
from keras import optimizers

# from sklearn.cross_validation import KFold, StratifiedKFold
from sklearn.model_selection import KFold

from helpers import *
import newnet
import math
import glob
import random
from PIL import Image
from sklearn.model_selection import train_test_split
import unet

np.set_printoptions(threshold='nan')

INPUT_PATH = '../input/'
OUTPUT_PATH = '../output/'



class HeadSeg():
    def __init__(self, input_dim=512, batch_size=5, epochs=100, learn_rate=1e-2, nb_classes=2):
        self.input_dim = input_dim
        self.batch_size = batch_size
        self.epochs = epochs
        self.learn_rate = learn_rate
        self.nb_classes = nb_classes
        # self.model = newnet.fcn_32s(input_dim, nb_classes)
        self.model = unet.get_unet_512(input_shape=(self.input_dim, self.input_dim, 3))

        with open('../weights/model.json', 'w') as json_file:
            json_file.write(self.model.to_json())
        self.model_path = '../weights/head-segmentation-model.h5'
        self.threshold = 0.5
        self.direct_result = True
        # self.nAug = 2 # incl. horizon mirror augmentation
        # self.nTTA = 1 # incl. horizon mirror augmentation
        self.load_data()
        self.factor = 1
        self.train_with_all = False
        self.apply_crf = False

    def load_data(self):
        ids_train = []
        with open(INPUT_PATH + 'trainSet.txt', 'r') as f:
            for line in f:
                ids_train.append(line.strip().split())
        self.ids_train_split, self.ids_valid_split = train_test_split(ids_train, test_size=0.2, random_state=42)

        # index = list(range(len(self.train_imgs)))
        # random.shuffle(index)
        # self.train_masks = self.train_masks[index]
        # self.train_masks = self.train_masks[index]

    def train(self):

        # train_datagen = ImageDataGenerator(
        #     rescale=1. / 255,
        #     zoom_range=0.15,
        #     rotation_range=360,
        #     width_shift_range=0.1,
        #     height_shift_range=0.1
        # )
        # val_datagen = ImageDataGenerator(rescale=1. / 255)

        # train_datagen.fit(x_train, augment=True, rounds=2, seed=1)
        # train_generator = train_datagen.flow(x_train[train_index], y_train[train_index], shuffle=True, batch_size=batch_size, seed=int(time.time()))
        # val_generator = val_datagen.flow(x_train[test_index], y_train[test_index], shuffle=False, batch_size=batch_size)

        nTrain = len(self.ids_train_split)
        nValid = len(self.ids_valid_split)
        print('Training on {} samples'.format(nTrain))
        print('Validating on {} samples'.format(nValid))

        def train_generator():
            while True:
                for start in range(0, nTrain, self.batch_size):
                    x_batch = []
                    y_batch = []
                    end = min(start + self.batch_size, nTrain)
                    ids_train_batch = self.ids_train_split[start:end]

                    for img_path, mask_path in ids_train_batch:
                        # j = np.random.randint(self.nAug)
                        img = cv2.imread(img_path)
                        img = cv2.resize(img, (self.input_dim, self.input_dim), interpolation=cv2.INTER_LINEAR)
                        # img = transformations2(img, j)
                        mask = cv2.imread(mask_path)[...,0]
                        mask = cv2.resize(mask, (self.input_dim, self.input_dim), interpolation=cv2.INTER_LINEAR)
                        # mask = transformations2(mask, j)
                        img, mask = randomShiftScaleRotate(img, mask,
                                                           shift_limit=(-0.0625, 0.0625),
                                                           scale_limit=(-0.125, 0.125),
                                                           rotate_limit=(-0, 0))
                        img, mask = randomHorizontalFlip(img, mask)
                        img = randomGammaCorrection(img)
                        if self.factor != 1:
                            img = cv2.resize(img, (self.input_dim/self.factor, self.input_dim/self.factor), interpolation=cv2.INTER_LINEAR)
                        # draw(img, mask)

                        if self.direct_result:
                            mask = np.expand_dims(mask, axis=2)
                            x_batch.append(img)
                            y_batch.append(mask)
                        else:
                            target = np.zeros((mask.shape[0], mask.shape[1], self.nb_classes))
                            for k in range(self.nb_classes):
                                target[:,:,k] = (mask == k)
                            x_batch.append(img)
                            y_batch.append(target)

                    x_batch = np.array(x_batch, np.float32) / 255.0
                    y_batch = np.array(y_batch, np.float32)
                    yield x_batch, [y_batch, y_batch]

        def valid_generator():
            while True:
                for start in range(0, nValid, self.batch_size):
                    x_batch = []
                    y_batch = []
                    end = min(start + self.batch_size, nValid)
                    ids_valid_batch = self.ids_valid_split[start:end]
                    for img_path, mask_path in ids_valid_batch:
                        img = cv2.imread(img_path)
                        img = cv2.resize(img, (self.input_dim, self.input_dim), interpolation=cv2.INTER_LINEAR)
                        mask = cv2.imread(mask_path)[...,0]
                        mask = cv2.resize(mask, (self.input_dim, self.input_dim), interpolation=cv2.INTER_LINEAR)
                        if self.factor != 1:
                            img = cv2.resize(img, (self.input_dim/self.factor, self.input_dim/self.factor), interpolation=cv2.INTER_LINEAR)
                        if self.direct_result:
                            mask = np.expand_dims(mask, axis=2)
                            x_batch.append(img)
                            y_batch.append(mask)
                        else:
                            target = np.zeros((mask.shape[0], mask.shape[1], self.nb_classes))
                            for k in range(self.nb_classes):
                                target[:,:,k] = (mask == k)
                            x_batch.append(img)
                            y_batch.append(target)

                    x_batch = np.array(x_batch, np.float32) / 255.0
                    y_batch = np.array(y_batch, np.float32)
                    yield x_batch, [y_batch, y_batch]


        # opt  = optimizers.SGD(lr=self.learn_rate, momentum=0.9)
        # self.model.compile(loss='binary_crossentropy', # We NEED binary here, since categorical_crossentropy l1 norms the output before calculating loss.
        #                   optimizer=opt,
        #                   metrics=[dice_loss])

        self.model.compile(optimizer=optimizers.SGD(lr=self.learn_rate, momentum=0.9),
                           loss=['binary_crossentropy', dice_loss],
                           metrics=[dice_loss])

        callbacks = [EarlyStopping(monitor='val_loss',
                                       patience=5,
                                       verbose=1,
                                       min_delta=1e-4),
                    ReduceLROnPlateau(monitor='val_loss',
                                           factor=0.1,
                                           patience=2,
                                           cooldown=2,
                                           verbose=1),
                    ModelCheckpoint(filepath=self.model_path,
                                         save_best_only=True,
                                         save_weights_only=True),
                    TensorBoard(log_dir='logs')]


        self.model.fit_generator(
            generator=train_generator(),
            steps_per_epoch=math.ceil(nTrain / float(self.batch_size)),
            epochs=10,
            verbose=2,
            callbacks=callbacks,
            validation_data=valid_generator(),
            validation_steps=math.ceil(nValid / float(self.batch_size)))


        opt  = optimizers.SGD(lr=0.1*self.learn_rate, momentum=0.9)
        self.model.compile(loss=['binary_crossentropy', dice_loss], # We NEED binary here, since categorical_crossentropy l1 norms the output before calculating loss.
                          optimizer=opt,
                          metrics=[dice_loss])


        self.model.fit_generator(
            generator=train_generator(),
            steps_per_epoch=math.ceil(nTrain / float(self.batch_size)),
            epochs=self.epochs - 10,
            verbose=2,
            callbacks=callbacks,
            validation_data=valid_generator(),
            validation_steps=math.ceil(nValid / float(self.batch_size)))

    def test(self):
        if not os.path.isfile(self.model_path):
            raise RuntimeError("No model found.")
        self.model.load_weights(self.model_path)

        df_test = pd.read_csv(INPUT_PATH + 'sample_submission.csv')
        test_imgs = np.array(df_test['img'])

        nTest = len(test_imgs)
        print('Testing on {} samples'.format(nTest))

        test_splits = 8  # Split test set (number of splits must be multiple of 2)
        ids_test_splits = np.split(test_imgs, indices_or_sections=test_splits)

        rles = []
        split_count = 0
        for test_x in ids_test_splits:
            split_count += 1
            nTestBatch = len(test_x)
            def test_generator():
                while True:
                    for start in range(0, nTestBatch, self.batch_size):
                        x_batch = []
                        end = min(start + self.batch_size, nTestBatch)

                        for i in range(start, end):
                            img = cv2.imread(INPUT_PATH + 'test/{}'.format(test_x[i]))
                            img = cv2.resize(img, (self.input_dim/self.factor, self.input_dim/self.factor), interpolation=cv2.INTER_LINEAR)
                            x_batch.append(img)
                        x_batch = np.array(x_batch, np.float32) / 255.0
                        yield x_batch

            print("Predicting on {} samples (split {}/{})".format(nTestBatch, split_count, test_splits))
            preds = self.model.predict_generator(generator=test_generator(),
                                                 steps=math.ceil(nTestBatch / float(self.batch_size)))
            preds = np.squeeze(preds, axis=3)

            print("Generating masks...")
            result = []
            for pred in tqdm(preds, miniters=1000):
                prob = cv2.resize(pred, (ORIG_WIDTH, ORIG_HEIGHT))
                mask = prob > self.threshold
                rle = run_length_encode(mask)
                rles.append(rle)
                result.append(mask)

            # # save predicted masks
            if not os.path.exists(OUTPUT_PATH):
                os.mkdir(OUTPUT_PATH)

            for i in range(nTestBatch):
                cv2.imwrite(OUTPUT_PATH + '{}'.format(test_x[i]), (255 * result[i]).astype(np.uint8))
        print("Generating submission file...")
        df_test['rle_mask'] = rles
        df_test.to_csv('../submit/submission.csv.gz', index=False, compression='gzip')

    def test_one(self, list_file='lfw-deepfunneled.txt'):
        if not os.path.isfile(self.model_path):
            raise RuntimeError("No model found.")
        self.model.load_weights(self.model_path)

        ids_test = []
        with open(INPUT_PATH + list_file, 'r') as f:
            for line in f:
                ids_test.append(line.strip())
        nTest = len(ids_test)
        print('Testing on {} samples'.format(nTest))

        if not os.path.isfile(self.model_path):
            raise RuntimeError("No model found.")

        self.model.load_weights(self.model_path)

        print('Create submission...')
        str = []
        nbatch = 0
        for start in range(0, nTest, self.batch_size):
            print(nbatch)
            nbatch += 1
            x_batch = []
            images = []
            end = min(start + self.batch_size, nTest)
            for i in range(start, end):
                raw_img = cv2.imread('../' + ids_test[i])
                img = cv2.resize(raw_img, (self.input_dim/self.factor, self.input_dim/self.factor), interpolation=cv2.INTER_LINEAR)
                x_batch.append(img)
                images.append(raw_img)
            x_batch = np.array(x_batch, np.float32) / 255.0
            p_test = self.model.predict(x_batch, batch_size=self.batch_size)[0]

            if self.direct_result:
                result, probs = get_final_mask(p_test, self.threshold, apply_crf=self.apply_crf, images=images)
            else:
                avg_p_test = p_test[...,1] - p_test[...,0]
                result = get_result(avg_p_test, 0)


            str.extend(map(run_length_encode, result))

            # save predicted masks
            if not os.path.exists(OUTPUT_PATH):
                os.mkdir(OUTPUT_PATH)

            for i in range(start, end):
                img_path = ids_test[i][ids_test[i].rfind('/')+1:]
                cv2.imwrite(OUTPUT_PATH + '{}'.format(img_path), (255 * probs[i-start]).astype(np.uint8))

        print("Generating submission file...")
        df = pd.DataFrame({'img': ids_test, 'rle_mask': str})
        df.to_csv('../submit/submission.csv.gz', index=False, compression='gzip')


if __name__ == "__main__":
    ccs = HeadSeg()

    ccs.train()
    # ccs.test_one(list_file='lfw-deepfunneled.txt')