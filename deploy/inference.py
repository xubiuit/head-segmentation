from keras.models import model_from_json
import cv2
import numpy as np
import tensorflow as tf
import os

INPUT_WIDTH = 512
INPUT_HEIGHT = 512

class TF_Portrait():
    def __init__(self, model_file, weight_file):
        self.model_file = model_file
        self.weight_file = weight_file
        self.seg_model = None
        self.init_model()

    def init_model(self):
        '''
        load model from file
        :param model_file: json file that defines the network graph
        :param weight_file: weight file that stores the weights of the graph
        :return: model object
        '''

        with open(self.model_file, 'r') as json_file:
            loaded_model_json = json_file.read()
            self.seg_model = model_from_json(loaded_model_json)
            # print self.seg_model
            # print os.path.isfile(self.weight_file)
            if not os.path.exists(self.weight_file):
                raise RuntimeError('No weights file founded.')
            self.seg_model.load_weights(self.weight_file)

            self.graph = tf.get_default_graph()

    def inference(self, raw_img):
        H, W = raw_img.shape[:2]
        img = cv2.resize(raw_img, (INPUT_WIDTH, INPUT_HEIGHT), interpolation=cv2.INTER_LINEAR)
        img = img.astype(np.float32) / 255.0
        img = img.reshape(1, *img.shape)

        with self.graph.as_default():

            pred = self.seg_model.predict(img, batch_size=1)[0]

            prob = cv2.resize(pred, (W, H))

            mask = prob > 0.5

            mask = np.dstack((mask,) * 3).astype(np.uint8)

            res = mask * raw_img

            # convert background color from black(0) to white(255)
            res_3chan = (res + 255 * np.logical_and(np.logical_not(mask), np.logical_not(res))).astype(np.uint8)

            red, green, blue = cv2.split(raw_img)
            res_4chan = cv2.merge([red, green, blue, 255 * mask[..., 0]])

            return res_4chan