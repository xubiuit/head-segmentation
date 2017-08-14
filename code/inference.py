from keras.models import model_from_json
import cv2
import numpy as np

def init(model_file, weight_file):
    '''
    load model from file
    :param model_file: json file that defines the network graph
    :param weight_file: weight file that stores the weights of the graph
    :return: model object
    '''
    json_file = open(model_file, 'r')
    loaded_model_json = json_file.read()
    model = model_from_json(loaded_model_json)
    model.load_weights(weight_file)
    return model

def inference(image, model):
    '''
    predict the mask given specific image and the model object
    :param image: [1, 512, 512, 3] 'float32' input image normalized to [0, 1]
    :param model: model object
    :return: [1, 512, 512, 1] 'float32' output mask
    '''
    # print image.shape
    # print image.dtype
    return model.predict(image, batch_size=1)[0]


if __name__ == '__main__':
    model = init('../weights/model.json', '../weights/head-segmentation-model.h5')
    list_file = '../input/name.txt'
    ids_test = []
    with open(list_file, 'r') as f:
        for line in f:
            ids_test.append(line.strip())

    i = 0
    raw_img = cv2.imread('../' + ids_test[i])
    H, W = raw_img.shape[:2]
    img = cv2.resize(raw_img, (512, 512), interpolation=cv2.INTER_LINEAR)
    img = img.astype(np.float32) / 255.0
    img = img.reshape(1, *img.shape)
    pred = inference(img, model)
    # print pred.shape
    # print pred.dtype

    prob = cv2.resize(pred[0], (W, H))

    mask = prob > 0.5

    mask = np.dstack((mask,)*3).astype(np.uint8)
    cv2.imwrite('../tmp_res.png', mask*raw_img)
