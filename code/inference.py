from keras.models import model_from_json
import cv2
import numpy as np
import matplotlib.pyplot as plt
import subprocess

INPUT_WIDTH = 512 # 576 # 512
INPUT_HEIGHT = 512 # 768 # 512
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
    # return model.predict(image, batch_size=1)
    return model.predict(image, batch_size=1)[0]


if __name__ == '__main__':
    model = init('../weights/koutou_tf_0123/model.json', '../weights/koutou_tf_0123/head-segmentation-model.h5')
    list_file = '../input/expo.txt'
    folder_dir = ''
    # list_file = '../input/kk0915.txt'
    # folder_dir = 'input/kk0915/kk0915/'
    ids_test = []
    with open(list_file, 'r') as f:
        for line in f:
            ids_test.append(folder_dir + line.strip())

    for i in range(len(ids_test)):
        print(i)
        raw_img = cv2.imread('../' + ids_test[i])
        H, W = raw_img.shape[:2]
        img = cv2.resize(raw_img, (INPUT_WIDTH, INPUT_HEIGHT), interpolation=cv2.INTER_LINEAR)
        img = img.astype(np.float32) / 255.0
        img = img.reshape(1, *img.shape)
        pred = inference(img, model)
        # print pred.shape
        # print pred.dtype

        prob = cv2.resize(pred, (W, H))

        mask = prob > 0.5

        mask = np.dstack((mask,)*3).astype(np.uint8)

        res = mask * raw_img

        # convert background color from black(0) to white(255)
        res_3chan = (res + 255*np.logical_and(np.logical_not(mask), np.logical_not(res))).astype(np.uint8)

        # show image
        f, (ax1, ax2) = plt.subplots(1, 2, sharey=True)
        ax1.axis('off')
        ax1.imshow(cv2.cvtColor(raw_img, cv2.COLOR_BGR2RGB), vmax=1.5, vmin=-0.4)
        ax1.set_title('Input')
        ax2.axis('off')

        red, green, blue = cv2.split(raw_img)
        res_4chan = cv2.merge([red, green, blue, 255*mask[...,0]])

        ax2.imshow(cv2.cvtColor(res_4chan, cv2.COLOR_BGR2RGBA))
        ax2.set_title('Output')

        plt.savefig('../output/tmp/{}.jpg'.format(i))
        # cv2.imwrite('../tmp_res.png', mask*raw_img)
        print(res_4chan.shape)
        cv2.imwrite('../tmp_res.png', res_4chan)
        # from PIL.Image import Image
        # from scipy.misc import imsave
        # imsave('../' + ids_test[i], res_4chan)
        pos = ids_test[i].rfind('.')
        cv2.imwrite('../'+ids_test[i][:pos]+'.png', res_4chan)
    subprocess.call([
        'ffmpeg', '-framerate', '4', '-i', '../output/tmp/%d.jpg', '-r', '30', '-pix_fmt', 'yuv420p', '../headseg_demo.mp4'
    ])
