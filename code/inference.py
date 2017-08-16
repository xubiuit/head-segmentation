from keras.models import model_from_json
import cv2
import numpy as np
import matplotlib.pyplot as plt
import subprocess

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
    model = init('../weights/model_unet512.json', '../weights/head-segmentation-model_unet512.h5')
    list_file = '../input/expo.txt'
    ids_test = []
    with open(list_file, 'r') as f:
        for line in f:
            ids_test.append(line.strip())

    for i in range(len(ids_test)):
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

        res = mask * raw_img

        # convert background color from black(0) to white(255)
        res = (res + 255*np.logical_and(np.logical_not(mask), np.logical_not(res))).astype(np.uint8)

        # show image
        f, (ax1, ax2) = plt.subplots(1, 2, sharey=True)
        ax1.axis('off')
        ax1.imshow(cv2.cvtColor(raw_img, cv2.COLOR_BGR2RGB), vmax=1.5, vmin=-0.4)
        ax1.set_title('Input')
        ax2.axis('off')
        ax2.imshow(cv2.cvtColor(res, cv2.COLOR_BGR2RGB))
        ax2.set_title('Output')

        plt.savefig('../output/demo/{}.jpg'.format(i))
        cv2.imwrite('../tmp_res.png', mask*raw_img)
    subprocess.call([
        'ffmpeg', '-framerate', '4', '-i', '../output/demo/%d.jpg', '-r', '30', '-pix_fmt', 'yuv1080p', '../headseg_demo.mp4'
    ])
