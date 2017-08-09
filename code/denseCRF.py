import pydensecrf.densecrf as dcrf

from pydensecrf.utils import compute_unary, create_pairwise_bilateral, \
    create_pairwise_gaussian, softmax_to_unary, unary_from_softmax

import matplotlib.pyplot as plt

import skimage.io as io

import numpy as np
import cv2
import os

INPUT_PATH = '/home/yzm/shenzhenyuan/AI/projects/koutou'
OUTPUT_PATH = '/home/yzm/shenzhenyuan/AI/projects/koutou'


def denseCRF(image, final_probabilities):

    # softmax = final_probabilities.squeeze()

    softmax = final_probabilities.transpose((2, 0, 1))

    # The input should be the negative of the logarithm of probability values
    # Look up the definition of the softmax_to_unary for more information
    unary = unary_from_softmax(softmax)

    # The inputs should be C-continious -- we are using Cython wrapper
    unary = np.ascontiguousarray(unary)

    d = dcrf.DenseCRF2D(image.shape[1], image.shape[0], 2)
    # d = dcrf.DenseCRF(image.shape[0] * image.shape[1], 2)

    d.setUnaryEnergy(unary)


    # # This potential penalizes small pieces of segmentation that are
    # # spatially isolated -- enforces more spatially consistent segmentations
    # feats = create_pairwise_gaussian(sdims=(3, 3), shape=image.shape[:2])
    #
    # d.addPairwiseEnergy(feats, compat=3,
    #                     kernel=dcrf.DIAG_KERNEL,
    #                     normalization=dcrf.NORMALIZE_SYMMETRIC)
    #
    # # This creates the color-dependent features --
    # # because the segmentation that we get from CNN are too coarse
    # # and we can use local color features to refine them
    # feats = create_pairwise_bilateral(sdims=(80, 80), schan=(13, 13, 13),
    #                                    img=image, chdim=2)
    #
    # d.addPairwiseEnergy(feats, compat=10,
    #                      kernel=dcrf.DIAG_KERNEL,
    #                      normalization=dcrf.NORMALIZE_SYMMETRIC)

    d.addPairwiseGaussian(sxy=3, compat=3)
    d.addPairwiseBilateral(sxy=80, srgb=13, rgbim=image, compat=10)
    Q = d.inference(50)

    res = np.argmax(Q, axis=0).reshape((image.shape[0], image.shape[1]))

    # cmap = plt.get_cmap('bwr')
    #
    # f, (ax1, ax2) = plt.subplots(1, 2, sharey=True)
    # ax1.imshow(res, vmax=1.5, vmin=-0.4, cmap=cmap)
    # ax1.set_title('Segmentation with CRF post-processing')
    # probability_graph = ax2.imshow(np.dstack((final_probabilities[:,:,0],)*3))
    # ax2.set_title('Original Prediction Mask')
    # plt.show()
    return res,Q


def getImageList(input_folder='/home/yzm/shenzhenyuan/AI/projects/koutou'):
    imgList = []
    for root, dirs, files in os.walk(input_folder):
        for file in files:
            if file.endswith('jpg'):
                imgList.append(os.path.join(root,file))
    return imgList

def cropHead(Q, img):
    mask = np.array(Q)[0].reshape(img.shape[:2])
    mask = 255*mask
    mask = mask.astype(np.uint8)
    ret, th = cv2.threshold(mask, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    # return np.dstack((th/255,)*3)*img
    r_channel, g_channel, b_channel = cv2.split(img)
    img_rgba = cv2.merge((r_channel, g_channel, b_channel, th))
    return img_rgba

def main():
    imgList = getImageList(input_folder='/home/jin/shenzhenyuan/head-segmentation/input/test')

    for img_path in imgList:
        img = cv2.imread('{}'.format(img_path))
        if img_path[:img_path.rfind('.')].endswith('png'):
            str = img_path[:img_path.rfind('.')] + '-seg.png'
        else:
            str = img_path[:img_path.rfind('.')] + '.png-seg.png'
        mask = cv2.imread('{}'.format(str))
        prob = mask[:,:,0:2] / 255.0
        prob[:, :, 1] = 1 - prob[:, :, 0]
        res, Q = denseCRF(img, prob)
        a = 1-res
        a = a.astype('uint8')

        r_channel, g_channel, b_channel = cv2.split(img)
        img_rgba = cv2.merge((r_channel, g_channel, b_channel, a*255))
        cv2.imwrite('{}_crf.png'.format(img_path[:img_path.find('.')]), img_rgba)

        # a = np.dstack((a,)*3)
        # plt.imshow(a*img)
        # cv2.imwrite('{}_crf.png'.format(img_path[:img_path.find('.')]), (a>0.1)*img)

        cv2.imwrite('{}_crf_qtsu.png'.format(img_path[:img_path.find('.')]), cropHead(Q, img))

if __name__ == '__main__':
    main()



