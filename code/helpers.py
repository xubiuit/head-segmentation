from sklearn.metrics import fbeta_score
import numpy as np
import cv2
import imutils
import keras.backend as K
# import matplotlib.pylab as plt

ORIG_WIDTH = 1918
ORIG_HEIGHT = 1280

def comp_mean(imglist):
    mean = [0, 0, 0]
    for img in imglist:
        mean += np.mean(np.mean(img, axis=0), axis=0)
    return mean/len(imglist)

def load_param():

    thresh = [[0.03, 0.03, 0.05, 0.07, 0.03, 0.02, 0.05, 0.03, 0.05, 0.05, 0.04, 0.03, 0.05, 0.1, 0.04, 0.04, 0.06],
     [0.05, 0.03, 0.09, 0.08, 0.03, 0.02, 0.05, 0.08, 0.04, 0.05, 0.02, 0.03, 0.03, 0.07, 0.04, 0.06, 0.05],
     [0.04, 0.03, 0.05, 0.06, 0.02, 0.04, 0.05, 0.05, 0.03, 0.05, 0.04, 0.03, 0.03, 0.11, 0.04, 0.05, 0.06],
     [0.02, 0.03, 0.06, 0.1, 0.03, 0.01, 0.06, 0.05, 0.04, 0.1, 0.05, 0.03, 0.03, 0.1, 0.04, 0.05, 0.1],
     [0.04, 0.03, 0.04, 0.06, 0.03, 0.03, 0.05, 0.09, 0.03, 0.07, 0.07, 0.04, 0.04, 0.08, 0.03, 0.06, 0.09]]
    val_score = 0.93065441478548683

    return thresh, val_score

def find_f_measure_threshold2(probs, labels, num_iters=100, seed=0.21):
    _, num_classes = labels.shape[0:2]
    best_thresholds = [seed] * num_classes
    best_scores = [0] * num_classes
    for t in range(num_classes):

        thresholds = list(best_thresholds)  # [seed]*num_classes
        for i in range(num_iters):
            th = i / float(num_iters)
            thresholds[t] = th
            f2 = fbeta_score(labels, probs > thresholds, beta=2, average='samples')
            if f2 > best_scores[t]:
                best_scores[t] = f2
                best_thresholds[t] = th
        print('\t(t, best_thresholds[t], best_scores[t])=%2d, %0.3f, %f' % (t, best_thresholds[t], best_scores[t]))
    print('')
    return best_thresholds, best_scores


def normalize(img):
    img = img.astype(np.float16)

    img[:, :, 0] = (img[:, :, 0] - 103.94) * 0.017
    img[:, :, 1] = (img[:, :, 1] - 116.78) * 0.017
    img[:, :, 2] = (img[:, :, 2] - 123.68) * 0.017
    # img = np.expand_dims(img, axis=0)
    return img


def transformations(src, choice):
    if choice == 0:
        # Rotate 90
        src = cv2.rotate(src, rotateCode=cv2.ROTATE_90_CLOCKWISE)
    if choice == 1:
        # Rotate 90 and flip horizontally
        src = cv2.rotate(src, rotateCode=cv2.ROTATE_90_CLOCKWISE)
        src = cv2.flip(src, flipCode=1)
    if choice == 2:
        # Rotate 180
        src = cv2.rotate(src, rotateCode=cv2.ROTATE_180)
    if choice == 3:
        # Rotate 180 and flip horizontally
        src = cv2.rotate(src, rotateCode=cv2.ROTATE_180)
        src = cv2.flip(src, flipCode=1)
    if choice == 4:
        # Rotate 90 counter-clockwise
        src = cv2.rotate(src, rotateCode=cv2.ROTATE_90_COUNTERCLOCKWISE)
    if choice == 5:
        # Rotate 90 counter-clockwise and flip horizontally
        src = cv2.rotate(src, rotateCode=cv2.ROTATE_90_COUNTERCLOCKWISE)
        src = cv2.flip(src, flipCode=1)
    return src

def transformations2(src, choice):
    mode = choice // 2
    src = imutils.rotate(src, mode * 90)
    if choice % 2 == 1:
        src = cv2.flip(src, flipCode=1)
    return src

def randomShiftScaleRotate(image, mask,
                           shift_limit=(-0.0625, 0.0625),
                           scale_limit=(-0.1, 0.1),
                           rotate_limit=(-45, 45), aspect_limit=(0, 0),
                           borderMode=cv2.BORDER_CONSTANT, u=0.5, factor=1):
    if np.random.random() < u:
        height, width, channel = image.shape

        angle = np.random.uniform(rotate_limit[0], rotate_limit[1])  # degree
        scale = np.random.uniform(1 + scale_limit[0], 1 + scale_limit[1])
        aspect = np.random.uniform(1 + aspect_limit[0], 1 + aspect_limit[1])
        sx = scale * aspect / (aspect ** 0.5)
        sy = scale / (aspect ** 0.5)
        dx = round(np.random.uniform(shift_limit[0], shift_limit[1]) * width)
        dy = round(np.random.uniform(shift_limit[0], shift_limit[1]) * height)

        cc = np.math.cos(angle / 180 * np.math.pi) * sx
        ss = np.math.sin(angle / 180 * np.math.pi) * sy
        rotate_matrix = np.array([[cc, -ss], [ss, cc]])

        box0 = np.array([[0, 0], [width, 0], [width, height], [0, height], ])
        box1 = box0 - np.array([width / 2, height / 2])
        box1 = np.dot(box1, rotate_matrix.T) + np.array([width / 2 + dx, height / 2 + dy])

        box0 = box0.astype(np.float32)
        box1 = box1.astype(np.float32)
        mat = cv2.getPerspectiveTransform(box0, box1)
        image = cv2.warpPerspective(image, mat, (width, height), flags=cv2.INTER_LINEAR, borderMode=borderMode,
                                    borderValue=(
                                        0, 0,
                                        0,))
        mask = cv2.warpPerspective(mask, mat, (width, height), flags=cv2.INTER_LINEAR, borderMode=borderMode,
                                   borderValue=(
                                       0, 0,
                                       0,))

    return image, mask


def randomHorizontalFlip(image, mask, u=0.5):
    if np.random.random() < u:
        image = cv2.flip(image, 1)
        mask = cv2.flip(mask, 1)

    return image, mask

# def rle(img):
#     '''
#     img: numpy array, 1 - mask, 0 - background
#     Returns run length as string formated
#     '''
#     bytes = np.where(img.flatten() == 1)[0]
#     runs = []
#     prev = -2
#     for b in bytes:
#         if (b > prev + 1): runs.extend((b + 1, 0))
#         runs[-1] += 1
#         prev = b
#
#     return ' '.join([str(i) for i in runs])

# https://www.kaggle.com/stainsby/fast-tested-rle
def run_length_encode(mask):
    '''
    img: numpy array, 1 - mask, 0 - background
    Returns run length as string formated
    '''
    inds = mask.flatten()
    runs = np.where(inds[1:] != inds[:-1])[0] + 2
    runs[1::2] = runs[1::2] - runs[:-1:2]
    rle = ' '.join([str(r) for r in runs])
    return rle


# def dice(im1, im2, empty_score=1.0):
#     im1 = im1.astype(np.bool)
#     im2 = im2.astype(np.bool)
#
#     if im1.shape != im2.shape:
#         raise ValueError("Shape mismatch: im1 and im2 must have the same shape.")
#
#     im_sum = im1.sum() + im2.sum()
#     if im_sum == 0:
#         return empty_score
#
#     intersection = np.logical_and(im1, im2)
#     return 2. * intersection.sum() / im_sum

def dice_loss(y_true, y_pred):
    smooth = 1.
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

def get_score(train_masks, avg_masks, thr):
    d = 0.0
    for i in range(train_masks.shape[0]):
        pred_mask = avg_masks[i][:,:,1] - avg_masks[i][:,:,0]
        pred_mask[pred_mask > thr] = 1
        pred_mask[pred_mask <= thr] = 0
        d += dice_loss(train_masks[i], pred_mask)
    return d/train_masks.shape[0]


def get_result(imgs, thresh):
    result = []
    for img in imgs:
        img[img > thresh] = 1
        img[img <= thresh] = 0
        result.append(cv2.resize(img, (1918, 1280), interpolation=cv2.INTER_LINEAR))
    return result

def get_final_mask(preds, thresh=0.5):
    result = []
    for pred in preds:
        prob = cv2.resize(pred, (ORIG_WIDTH, ORIG_HEIGHT))
        mask = prob > thresh
        result.append(mask)
    return result

def find_best_seg_thr(masks_gt, masks_pred):
    best_score = 0
    best_thr = -1
    for t in range(-200, 200):
        thr = t/float(1000)
        score = get_score(masks_gt, masks_pred, thr)
        print('THR: {:.3f} SCORE: {:.6f}'.format(thr, score))
        if score > best_score:
            best_score = score
            best_thr = thr

    print('Best score: {} Best thr: {}'.format(best_score, best_thr))
    return best_score, best_thr


def draw(img, mask):
    img_masked = cv2.bitwise_and(img, img, mask=mask)

    print("Image shape: {} | image type: {} | mask shape: {} | mask type: {}".format(img.shape, img.dtype, mask.shape, mask.dtype) )

    plt.subplot(131)
    plt.imshow(img)
    plt.subplot(132)
    plt.imshow(mask)
    plt.subplot(133)
    plt.imshow(img_masked)