import os
import sys
import pickle
import numpy as np
from skimage.io import imread, imsave

def transform_image(mask_list):
    for mask in mask_list:
        image = imread(mask)
        image[image > 0] = 1
        imsave(mask, image)

def split_sample_and_gt(input_file):
    '''
    Get image list and mask list.
    :param input_file: 
    :return: 
    '''
    with open(input_file, 'r') as f:
        images = pickle.load(f)
        print(len(images))
        portraits = [i for i in images if i.find('_mask') == -1]
        masks = [i for i in images if i.find('_mask') != -1]
    return portraits, masks

def pair_sample_and_gt(portrait_list, mask_list):
    '''
    Get sample-mask pairs.
    :param portrait_list: 
    :param mask_list: 
    :return: 
    '''
    # dict{maskname: corresponding_portrait_path}
    mask_dict = {}
    for portrait in portrait_list:
        imgname = portrait[portrait.rfind('/')+1:]
        pos = imgname.rfind('.')
        imgid = imgname[:pos]
        print('imgname: ', imgname)
        maskname = imgid + '_mask' + '.png' # + imgname[pos:]
        print('maskname: ', maskname)
	maskname_list = map(lambda x: x[x.rfind('/')+1:], mask_list)

        if maskname in maskname_list:
            mask_dict[maskname] = portrait
    all_list = []
    for mask in mask_list:
        maskname = mask[mask.rfind('/') + 1:]
        
        #print('maskname: ', maskname)
        if maskname in mask_dict:
            all_list.append([mask_dict[maskname], mask])
    return all_list

def split_train_test_set(all_list, ratio = 1):
    '''
    Split training and test sets from all.
    :param all_list: 
    :param ratio: 
    :return: 
    '''
    np.random.shuffle(all_list)
    N = len(all_list)
    Ntr = int(N * ratio)
    Nt = N - Ntr

    print("# of train samples: ", Ntr)
    print("# of test samples: ", Nt)

    cwd = os.getcwd()
    train_file = os.path.join(cwd, 'trainSet.txt')
    test_file = os.path.join(cwd, 'testSet.txt')
    if os.path.exists(train_file):
        os.remove(train_file)
    if os.path.exists(test_file):
	os.remove(test_file)

    with open(train_file, 'w') as trFile:
        for i in xrange(Ntr):
            trFile.write(all_list[i][0] + '\t' + all_list[i][1] + '\n')

    with open(test_file, 'w') as tFile:
        for i in xrange(Ntr, N):
            tFile.write(all_list[i][0] + '\t' + all_list[i][1] + '\n')
    return

if __name__ == "__main__":
    filename = str(sys.argv[1])
    portrait_list, mask_list = split_sample_and_gt(filename)
    print(portrait_list)
    #transform_image(mask_list)
    all_list = pair_sample_and_gt(portrait_list, mask_list)
    print(len(all_list))
    split_train_test_set(all_list, 0.9)
