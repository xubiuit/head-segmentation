import cv2, os
import numpy as np
import shutil

np.set_printoptions(threshold='nan')

img = cv2.imread('/home/jin/Desktop/0a6df7a3cf48030f25b87d9a2ad9f9b5_mask.png')

def getList(path, suffix_tuple = ('jpg', 'png')):
    file_list=[]
    for root, dirs, files in os.walk(path):
        for file in files:
            if file.endswith(suffix_tuple):
                if file not in file_list:
                    f = os.path.join(root, file)
                    file_list.append(f)
    return file_list

def mask_process(masks_folder):
    img_files = getList(masks_folder, suffix_tuple='jpg')
    for img_file in img_files:
        os.remove(img_file)
    mask_files = getList(masks_folder, suffix_tuple='png')
    cnt = 0
    for mask_file in mask_files:
        cnt += 1
        img = cv2.imread(mask_file)
        if np.max(img) < 240:
            continue
        res = (img >= 100).astype(np.uint8)
        assert(np.max(res) == 1)
        os.remove(mask_file)
        if mask_file.find('_mask.png') == -1:
            pos = mask_file.rfind('.png')
            mask_file = mask_file[:pos] + '_mask.png'
        cv2.imwrite(mask_file, res)

    print(len(mask_files))
    print(cnt)

def generate_list(images_folder, masks_folder):
    mask_files = getList(masks_folder)
    image_files = getList(images_folder)

    ### TODO: pair mask file path with image file path

if __name__ == '__main__':
    # masks_folder = '/home/jin/VSS/koutou/data/kk/mask'
    # mask_process(masks_folder)
    #
    # images_folder = '/home/jin/VSS/koutou/data/kk/image'
    # generate_list(images_folder, masks_folder)

    # masks_folder = '../input/data/20171101-20171108-mask'
    masks_folder = '../input/head/20180322-mask'
    mask_process(masks_folder)

    # images_folder = '../input/20171107/src201709'
    # generate_list(images_folder, masks_folder)



def kk_65mask():
    path = '/home/jin/VSS/koutou/data/MASK_65'
    path1 = '/home/jin/VSS/koutou/data/kk/mask/20170915-20170926_all'
    path2 = '/home/jin/VSS/koutou/data/kk/mask/20170927-20170929_all'

    folder1 = path1[path1.rfind('/')+1:]
    folder2 = path2[path2.rfind('/')+1:]

    l = os.listdir(path)
    l1 = os.listdir(path1)
    l2 = os.listdir(path2)

    # add suffix '_mask' to the files
    for mask in l:
        pos = mask.rfind('.png')
        shutil.move(os.path.join(path, mask), os.path.join(path, mask[:pos] + '_mask.png'))

    # assign every file to the corresponding folder
    l = os.listdir(path)
    if not os.path.exists(os.path.join(path, folder1)):
        os.mkdir(os.path.join(path, folder1))
    if not os.path.exists(os.path.join(path, folder2)):
        os.mkdir(os.path.join(path, folder2))

    for mask in l:
        if mask in l1:
            shutil.copy(os.path.join(path, mask), os.path.join(path, folder1, mask))
            # os.remove(os.path.join(path, mask))
        elif mask in l2:
            shutil.copy(os.path.join(path, mask), os.path.join(path, folder2, mask))
            # os.remove(os.path.join(path, mask))
        else:
            raise RuntimeError('Error: no file found.')
