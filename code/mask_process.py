import cv2, os
import numpy as np

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
    mask_files = getList(masks_folder)
    for mask_file in mask_files:
        img = cv2.imread(mask_file)
        res = (img >= 100).astype(np.uint8)
        cv2.imwrite(mask_file, res)

def generate_list(images_folder, masks_folder):
    mask_files = getList(masks_folder)
    image_files = getList(images_folder)

    ### TODO: pair mask file path with image file path

if __name__ == '__main__':
    masks_folder = '/home/jin/VSS/koutou/data/kk/mask'
    # mask_process(masks_folder)

    images_folder = '/home/jin/VSS/koutou/data/kk/image'
    generate_list(images_folder, masks_folder)