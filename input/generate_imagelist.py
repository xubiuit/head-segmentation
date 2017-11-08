#!/usr/local/python2.7/bin/python2.7
# -*- coding: utf-8 -*-

import os, sys, pickle
def getList(path, suffix_tuple = ('jpg', 'png')):
    file_list=[]
    for root, dirs, files in os.walk(path):
        for file in files:
            if file.endswith(suffix_tuple):
                if file not in file_list:
                    f = os.path.join(root, file)
                    file_list.append(f)
    return file_list

def getLists(paths, suffix_tuple = ('jpg', 'png')):
    file_list = []
    for path in paths:
        file_list.extend(getList(path, suffix_tuple))
    return file_list

if __name__ == '__main__':
    cwd = os.getcwd()
    folder = str(sys.argv[1])
    input_folders = os.path.join(cwd, folder)
    imagelist = getList(input_folders)

    with open(os.path.join(cwd, 'imagelist.txt'), 'w') as fl:
        pickle.dump(imagelist, fl)
