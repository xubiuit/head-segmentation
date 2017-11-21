python generate_imagelist.py  ./upperbody upperbody_imagelist.txt
wc -l upperbody_imagelist.txt
python split_train_test_list.py upperbody_imagelist.txt 0.9 v2.3u