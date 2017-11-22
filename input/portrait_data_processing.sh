python generate_imagelist.py  ./portrait portrait_imagelist.txt
wc -l portrait_imagelist.txt
python split_train_test_list.py portrait_imagelist.txt 0.9 v2.3u