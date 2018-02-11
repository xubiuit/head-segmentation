# python generate_imagelist.py  ./headtraning
# wc -l imagelist.txt
# python split_train_test_list.py imagelist.txt

python generate_imagelist.py  ./head head_imagelist.txt
wc -l head_imagelist.txt
python split_train_test_list.py head_imagelist.txt 0.9 v2.4