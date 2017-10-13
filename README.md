# head-segmentation
Head segmentation with Keras.
Find the demo here: https://www.youtube.com/watch?v=sOOAfrPwrNQ&t=1s

## update
pspnet2

- IoU is 0.9236 on kk-252 testset

0915

- IoU is 0.9411 on kk-252 testset.

0918

- IoU is 0.949 on kk-252 testset.

1010

- add one more activation after last (1,1) conv layer in each block
- IoU is 0.9517 on kk-252 testset.

1010+large

- use large input size (768,576) instead of (512,512)
- IoU is 0.9522 on kk-252 testset.

1011

- add more kk data on the same condition as the testset to the training set
- IoU is 0.9647 on kk-252 testset.

1013

- use dilated convolution layers for the center block
- IoU is 0.9689 on kk-252 testset.
