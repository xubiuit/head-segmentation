# head-segmentation
Head segmentation with Keras.
Find the demo here: https://www.youtube.com/watch?v=sOOAfrPwrNQ&t=1s

## update
pspnet2
- train with 5780 samples
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

1013+dilated

- use dilated convolution layers for the center block
- IoU is 0.9689 on kk-252 testset.

1024

- correct ground truth annotation
- training record: 
```
2520s - loss: 0.1011 - dice_score: 0.9619 - weightedLoss: 0.7321 - bce_dice_loss: 0.1011 - val_loss: 0.1267 - val_dice_score: 0.9561 - val_weightedLoss: 0.7388 - val_bce_dice_loss: 0.1267

... ...

Epoch 00031: reducing learning rate to 1.00000001169e-08.
2518s - loss: 0.0990 - dice_score: 0.9628 - weightedLoss: 0.7310 - bce_dice_loss: 0.0990 - val_loss: 0.1300 - val_dice_score: 0.9570 - val_weightedLoss: 0.7376 - val_bce_dice_loss: 0.1300
```

- IoU is 0.969942334371 on kk-252 testset.

1108
- rule out bad training and test samples
- training record:
```
2717s - loss: 0.0990 - dice_score: 0.9634 - weightedLoss: 0.7290 - bce_dice_loss: 0.0990 - val_loss: 0.1177 - val_dice_score: 0.9602 - val_weightedLoss: 0.7334 - val_bce_dice_loss: 0.1177
Epoch 16/100
```
- IoU is 0.974359960128 on kk-111 testset

1116 (v2.3)
- add more data and retrain
- training record:
```
Epoch 00027: reducing learning rate to 1.00000001169e-08.
2491s - loss: 0.0923 - dice_score: 0.9616 - weightedLoss: 0.7364 - bce_dice_loss: 0.0923 - val_loss: 0.0954 - val_dice_score: 0.9602 - val_weightedLoss: 0.7383 - val_bce_dice_loss: 0.0954
Epoch 00027: early stopping
```
- IoU is 0.972726227225 on kk-327 testset

1120 (v2.3u)
- portrait segmentation
- training record:
```
Epoch 30/100
921s - loss: 0.1084 - dice_score: 0.9678 - weightedLoss: 0.6071 - bce_dice_loss: 0.1084 - val_loss: 0.1480 - val_dice_score: 0.9608 - val_weightedLoss: 0.6143 - val_bce_dice_loss: 0.1480
```
- IoU is 0.969091977837 on portrait-352 testset

1218 (v2.4)
- add more data and retrain (5780+4199=9979 train & validation images)
- training record:
```
2844s - loss: 0.0858 - dice_score: 0.9648 - weightedLoss: 0.7299 - bce_dice_loss: 0.0858 - val_loss: 0.0791 - val_dice_score: 0.9650 - val_weightedLoss: 0.7288 - val_bce_dice_loss: 0.0791
```
- IoU is 0.974909528025 on head-467 testset

0123 (v2.5)
- use TiramisuNet
- training record:
```
 - 5692s - loss: 0.0718 - dice_score: 0.9704 - weightedLoss: 0.7157 - bce_dice_loss: 0.0718 - val_loss: 0.1115 - val_dice_score: 0.9524 - val_weightedLoss: 0.7410 - val_bce_dice_loss: 0.1115
```
- IoU is 0.95XXXX on head-467 testset

0211 (v2.6)
- add refineNet after the 1st stage UNet (references: Deep Image Matting)
- training record:
``` 
- 4803s - loss: 0.1698 - conv2d_66_loss: 0.0865 - average_1_loss: 0.0833 - val_loss: 0.1561 - val_conv2d_66_loss: 0.0790 - val_average_1_loss: 0.0771
```
- IoU is 0.972961458863 after 1st Unet stage, and 0.973500913981 after 2nd refineNet stage on head-467 testset

0321 (v2.7)
- clean code
- retrain pspnet2 with 9979 samples
- training record:
```
Epoch 26/100
 - 671s - loss: 0.1303 - dice_score: 0.9447 - val_loss: 0.1190 - val_dice_score: 0.9451
 ...
 Epoch 32/100
 - 673s - loss: 0.1311 - dice_score: 0.9449 - val_loss: 0.1198 - val_dice_score: 0.9456
Epoch 00032: early stopping
```
- IoU: 0.954327620642 on head-467 testset

0323 (v3.0)
- add more data (5780+8387=14167 train & validation images) to retrain 1218's UNet
- IoU: 0.974027543933 on head-932 testset
