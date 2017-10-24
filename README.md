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
