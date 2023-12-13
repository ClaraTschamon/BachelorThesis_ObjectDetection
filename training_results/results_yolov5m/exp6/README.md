Vorheriges weniger lang trainiert um overfitting
bei val_ObjectLoss loszuwerden.

```
!python train.py \
--data 'data.yaml' \
--weights yolov5m.pt \
--img 640 \
--epochs 25 \
--batch-size 64 \
--cfg 'models/yolov5m.yaml' \
--cache disk \
--device 0
```