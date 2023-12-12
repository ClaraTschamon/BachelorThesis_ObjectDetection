Vorheriger Versuch hat Overfittet. Anhand des result.png von vorhin 
wurden 95 Epochen ausgewählt, um Overfitting loszuwerden.

```
!python train.py \
--data 'data.yaml' \
--weights yolov5m.pt \
--img 640 \
--epochs 95 \
--batch-size 64 \
--cfg 'models/yolov5m.yaml' \
--cache disk \
--device 0
```
