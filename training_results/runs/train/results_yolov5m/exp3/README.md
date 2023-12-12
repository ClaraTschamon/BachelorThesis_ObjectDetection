Vorheriger Versuch hat immernoch overfittet. Anhand des result.png von vorhin 
wurden 70 Epochen ausgewählt um Overfitting loszuwerden.

```
!python train.py \
--data 'data.yaml' \
--weights yolov5m.pt \
--img 640 \
--epochs 70 \
--batch-size 64 \
--cfg 'models/yolov5m.yaml' \
--cache disk \
--device 0
```