Verwendet wurde dieser Datensatz:  
https://universe.roboflow.com/deneme-iq93l/chessv1-sswl9  
Die Bilder, in denen die Bounding Boxes verschoben sind, wurden aussortiert.
Übrig geblieben sind 6588 Bilder.
Preprocessing Step 'Resize Fit (white edges)' wurde angewandt,
um die Bilder alle auf 640x640 Pixel zu bringen.

Paramter nach diesen Tipps ausgewählt:  
https://docs.ultralytics.com/yolov5/tutorials/tips_for_best_training_results/

```
!python train.py \
--data 'data.yaml' \
--weights yolov5m.pt \
--img 640 \
--epochs 300 \
--batch-size 64 \
--cfg 'models/yolov5m.yaml' \
--cache disk \
--device 0
```
