Verwendet wurde dieser Datensatz:  
https://universe.roboflow.com/deneme-iq93l/chessv1-sswl9  
Die Bilder, in denen die Bounding Boxes verschoben sind, wurden aussortiert.
Übrig geblieben sind 6588 Bilder.
Preprocessing Step 'Resize Fit (white edges)' wurde angewandt,
um die Bilder alle auf 640x640 Pixel zu bringen.


```
from ultralytics import YOLO
model = YOLO('yolov8m.pt')
results = model.train(data='/data1/home/tscl/Chess-Pieces-2-5/data.yaml', 
imgsz=640, epochs=100, batch=16, cache=True, device=0)
```
