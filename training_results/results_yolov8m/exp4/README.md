Den größeren Datensatz verwenden!

```
model = YOLO('yolov8m.pt')
results = model.train(data='/data1/home/tscl/Chess-Pieces-3-2/data.yaml', imgsz=640,  epochs=25, batch=32, cache=True, device=0)
```