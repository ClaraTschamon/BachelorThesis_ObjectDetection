Frage: warum sind Ergebnisse von yolov5 besser?

Versucht mit größerer Batchsize. 32 statt 16. 64 hat nicht funktioniert wegen CUDA out of memory.

```
model = YOLO('yolov8m.pt')
results = model.train(data='/data1/home/tscl/Chess-Pieces-2-5/data.yaml', imgsz=640,  epochs=75, batch=32, cache=True, device=0)
```
