
```
from detectron2.config import get_cfg
import os

cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_X_101_32x8d_FPN_3x.yaml"))
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/faster_rcnn_X_101_32x8d_FPN_3x.yaml")  # Let training initialize from model zoo
cfg.MODEL.DEVICE = 'cuda:0'

cfg.DATASETS.TRAIN = ("my_dataset_train",)
cfg.DATASETS.TEST = ("my_dataset_test",)
cfg.DATASETS.VALID = ("my_dataset_val",)

# Number of data loading threads
cfg.DATALOADER.NUM_WORKERS = 4

cfg.SOLVER.IMS_PER_BATCH = 2  # Increase batch size for faster convergence
cfg.SOLVER.BASE_LR = 0.001  # Learning Rate

cfg.SOLVER.WARMUP_ITERS = 1500
cfg.SOLVER.MAX_ITER = 8000  # Iterations

cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 13 #your number of classes + 1 (superclass)

cfg.TEST.EVAL_PERIOD = 200 # No. of iterations after which the Validation Set is evaluated. 

os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
trainer = CocoTrainer(cfg)
trainer.resume_or_load(resume=False)
trainer.train()
```