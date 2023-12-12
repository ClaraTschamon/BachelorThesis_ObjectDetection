Trainieren auf größeren Datensatz!

```
#select from modelzoo here: https://github.com/facebookresearch/detectron2/blob/master/MODEL_ZOO.md#coco-object-detection-baselines
#config parameters explained: https://detectron2.readthedocs.io/en/latest/modules/config.html#config-references
#hypterparameter tuning: https://www.researchgate.net/figure/Detectron2-hyperparameter-values_tbl4_363233537
# fix detectron2 error: https://github.com/facebookresearch/detectron2/commit/fc9c33b1f6e5d4c37bbb46dde19af41afc1ddb2a

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

# Number of images per batch across all machines.
cfg.SOLVER.IMS_PER_BATCH = 8  # Increase batch size for faster convergence
cfg.SOLVER.BASE_LR = 0.001  # Learning Rate

cfg.SOLVER.WARMUP_ITERS = 1000
cfg.SOLVER.MAX_ITER = 8000  # Iterations
#cfg.SOLVER.MAX_ITER = 100  # Train for more iterations for better performance
cfg.SOLVER.STEPS = (4000, 6000, 7000)  # Adjust the learning rate schedule
cfg.SOLVER.GAMMA = 0.1  # Reduce LR by a factor of 0.1 at each step

cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 13 #your number of classes + 1 (superclass)

cfg.TEST.EVAL_PERIOD = 200 # No. of iterations after which the Validation Set is evaluated. 

# Save the configuration to a YAML file
f = open('config.yml','w')
f.write(cfg.dump())
f.close()
os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
trainer = CocoTrainer(cfg)
trainer.resume_or_load(resume=False)
trainer.train()
```