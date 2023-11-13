# Readme: Sigma Delta YOLO

This is the readme for sigma-delta YOLO training and inference on GPU. The examples make use of the __object detetction (OBD) modules __ in `lava.lib.dl.slayer`. For additional description about the module, click [here](https://github.com/lava-nc/lava-dl/blob/main/src/lava/lib/dl/slayer/object_detection/README.md).

Here we make use of sigma-delta ReLU spiking neuron to exploit temporal redundancy in video object detection. For a basic level introduction of sigma-delta network and it's training, refer to [PilotNet SDNN training example](https://github.com/lava-nc/lava-dl/blob/main/tutorials/lava/lib/dl/slayer/pilotnet/train.ipynb).

## Training

The YOLO training makes use of `train_sdnn.py`. It is a general YOLO SDNN training script which can be configured to use different available YOLO models, hyperparameters and initializations. The python script form enables easy hyperparameter exploration for your use cases to get the best out of the network architecure.

The arguments are described in [hyperparameters](#hyperparameters).

### 1. How to train your network from scratch?
```bash
python train_sdnn_base.py # +additional hyperparameters
```

### 2. How to choose the YOLO SDNN network archiecture?
Select using `-model` argument. The value can be `tiny_yolov3_str` or `yolo_kp`.

`tiny_yolov3_str` means strided TinyYOLOv3 architecture.
```bash
python train_sdnn.py -model tiny_yolov3_str  # +additional hyperparameters
```

`yolo_kp` means 8 chip Kapoho Point YOLO architecture.
```bash
python train_sdnn.py -model yolo_kp  # +additional hyperparameters
```

### 3. How to warmstart using pre-trained model?
Select using `-load` argument. `-load slayer` or `-load lava-dl` will load pretrained model included in Lava-DL SLAYER. Path to a `*.pt` file will initialize the model using the states saved in the file.
```bash
python train_sdnn.py -load lava-dl  # +additional hyperparameters
```

## Example training
The training command for the two models included in this example were trained using the following.

1. __Sigma-Delta TinyYOLOv3str model__ training command
```bash
python train_sdnn.py -model tiny_yolov3_str -load slayer \
                     -num_workers 16 -epoch 200 -lr 0.0001 -lrf 0.01 -warmup 40 \
                     -lambda_coord 2 -lambda_noobj 4 -lambda_obj 1.8 -lambda_cls 1 -lambda_iou 2.25 \
                     -alpha_iou 0.8 -clip 1 -label_smoothing 0.03 -tgt_iou_thr 0.25 -aug_prob 0.4 \
                     -track_iter 100 -sparsity -sp_lam 0.01 -sp_rate 0.01
```
2. __Sigma-Delta YOLO-KP model__ training command
```bash
python train_sdnn.py -model yolo_kp -load slayer \
                     -epoch 200 -lr 0.0001 -lrf 0.01 -warmup 40 \
                     -lambda_coord 2 -lambda_noobj 4 -lambda_obj 1.8 -lambda_cls 1 -lambda_iou 2.25 \
                     -alpha_iou 0.8 -clip 1 -label_smoothing 0.03 -tgt_iou_thr 0.25 -aug_prob 0.4 \
                     -track_iter 100 -sparsity -sp_lam 0.01 -sp_rate 0.01
```
> Note: By default the dataset is expected in `data/` folder.

## Inference (GPU)
The detailed inference example of trained sigma-delta network on GPU is described in [inference.ipynb](https://github.com/lava-nc/lava-dl/blob/main/tutorials/lava/lib/dl/slayer/tiny_yolo_sdnn/inference.ipynb).

## Hyperparameters
| Arg name | Description |
|----------|-------------|
|||
|`gpu`    | Which gpu(s) to use |
|`b`      | Batch size for dataloader |
|`verbose`| Enables lots of debug printouts |
|||
| __Model__ ||
|`model` | Network model. It can be `tiny_yolov3_str` or  `yolo_kp`|
|||
| __Sparsity__ ||
|`sparsity` | Enable sparsity loss |
|`sp_lam`   | Sparsity loss mixture ratio |
|`sp_rate`  | Minimum rate for sparsity penalization |
|||
| __Optimizer__ ||
|`lr`  | Initial learning rate |
|`wd`  | Optimizer weight decay |
|`lrf` | Learning rate reduction factor for lr scheduler |
|||
| __Network/SDNN__ ||
|`threshold`  | Neuron threshold |
|`tau_grad`   | Surrogate gradient time constant |
|`scale_grad` | Surrogate gradient scale |
|`clip`       | Gradient clipping limit |
|||
| __Pretrained model__ ||
|`load` | Model file path or `slayer` or `lava-dl` to warmstart the network. In the latter two cases, pretrained model in `lava.lib.dl.slayer.obd` will be loaded. |
|||
| __Target generation__ ||
|`tgt_iou_thr` | IoU threshold to ignore overlapping bounding box in YOLO target generation |
|||
| __YOLO loss__ ||
|`lambda_coord`    | YOLO coordinate loss lambda |
|`lambda_noobj`    | YOLO no-object loss lambda |
|`lambda_obj`      | YOLO object loss lambda |
|`lambda_cls`      | YOLO class loss lambda |
|`lambda_iou`      | YOLO iou loss lambda |
|`alpha_iou`       | YOLO loss object target iou mixture factor |
|`label_smoothing` | YOLO class cross entropy label smoothing |
|`track_iter`      | YOLO loss tracking interval. It helps track the progression of each individual loss. |
|||
| __Experiment tracking__ ||
|`exp`   | Experiment differentiater string |
|`seed`  | Random seed of the experiment |
|||
| __Training__ ||
|`epoch` | Number of epochs to run |
|`warmup` | Number of epochs to warmup the learning rate |
|||
| __Dataset__ ||
|`dataset`     | Dataset to use. Currently supported: [BDD100K](https://bdd-data.berkeley.edu/) |
|`path`        | Dataset path |
|`aug_prob`    | Training augmentation probability |
|`output_dir`  | Directory in which to put log folders |
|`num_workers` | Number of dataloader workers |
|`clamp_max`   | Exponential clamp in height/width calculation |