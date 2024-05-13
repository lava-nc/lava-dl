# Readme: sigma delta Yolo - micro version

sdn-YOLO-KP model training, pruning, and fine tuning on GPU and inference on GPU/Loihi. The examples make use of the __object detetction (OBD) modules __ in `lava.lib.dl.slayer`. For additional description about the module, click [here](https://github.com/lava-nc/lava-dl/blob/main/src/lava/lib/dl/slayer/object_detection/README.md).

Here we make use of sigma-delta ReLU spiking neuron to exploit temporal redundancy in video object detection. For a basic level introduction of sigma-delta network and it's training, refer to [PilotNet SDNN training example](https://github.com/lava-nc/lava-dl/blob/main/tutorials/lava/lib/dl/slayer/pilotnet/train.ipynb).

pretrain &amp; prune model yolo-KP -> the result is yolo-KP4 which after fine tune on BDD outperform yolo-KP (with 4x less ops and mem footprint)

## Training

The YOLO training makes use of `train_sdnnYolo.py`. It is a general YOLO SDNN training script which can be configured to use different available YOLO models, hyperparameters and initializations. The python script form enables easy hyperparameter exploration for your use cases to get the best out of the network architecure.

The arguments are described in [hyperparameters](#hyperparameters) below:

## Hyperparameters
| Arg name | Description |
|----------|-------------|
|||
|`gpu`    | Which gpu(s) to use |
|`b`      | Batch size for dataloader |
|||
| __Model__ ||
|`model` | Network model = {`single_head_KP` (default), `short_single_head_KP`, `dual_head`, `multi_head`}|
|`Heads` | [list of] head ids activated only if model above is set to `dual_head`/`multi_head` |
|||
| __Pretrained model__ ||
|`load` | Model file path. The model is a torch weight file |
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
| __Network/SDNN parameters__ ||
|`threshold`  | Neuron threshold |
|`tau_grad`   | Surrogate gradient time constant |
|`scale_grad` | Surrogate gradient scale |
|`clip`       | Gradient clipping limit |
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
|`output_dir`      | directory(s) in which to put log folders |
|`strID`           | str ID to attach to file name |
|`seed`            | Random seed of the experiment |
|`verbose`         | Lots of debug printouts |
|`print_summary`   | Prints model summary and exits |  
|||
| __Training__ ||
|`epoch` | Number of epochs to run |
|`warmup` | Number of epochs to warmup the learning rate |
|||
| __Dataset__ ||
|`dataset`     | Dataset to use. {COCO|BDD100K} |
|`path`        | Dataset path |
|`aug_prob`    | Training augmentation probability |
|`output_dir`  | Directory in which to put log folders |
|`num_workers` | Number of dataloader workers |
|`clamp_max`   | Exponential clamp in height/width calculation |