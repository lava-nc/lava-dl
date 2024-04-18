import os
from datetime import datetime
import yaml

import torch
from torch.utils.data import DataLoader

from lava.lib.dl import slayer
from lava.lib.dl.slayer import obd


if __name__ == '__main__':
    # inference_model = 'tiny_yolov3_str'  # Loihi compatible TinyYOLOv3 network
    inference_model = 'yolo_kp'          # Customized model tragetted for 8 chip Kapoho Point form factor

    args = slayer.utils.dotdict(load=f'/home/lecampos/leo-internal/lava-dl/src/lava/lib/dl/slayer/object_detection/models/Trained_{inference_model}/network.pt')
    trained_folder = os.path.dirname(args.load)
    print(trained_folder)

    with open(trained_folder + '/args.txt', 'rt') as f:
        model_args = slayer.utils.dotdict(yaml.safe_load(f))
        for (k, v) in model_args.items():
            if k not in args.keys():
                args[k] = v
                
    print('Using GPUs {}'.format(args.gpu))
    device = torch.device('cuda:{}'.format(args.gpu[0]))
    print()
    print('Hyperparameters')
    print('===============')
    for k,v in args.items():
        print(f'{k} : {v}')
        
    if inference_model == 'tiny_yolov3_str':
        Network = obd.models.tiny_yolov3_str.Network
    elif inference_model == 'yolo_kp':
        Network = obd.models.yolo_kp.Network
    else:
        raise RuntimeError

    net = Network(threshold=model_args.threshold,
                tau_grad=model_args.tau_grad,
                scale_grad=model_args.scale_grad,
                num_classes=11,
                clamp_max=model_args.clamp_max).to(device)
    net.init_model((448, 448))
    # net.load_state_dict(torch.load(args.load))
    
    yolo_target = obd.YOLOtarget(anchors=net.anchors,
                             scales=net.scale,
                             num_classes=net.num_classes,
                             ignore_iou_thres=model_args.tgt_iou_thr)
    
    test_set = obd.dataset.BDD(root='/home/lecampos/leo-internal/lava-dl/tutorials/lava/lib/dl/slayer/tiny_yolo_sdnn/data/bdd100k', 
                               dataset='track', train=False, randomize_seq=False, seq_len=200)

    test_loader = DataLoader(test_set,
                            batch_size=1,
                            shuffle=True,
                            collate_fn=yolo_target.collate_fn,
                            num_workers=4,
                            pin_memory=True)
    
    epoch = 0
    stats = slayer.utils.LearningStats(accuracy_str='AP@0.5')
    t_st = datetime.now()
    ap_stats = obd.bbox.metrics.APstats(iou_threshold=0.5)

    for i, (inputs, targets, bboxes) in enumerate(test_loader):
        net.eval()

        with torch.no_grad():
            print('inputs ', inputs.shape)
            inputs = inputs.to(device)
            predictions, counts, latent_space = net(inputs, latent_space_backcounter=1)

            T = inputs.shape[-1]
            
            predictions = [obd.bbox.utils.nms_ls(predictions[..., t], latent_space[..., t]) for t in range(T)]
            
            print("predictions ", len(predictions), predictions[0][0].shape)
        

            break

