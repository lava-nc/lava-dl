import os
import argparse
from typing import Any, Dict, Tuple
import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from datetime import datetime
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter
from torchinfo import summary


from lava.lib.dl import slayer


# import sys
# path = ['/home/dbendaya/work/ContinualLearning/tinyYolov3_lava/YOLOsdnn/',
#          '/home/dbendaya/_work_/ContinualLearning/tinyYolov3_lava/YOLOsdnn/']
# sys.path.extend(path)

from lava.lib.dl.slayer import object_detection
from object_detection.boundingbox.utils import storeData
from object_detection.dataset.bdd100k import BDD
from object_detection.dataset.utils import collate_fn
from yolo_base import YOLOBase, YOLOLoss, YOLOtarget
from object_detection.boundingbox.metrics import APstats
from object_detection.boundingbox.utils import (
    Height, Width, annotation_from_tensor, mark_bounding_boxes,
)
from object_detection.boundingbox.utils import non_maximum_suppression as nms


# DATASET FOLDER LOCATIONS (on different servers) or set -dataset <path>
BDD100K_path = ['/export/share/datasets/BDD100K/MOT2020/bdd100k',
                '/data-raid/sshresth/data/bdd100k/MOT2020/bdd100k']

yourFavouriteDataset_path = []


BOX_COLOR_MAP = [(np.random.randint(256),
                  np.random.randint(256),
                  np.random.randint(256))
                 for i in range(80)]

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '-gpu', type=int, default=[0], help='which gpu(s) to use', nargs='+')
    parser.add_argument('-b', type=int, default=32,
                        help='batch size for dataloader')
    # Sparsity
    parser.add_argument('-sparsity', default=False,
                        action='store_true', help='enable sparsity loss')
    parser.add_argument('-sp_lam', type=float, default=0.01,
                        help='sparsity loss mixture ratio')
    parser.add_argument('-sp_rate', type=float, default=0.01,
                        help='sparsity penalization rate')
    # Optimizer
    parser.add_argument('-lr', type=float, default=0.001,
                        help='initial learning rate')
    parser.add_argument('-wd', type=float, default=1e-5,
                        help='optimizer weight decay')
    parser.add_argument('-lrf', type=float, default=0.01,
                        help='learning rate factor for lr scheduler')
    # Network/SDNN
    parser.add_argument('-threshold', type=float,
                        default=0.1, help='neuron threshold')
    parser.add_argument('-tau_grad', type=float, default=0.1,
                        help='surrogate gradient time constant')
    parser.add_argument('-scale_grad', type=float,
                        default=0.2, help='surrogate gradient scale')
    parser.add_argument('-clip', type=float, default=10,
                        help='gradient clipping limit')
    # Pretrained model
    parser.add_argument('-load', type=str, default='',
                        help='pretrained model', nargs='+')
    # Target generation
    parser.add_argument('-tgt_iou_thr', type=float, default=0.5,
                        help='ignore iou threshold in target generation')
    # YOLO loss
    parser.add_argument('-lambda_coord', type=float,
                        default=1.0, help='YOLO coordinate loss lambda')
    parser.add_argument('-lambda_noobj', type=float,
                        default=10.0, help='YOLO no-object loss lambda')
    parser.add_argument('-lambda_obj', type=float,
                        default=5.0, help='YOLO object loss lambda')
    parser.add_argument('-lambda_cls', type=float,
                        default=1.0, help='YOLO class loss lambda')
    parser.add_argument('-lambda_iou', type=float,
                        default=1.0, help='YOLO iou loss lambda')
    parser.add_argument('-alpha_iou', type=float, default=0.25,
                        help='YOLO loss object target iou mixture factor')
    parser.add_argument('-label_smoothing', type=float, default=0.10,
                        help='YOLO class cross entropy label smoothing')
    parser.add_argument('-track_iter', type=int,
                        default=1000, help='YOLO loss tracking interval')
    parser.add_argument('-seed', type=int,   default=None,
                        help='random seed of the experiment')
    # Training
    parser.add_argument('-epoch', type=int,   default=50,
                        help='number of epochs to run')
    parser.add_argument('-warmup', type=int,   default=10,
                        help='number of epochs to warmup')
    # dataset
    parser.add_argument('-dataset', type=str,   default='BDD',
                        help='dataset to use [yourFavouriteDataset, BDD100K]')
    parser.add_argument('-path', type=str,   default=None,
                        help='dataset path')  # this one will be set to
    parser.add_argument('-aug_prob', type=float, default=0.2,
                        help='training augmentation probability')
    parser.add_argument('-output_dir', type=str,   default=".",
                        help="directory in which to put log folders")
    parser.add_argument('-num_workers', type=int,   default=12,
                        help="number of dataloader workers")
    parser.add_argument('-clamp_max', type=float,   default=5.0,
                        help="exponential clamp in height/width calculation")
    parser.add_argument('-verbose', default=False,
                        action='store_true', help='lots of debug printouts')
    parser.add_argument('-train', default=False,
                        action='store_true', help='activate training')
    parser.add_argument('-print_summary', default=False,
                        action='store_true', help='Print model summary and exit')
    parser.add_argument('-strID', type=str, default='SDNN_',
                        help='str ID to attach to file name')
    parser.add_argument('-DVSlike', default=False, action='store_true',
                        help='activate random image jitter subtraction')
    parser.add_argument('-model', type=str, default='single_head_KP',
                        help='(list of) model(-heads) to load [dual_head, single_head_KP, short_single_head_KP, ...]', nargs='+')
    parser.add_argument('-Heads', type=int, default=None,
                        help="[list of] head id", nargs='+')

    args = parser.parse_args()
    # import network
    vars(args)['combined_models'] = len(args.model)
    if args.combined_models > 1:  # multiple heads loading the combined head model
        from models.sdnn_combine_single_heads_KP import Network, SparsityMonitor
    else:
        exec('from models.sdnn_%s import Network, SparsityMonitor' %
             args.model[0])

    # selecting path according to its existance - see above for including another server path
    DATASET_path = yourFavouriteDataset_path if args.dataset == 'yourDataset' else BDD100K_path
    args.path = [g for g in DATASET_path if os.path.exists(
        g)][0] if not args.path else args.path
    print(f'loading dataset from {args.path}')

    if args.train:
        from datetime import datetime
        date_str = str(datetime.now())[
            :-9].replace(':', '').replace('-', '').replace('.', '').replace(' ', '')
        identifier = args.strID + date_str
        if args.seed is not None:
            torch.manual_seed(args.seed)
            identifier += '_{}'.format(args.seed)

        trained_folder = args.output_dir + '/Trained_' + \
            identifier if len(identifier) > 0 else args.output_dir + '/Trained'
        os.makedirs(trained_folder, exist_ok=True)
        print(trained_folder)
        writer = SummaryWriter(args.output_dir + '/runs/' + identifier)
        vars(args)['trained_folder'] = trained_folder
        with open(trained_folder + '/args.txt', 'wt') as f:
            for arg, value in sorted(vars(args).items()):
                f.write('{} : {}\n'.format(arg, value))

    print('Using GPUs {}'.format(args.gpu))
    device = torch.device('cuda:{}'.format(args.gpu[0]))

    # add here your favouriteDataset # classes
    classes_output = {'BDD100K': 11}
    print('Classes output:', classes_output[args.dataset])
    vars(args)['num_classes'] = classes_output[args.dataset]

    if args.sparsity:
        print('Sparsity loss on!')
        sparsity_monitor = SparsityMonitor(
            max_rate=args.sp_rate, lam=args.sp_lam)
    else:
        sparsity_monitor = None

    print('making net')
    if len(args.gpu) == 1:
        if args.combined_models > 1:
            net = Network(args=args).to(device)
        else:
            net = Network(threshold=args.threshold,
                          tau_grad=args.tau_grad,
                          scale_grad=args.scale_grad,
                          num_classes=args.num_classes,
                          clamp_max=args.clamp_max).to(device)
        module = net
    else:
        if args.combined_models > 1:
            net = Network(args=args).to(device)
        else:
            net = torch.nn.DataParallel(Network(threshold=args.threshold,
                                        tau_grad=args.tau_grad,
                                        scale_grad=args.scale_grad,
                                        num_classes=args.classes_output,
                                        clamp_max=args.clamp_max).to(device),
                                        device_ids=args.gpu)
        module = net.module
    override_anchors = module.anchors.clone()

    if args.Heads:
        override_anchors = override_anchors[args.Heads]

    print(f'loading net on {device}')
    if args.load != '':
        if args.combined_models > 1:  # combined_model
            for k, load_model in enumerate(args.load):
                print(f'Initializing model from {load_model}')
                exec(f'module.Head{k}.load_model(load_model)')
                exec(f'module.Head{k}.to(device)')
        else:
            print(f'Initializing model from {args.load[0]}')
            module.load_model(args.load[0])
        # need to force the intended heads onto the model if loaded from file given another head
        if args.train:
            module.anchors = override_anchors
            net.scale = None
        module.to(device)

    print('module.init_model')
    module.init_model((448, 448))

    # Define optimizer module.
    # optimizer = torch.optim.RAdam(net.parameters(),
    #                               lr=args.lr,
    #                               weight_decay=args.wd)
    print("optimizer")
    optimizer = torch.optim.Adam(net.parameters(),
                                 lr=args.lr,
                                 weight_decay=args.wd)
    # optimizer = torch.optim.SGD(net.parameters(),
    #                             lr=args.lr,
    #                             weight_decay=args.wd)

    # Define learning rate scheduler
    def lf(x): return min(x / args.warmup, 1) * \
        ((1 + np.cos(x * np.pi / args.epoch)) / 2) * (1 - args.lrf) + args.lrf

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)
    yolo_target = YOLOtarget(anchors=net.anchors,
                             scales=net.scale,
                             num_classes=net.num_classes,
                             ignore_iou_thres=args.tgt_iou_thr)

    print('dataset')
    elif args.dataset == 'BDD100K':
        if args.train:
            train_set = BDD(root=args.path, dataset='track', train=True,
                            augment_prob=args.aug_prob, randomize_seq=True, image_jitter=args.DVSlike)
            train_loader = DataLoader(train_set,
                                      batch_size=args.b,
                                      shuffle=True,
                                      collate_fn=yolo_target.collate_fn,
                                      num_workers=args.num_workers,
                                      pin_memory=True)
        test_set = BDD(root=args.path, dataset='track', train=False,
                       randomize_seq=True, image_jitter=args.DVSlike)
        test_loader = DataLoader(test_set,
                                 batch_size=args.b,
                                 shuffle=False,
                                 collate_fn=yolo_target.collate_fn,
                                 num_workers=args.num_workers,
                                 pin_memory=True)

    print('yolo_loss')
    yolo_loss = YOLOLoss(anchors=net.anchors,
                         lambda_coord=args.lambda_coord,
                         lambda_noobj=args.lambda_noobj,
                         lambda_obj=args.lambda_obj,
                         lambda_cls=args.lambda_cls,
                         lambda_iou=args.lambda_iou,
                         alpha_iou=args.alpha_iou,
                         label_smoothing=args.label_smoothing).to(device)

    print('stats')
    stats = slayer.utils.LearningStats(accuracy_str='AP@0.5')

    print('loss_tracker')
    loss_tracker = dict(coord=[], obj=[], noobj=[], cls=[], iou=[])
    loss_order = ['coord', 'obj', 'noobj', 'cls', 'iou']

    # function mean and Std
    def m_s(x): return '%.2f±%.2f' % (np.mean(x), np.std(x))
    rate_count_test = []

    (summary(module, (args.b, 3, 448, 448)),
     os.exit()) if args.print_summary else None

    # print('ON!: net grads frozen')
    # for name, params in module.named_parameters():
    #     if "blocks.8" in name:
    #         break
    #     params.requires_grad = False
    # for name, params in module.named_parameters():
    #     print(name, params.requires_grad)

    print('train loop') if args.train else None

    from tqdm import tqdm
    args.epoch = 1 if not args.train else args.epoch
    pbar_epoch = tqdm(range(args.epoch), desc="epoch")
    for epoch in pbar_epoch:
        if args.train:
            t_st = datetime.now()
            ap_stats = APstats(iou_threshold=0.5)
            # print(f'{epoch=}')
            pbar = tqdm(train_loader, desc='Train', leave=False,
                        bar_format="{desc}: {percentage:.1f}%|{bar}|{n_fmt}/{total_fmt} {postfix}")
            for i, (inputs, targets, bboxes) in enumerate(pbar):

                print(f'{i=}') if args.verbose else None

                net.train()
                print('inputs') if args.verbose else None
                inputs = inputs.to(device)

                print('forward') if args.verbose else None
                predictions, counts = net(inputs, sparsity_monitor)

                loss, loss_distr = yolo_loss(predictions, targets)
                if sparsity_monitor is not None:
                    loss += sparsity_monitor.loss
                    sparsity_monitor.clear()

                if torch.isnan(loss):
                    print("loss is nan, continuing")
                    continue
                optimizer.zero_grad()
                loss.backward()
                net.validate_gradients()
                torch.nn.utils.clip_grad_norm_(net.parameters(), args.clip)
                optimizer.step()
                scheduler.step()

                if i < 10:
                    net.grad_flow(path=trained_folder + '/')

                # MAP calculations
                T = inputs.shape[-1]
                try:
                    # predictions = torch.concat([net.yolo(predictions[0], net.anchors[0]),
                    #                             net.yolo(predictions[1], net.anchors[1])], dim=1)
                    predictions = torch.concat([net.yolo(p, a) for (p, a)
                                                in zip(predictions, net.anchors)],
                                               dim=1)
                except AssertionError:
                    print(
                        "assertion error on MAP predictions calculation train set. continuing")
                    continue

                predictions = [nms(predictions[..., t]) for t in range(T)]

                for t in range(T):
                    ap_stats.update(predictions[t], bboxes[t])

                if not torch.isnan(loss):
                    stats.training.loss_sum += loss.item() * inputs.shape[0]
                stats.training.num_samples += inputs.shape[0]
                stats.training.correct_samples = ap_stats[:] * \
                    stats.training.num_samples

                processed = i * train_loader.batch_size
                total = len(train_loader.dataset)
                time_elapsed = (datetime.now() - t_st).total_seconds()
                samples_sec = time_elapsed / (i + 1) / train_loader.batch_size
                header_list = [f'Train: [{processed}/{total} '
                               f'({100.0 * processed / total:.0f}%)]']
                header_list += ['Rate: ['
                                + ', '.join([f'{c.item():.2f}'
                                             for c in counts[0]]) + ']']
                header_list += [f'Coord loss: {loss_distr[0].item()}']
                header_list += [f'Obj   loss: {loss_distr[1].item()}']
                header_list += [f'NoObj loss: {loss_distr[2].item()}']
                header_list += [f'Class loss: {loss_distr[3].item()}']
                header_list += [f'IOU   loss: {loss_distr[4].item()}']

                if i % args.track_iter == 0:
                    plt.figure()
                    for loss_idx, loss_key in enumerate(loss_order):
                        loss_tracker[loss_key].append(
                            loss_distr[loss_idx].item())
                        plt.semilogy(loss_tracker[loss_key], label=loss_key)
                        if not args.subset:
                            writer.add_scalar(f'Loss Tracker/{loss_key}',
                                              loss_distr[loss_idx].item(),
                                              len(loss_tracker[loss_key]) - 1)
                    plt.xlabel(f'iters (x {args.track_iter})')
                    plt.legend()
                    plt.savefig(f'{trained_folder}/yolo_loss_tracker.png')
                    plt.close()
                # stats.print(epoch, i, samples_sec, header=header_list)
                H = '|Trn:%.3f' % stats.training.accuracy + \
                    ' |Rate:' + m_s([c.item() for c in counts[0]])
                H = H + \
                    '(Mx: %.3f)' % stats.training.max_accuracy if stats.training.best_accuracy else H
                try:
                    H += '|Tst:%.3f' % stats.testing.accuracy
                    H += '(Max: %.3f)' % stats.testing.max_accuracy
                except:
                    pass
                pbar.set_postfix_str(header_list[1])
                pbar_epoch.set_postfix_str(H)
        # end training loop

        t_st = datetime.now()
        ap_stats = APstats(iou_threshold=0.5)

        pbar = tqdm(test_loader, desc='Test', leave=False,
                    bar_format="{desc}: {percentage:.1f}%|{bar}|{n_fmt}/{total_fmt} {postfix}")
        for i, (inputs, targets, bboxes) in enumerate(pbar):
            net.eval()

            with torch.no_grad():
                inputs = inputs.to(device)
                predictions, counts = net(inputs)
                ###################################################################################
                # if not args.train:
                #     storeData.save([predictions, bboxes], args.output_dir+'/'+args.strID+'%03d.pkl'%i)
                ###################################################################################
                T = inputs.shape[-1]
                predictions = [nms(predictions[..., t]) for t in range(T)]

                for t in range(T):
                    ap_stats.update(predictions[t], bboxes[t])
                if args.train:
                    stats.testing.loss_sum += loss.item() * inputs.shape[0]
                stats.testing.num_samples += inputs.shape[0]
                stats.testing.correct_samples = ap_stats[:] * \
                    stats.testing.num_samples

                processed = i * test_loader.batch_size
                total = len(test_loader.dataset)
                time_elapsed = (datetime.now() - t_st).total_seconds()
                samples_sec = time_elapsed / (i + 1) / test_loader.batch_size
                header_list = [f'Test: [{processed}/{total} '
                               f'({100.0 * processed / total:.0f}%)]']
                header_list += ['Event Rate: ['
                                + ', '.join([f'{c.item():.2f}'
                                             for c in counts[0]]) + ']']
                if args.train:
                    header_list += [f'Coord loss: {loss_distr[0].item()}']
                    header_list += [f'Obj   loss: {loss_distr[1].item()}']
                    header_list += [f'NoObj loss: {loss_distr[2].item()}']
                    header_list += [f'Class loss: {loss_distr[3].item()}']
                    header_list += [f'IOU   loss: {loss_distr[4].item()}']
                # stats.print(epoch, i, samples_sec, header=header_list)
                try:
                    H = '|Trn:%.3f' % stats.training.accuracy
                    H += '(Mx: %.3f)' % stats.training.max_accuracy
                except:
                    pass
                H = '|Tst:%.3f' % stats.testing.accuracy + \
                    (H if stats.training.accuracy else '') + \
                    ' |Rate:' + m_s([c.item() for c in counts[0]])
                H += '(Max: %.3f)' % stats.testing.max_accuracy if stats.testing.best_accuracy else ''
                pbar.set_postfix_str(header_list[1])
                pbar_epoch.set_postfix_str(H)

                pbar.set_postfix_str(header_list[1])
                pbar_epoch.set_postfix_str(H)
                if epoch == args.epoch-1:
                    rate_count_test.append(counts)
        if epoch == args.epoch-1:
            storeData.save(rate_count_test, args.output_dir +
                           '/'+args.strID+"_rate_count_test.pkl")

        if not args.subset and args.train:
            writer.add_scalar('Loss/train', stats.training.loss, epoch)
            writer.add_scalar('mAP@50/train', stats.training.accuracy, epoch)
            writer.add_scalar('mAP@50/test', stats.testing.accuracy, epoch)

        stats.update()

        if args.train:
            stats.plot(path=trained_folder + '/')
            b = -1
            image = Image.fromarray(np.uint8(
                inputs[b, :, :, :, 0].cpu().data.numpy(
                ).transpose([1, 2, 0]) * 255
            ))
            annotation = annotation_from_tensor(
                predictions[0][b],
                {'height': image.height, 'width': image.width},
                test_set.classes,
                confidence_th=0
            )
            # print(type(image), annotation['annotation']['object'], type(BOX_COLOR_MAP))

            marked_img = mark_bounding_boxes(
                image, annotation['annotation']['object'],
                box_color_map=BOX_COLOR_MAP, thickness=5
            )

            image = Image.fromarray(np.uint8(
                inputs[b, :, :, :, 0].cpu().data.numpy(
                ).transpose([1, 2, 0]) * 255
            ))
            annotation = annotation_from_tensor(
                bboxes[0][b],
                {'height': image.height, 'width': image.width},
                test_set.classes,
                confidence_th=0
            )
            marked_gt = mark_bounding_boxes(
                image, annotation['annotation']['object'],
                box_color_map=BOX_COLOR_MAP, thickness=5
            )

            marked_images = Image.new(
                'RGB', (marked_img.width + marked_gt.width, marked_img.height))
            marked_images.paste(marked_img, (0, 0))
            marked_images.paste(marked_gt, (marked_img.width, 0))
            if not args.subset:
                writer.add_image(
                    'Prediction', transforms.PILToTensor()(marked_images), epoch)

            if stats.testing.best_accuracy is True:
                torch.save(module.state_dict(), trained_folder + '/network.pt')
                if inputs.shape[-1] == 1:
                    marked_images.save(
                        f'{trained_folder}/prediction_{epoch}_{b}.jpg')
                else:
                    video_dims = (2 * marked_img.width, marked_img.height)
                    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                    video = cv2.VideoWriter(f'{trained_folder}/prediction_{epoch}_{b}.mp4',
                                            fourcc, 10, video_dims)
                    for t in range(inputs.shape[-1]):
                        image = Image.fromarray(
                            np.uint8(inputs[b, :, :, :, t].cpu().data.numpy().transpose([1, 2, 0]) * 255))
                        annotation = annotation_from_tensor(predictions[t][b],
                                                            {'height': image.height,
                                                             'width': image.width},
                                                            test_set.classes,
                                                            confidence_th=0)
                        marked_img = mark_bounding_boxes(image, annotation['annotation']['object'],
                                                         box_color_map=BOX_COLOR_MAP, thickness=5)
                        image = Image.fromarray(
                            np.uint8(inputs[b, :, :, :, t].cpu().data.numpy().transpose([1, 2, 0]) * 255))
                        annotation = annotation_from_tensor(bboxes[t][b],
                                                            {'height': image.height,
                                                             'width': image.width},
                                                            test_set.classes,
                                                            confidence_th=0)
                        marked_gt = mark_bounding_boxes(image, annotation['annotation']['object'],
                                                        box_color_map=BOX_COLOR_MAP, thickness=5)
                        marked_images = Image.new(
                            'RGB', (marked_img.width + marked_gt.width, marked_img.height))
                        marked_images.paste(marked_img, (0, 0))
                        marked_images.paste(marked_gt, (marked_img.width, 0))
                        video.write(cv2.cvtColor(
                            np.array(marked_images), cv2.COLOR_RGB2BGR))
                    video.release()

            stats.save(trained_folder + '/')

    if hasattr(module, 'export_hdf5') and args.train:
        module.load_state_dict(torch.load(trained_folder + '/network.pt'))
        module.export_hdf5(trained_folder + '/network.net')

    if not args.subset and args.train:
        params_dict = {}
        for key, val in args._get_kwargs():
            params_dict[key] = str(val)
        writer.add_hparams(params_dict, {'mAP@50': stats.testing.max_accuracy})
        writer.flush()
        writer.close()
