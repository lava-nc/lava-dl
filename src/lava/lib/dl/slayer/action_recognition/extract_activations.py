import argparse
parser = argparse.ArgumentParser()

# Dataset
parser.add_argument('--dataset', type=str, default="NTU", help='NTU or HARDVS. Default: NTU')
parser.add_argument('--data-root', type=str, help='Path to root folder of the data.')
args = parser.parse_args()

from model import model_registry 
import os
from torchvision import transforms
import torch
from torch import nn
from torch import optim
import time
import numpy as np
from dataloaders import dataset_registry 
from metavision_core.event_io.py_reader import EventDatReader
from metavision_ml.preprocessing import histo_quantized

# Params
model_cls = model_registry["efficientnet-b0-S4D"] 
model_params = {"lstm_num_hidden": None,
                "num_readout_hidden": 256,
                "readout_bias": False,
                "s4d_num_hidden": 64,
                "s4d_states": 1280,
                "s4d_is_real": True,
                "s4d_lr": 0.0,
                }

resolution = 224


num_classes = 1
model = model_cls(num_classes=num_classes, **model_params).cuda()
model = model.efficientnet
model.eval()

annotation_file_train = os.path.join(args.data_root, 'train.txt')
annotation_file_val = os.path.join(args.data_root, 'val.txt')
annotation_file_test = os.path.join(args.data_root, 'test.txt')

preprocess = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize(resolution + 2, antialias=None),  # image batch, resize smaller edge to 256
    transforms.CenterCrop(resolution),  # image batch, center crop to square 224x224
    #transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


for annotation_file in [annotation_file_train, annotation_file_val, annotation_file_test]:
    binning_dt = 10**6 / 150
    fns = []
    with open(annotation_file, "r") as f:
        for entry in f:
            fn, label = entry.split(" ")
            fns.append(fn)

    for fn in fns:
        print(fn)
        record_dat = EventDatReader(fn)
        if record_dat.event_count() == 0:
            print("EMPTY", fn)
        height, width = record_dat.get_size()
        tbins=1
        img = np.zeros([3, width, height])
        imgs = [img.T]

        while record_dat.current_time / 10**6 < record_dat.duration_s:
            events = record_dat.load_delta_t(binning_dt)
            volume = np.zeros((1, 2, height, width), dtype=np.uint8)
            histo_quantized(events, volume, binning_dt)
            img[:] += volume[0, 0].astype(bool).T + volume[0, 1].astype(bool).T
            imgs.append(img.copy().T)
            img *= 0

        imgs = [preprocess(img) for img in imgs]
        out_buffer = torch.empty((len(imgs), 1280))
        batch_size = 64 
        num_batches = int(np.ceil(len(imgs) / batch_size))
        for batch_idx in range(num_batches):
            batch = torch.stack(imgs[batch_idx*batch_size:(batch_idx+1)*batch_size]).float().cuda()
            outputs = model(batch).detach().cpu()
            out_buffer[batch_idx*batch_size:(batch_idx+1)*batch_size] = outputs
        
        fn_save = fn.replace("ssd1", "ssd2") 
        if "data_120/" in fn:
            fn_save = fn_save.replace("data_120/", "data_eff_features_120/") 
        else:
            fn_save = fn_save.replace("data/", "data_eff_features/") 
        print(fn_save, out_buffer.shape)
        np.save(fn_save, out_buffer.numpy())






#def extract_forward(self, x):
#
#        inp_shape = x.shape
#
#        # Move batch into images dimension for efficientnet
#        if len(inp_shape) == 5:
#            x = x.reshape(inp_shape[0] * inp_shape[1], *inp_shape[2:])
#        else:
#            x = x.squeeze(0)
#
#        # Pass input through EfficientNet
#        x = self.efficientnet(x)
#
#        if len(inp_shape) == 5:
#            x = x.reshape(inp_shape[0], inp_shape[1], *x.shape[1:]) # Get to dimension (B, T, C) 
#        else:
#            raise NotImplementedError("Not implement for unbatched data")
#            x = x.reshape(inp_shape[0], -1)
#
#        eff_activations = x.clone()
#
#        # Pass output through S4D layer
#        x = self.s4d(x)
#
#        s4d_activations = x.clone()
#
#        # Take last step output from S4D and pass through readout layer
#        # x = x[:, -1, :]
#        x = self.readout(x)
#
#        return x, eff_activations, s4d_activations
#
#model_cls.forward = extract_forward
#
#checkpoint = torch.load(f"{args.model}-{args.dataset}.pth")
#model.load_state_dict(checkpoint)
#model.eval()
#
#print("start testing")
#with torch.no_grad():
#    test_preds = []
#    test_tgts = []
#
#    for batch_idx, (inputs, targets) in enumerate(test_dataloader):
#        # Forward pass
#        outputs, eff_act, s4d_act = model(inputs)
#
#        pred = torch.argmax(outputs[:, -1], dim=1).detach().cpu().numpy().tolist()
#        test_preds += pred
#        tgt = targets.cpu().numpy().tolist()
#        test_tgts += tgt
#        if batch_idx % print_interval == 0:
#            print(f'Batch [{batch_idx+1}/{len(test_dataloader)}]')
#            print(f'pred {pred}, targets {tgt} acc {accuracy_score(test_tgts, test_preds)}')
#        break
#
#
#    cm = ConfusionMatrixDisplay.from_predictions(test_tgts, test_preds)
#    cm_norm = ConfusionMatrixDisplay.from_predictions(test_tgts, test_preds, normalize='true')
#    test_acc = cm.confusion_matrix.diagonal() / cm.confusion_matrix.sum(axis=1)
#    print(f'Test Acc {test_acc}, mean {np.mean(test_acc)}')
#    print(cm.confusion_matrix)
#    print(cm_norm.confusion_matrix)
#
#
#
#np.save("activations/class_act.dat", outputs.numpy())
#np.save("activations/eff_act.dat", eff_act.numpy())
#np.save("activations/s4d_act.dat", s4d_act.numpy())
#
#model.s4d.setup_step()
#A = model.s4d.layer.kernel.dA.detach()
#B = model.s4d.layer.kernel.dB.detach()
#C = model.s4d.layer.kernel.dC.detach()
#
#np.save("activations/s4d_A.dat", A)
#np.save("activations/s4d_B.dat", B)
#np.save("activations/s4d_C.dat", C) 
#
#class_state_dict = model.readout.state_dict()
#torch.save(class_state_dict, "activations/classifier_params.pt")
#
#s4d_state_dict = model.s4d.state_dict()
#torch.save(s4d_state_dict, "activations/s4d_params.pt")
#
#np.save("activations/ground_truth.dat", tgt)
#np.save("activations/predictions.dat", pred)