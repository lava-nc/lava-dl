import numpy as np
import torch
from tqdm import tqdm
from glob import glob
from object_detection.boundingbox.utils import storeData, accuracy, non_maximum_suppression
import argparse
import os
import sys
# sys.path.extend(path)
nms = non_maximum_suppression


# improving the module by not importing models but just the saved weight tensors

def networkPruning(saved_net_path, saved_to_net_path=None, pruning_ratio=2, pruning_method="random", log_level=0):
    # loads model from saved_net_path and saves it into saved_to_net_path after pruning it by pruningLevel using pruning_method
    # pruning_method = {"random", 'L1', 'l2'}
    # saved_to_net_path by default saves the pruned model to saved_net_path directory + '/halfed/network.pt'
    # pruning_ratio = 2 (accepts floats) [tested 2]

    # load (full) KP model network - examples below
    # saved_net_path = 'KP/Trained_DVS_FULL_KP_H1_202402201714/network.pt'

    M0 = torch.load(saved_net_path)
    MM = torch.load(saved_net_path)

    if pruning_method == 'random':
        for k, (mk, mv) in enumerate(zip(MM.keys(), MM.values())):
            S = list(mv.shape)
            S[0] = int(S[0]//pruning_ratio) if S[0] > 1 else S[0]
            MM[mk] = MM[mk][:S[0], ...] if k < len(MM)-1 else MM[mk]
            if len(mv.shape) > 1:
                S[1] = int(S[1]//pruning_ratio) if S[1] > 3 else S[1]
                MM[mk] = MM[mk][:, :S[1], ...]

    elif pruning_method in ('L1', 'L2'):
        def nrm(x): return torch.norm(
            x, 1 if pruning_method == 'L1' else 2, dim=(1, 2, 3, 4))
        index_weights = np.where(['weight_v' in name for name in MM.keys()])[0]
        pruned_index_0 = None
        for k, iw in enumerate(index_weights):
            sortArgNorm = torch.argsort(nrm(MM[[*MM.keys()][iw]]))
            pruned_index = sortArgNorm >= len(sortArgNorm)//pruning_ratio
            for kk in range(index_weights[k-1]+1 if k > 0 else 1, iw+1):
                S = list(MM[[*MM.keys()][kk]].size())
                check = (S[0] > 1)
                check = 0 if (kk == 0 or kk == len(
                    MM)-1 or 'head1_blocks.1.weight' in [*MM.keys()][kk]) else check
                if check:
                    tmp = MM[[*MM.keys()][kk]][pruned_index, ...]
                    tmp = tmp[:, pruned_index_0, ...] if kk == iw and pruned_index_0 is not None else tmp
                    MM[[*MM.keys()][kk]] = tmp
                    # print(kk, nameList[kk], eval(nameList[kk]).shape)
            pruned_index_0 = pruned_index
        MM[[*MM.keys()][-1]] = MM[[*MM.keys()][-1]][:, pruned_index_0, ...]

    if log_level:
        print(*[(f'{mk}: {list(mv0.shape)} --> {list(mv.shape)}')
              for (mk, mv, mv0) in zip(MM.keys(), MM.values(), M0.values())], sep='\n')

    saved_to_net_path = '/'.join(saved_net_path.split('/')[:-1]) + '/%s_pruning_by%.1f/network.pt' % (
        pruning_method, pruning_ratio) if not saved_to_net_path else saved_to_net_path
    print(f'saving to {saved_to_net_path}')
    os.makedirs(os.path.dirname(saved_to_net_path), exist_ok=True)
    torch.save(MM, saved_to_net_path)

    # Check if loading works
    from models.sdnn_short_single_head_KP import Network as Network_short
    device = torch.device('cuda:{}'.format(1))
    net_short = Network_short(threshold=0.1,
                              tau_grad=0.1,
                              scale_grad=0.2,
                              num_classes=80,
                              clamp_max=5).to(device)
    module_short = net_short
    module_short.load_model(saved_to_net_path)
    from torchinfo import summary
    print(summary(module_short, (3, 448, 448))) if log_level > 1 else None
    print('succesfull pruning!')


def networkPruning_(saved_net_path, saved_to_net_path=None, pruning_ratio=2, pruning_method="random", log_level=0):
    # loads model from saved_net_path and saves it into saved_to_net_path after pruning it by pruningLevel using pruning_method
    # pruning_method = {"random", 'L1', 'l2'}
    # saved_to_net_path by default saves the pruned model to saved_net_path directory + '/halfed/network.pt'
    # pruning_ratio = 2 (accepts floats) [tested 2]

    # load (full) KP model network - examples below
    # saved_net_path = 'single_head_KP/Trained_DVS_FULL_KP_H1_202402201716/network.pt'
    # saved_net_path = 'single_head_KP/Trained_DVS_FULL_KP_H1_202402201714/network.pt'
    from torchinfo import summary
    from models.sdnn_single_head_KP import Network

    device = torch.device('cuda:{}'.format(1))
    net = Network(threshold=0.1,
                  tau_grad=0.1,
                  scale_grad=0.2,
                  num_classes=80,
                  clamp_max=5).to(device)
    module = net
    module.load_model(saved_net_path)

    print(summary(module, (3, 448, 448))) if log_level > 1 else None
    # pruning_method = 'halfed' if pruning_method=='random' else pruning_method

    # prunes to model and saves it
    import pickle
    module_short = pickle.loads(pickle.dumps(module))
    nameList = []

    for k, param in enumerate(module_short.state_dict()):
        S = list(module_short.state_dict()[param].size())
        check = (S[0] > 1)
        check = 0 if (k == 0 or k == len(module_short.state_dict()) -
                      1 or param == 'head1_blocks.1.weight') else check
        name_blk = param.split('.')
        name = 'module_short.%s[%s].' % (
            tuple(name_blk[:2])) + ".".join(name_blk[2::]) + ".data" if k > 0 else ''
        nameList.append(name)
        if pruning_method == 'random':
            if k == 0:
                continue
            if check:
                S[0] = S[0]//pruning_ratio
                tmp = eval(name)[0:S[0], ...]
                exec('%s = tmp' % name)
            if len(S) > 1 or (k == len(module_short.state_dict())-1):
                S[1] = S[1]//pruning_ratio if (S[1] %
                                               pruning_ratio) == 0 else S[1]
                tmp = eval(name)[:, 0:S[1], ...]
                exec('%s = tmp' % name)

    if pruning_method in ('L1', 'L2'):
        def nrm(x): return torch.norm(
            x, 1 if pruning_method == 'L1' else 2, dim=(1, 2, 3, 4))
        index_weights = np.where(['weight_v' in name for name in nameList])[0]
        pruned_index_0 = None
        for k, iw in enumerate(index_weights):
            sortArgNorm = torch.argsort(nrm(eval(nameList[iw])))
            pruned_index = sortArgNorm >= len(sortArgNorm)//pruning_ratio
            for kk in range(index_weights[k-1]+1 if k > 0 else 1, iw+1):
                S = list(eval(nameList[kk]).size())
                check = (S[0] > 1)
                check = 0 if (kk == 0 or kk == len(
                    nameList)-1 or 'head1_blocks.1.weight' in nameList[kk]) else check
                if check:
                    tmp = eval(nameList[kk])[pruned_index, ...]
                    tmp = tmp[:, pruned_index_0, ...] if kk == iw and pruned_index_0 is not None else tmp
                    exec('%s = tmp' % nameList[kk])
                    # print(kk, nameList[kk], eval(nameList[kk]).shape)
            pruned_index_0 = pruned_index
        tmp = eval(nameList[-1])[:, pruned_index_0, ...]
        exec('%s = tmp' % nameList[-1])
        # print(len(nameList)-1, nameList[-1], eval(nameList[-1]).shape)

    if log_level:
        for k, param in enumerate(module_short.state_dict()):
            S = list(module_short.state_dict()[param].size())
            S0 = list(module.state_dict()[param].size())
            print(f'{k}: {param}: ', S0, '-->', S)

    saved_to_net_path = '/'.join(saved_net_path.split('/')[:-1]) + '/%s_pruning_by%.1f/network.pt' % (
        pruning_method, pruning_ratio) if not saved_to_net_path else saved_to_net_path
    print(f'saving to {saved_to_net_path}')
    os.makedirs(os.path.dirname(saved_to_net_path), exist_ok=True)
    torch.save(module_short.state_dict(), saved_to_net_path)

    # Check if loading works
    from models.sdnn_short_single_head_KP import Network as Network_short
    device = torch.device('cuda:{}'.format(1))
    net_short = Network_short(threshold=0.1,
                              tau_grad=0.1,
                              scale_grad=0.2,
                              num_classes=80,
                              clamp_max=5).to(device)
    module_short = net_short
    module_short.load_model(saved_to_net_path)
    print(summary(module_short, (3, 448, 448))) if log_level > 1 else None
    print('succesfull pruning!')


def toPanda(arg_list, update_flag=True, export_to_pkl=True, export_to_excel=False):
    import pandas as pd
    try:
        Collection = storeData.load('trained_models_collection.pkl')
    except:
        Collection = pd.DataFrame()

    a_ = []
    for aa in arg_list:
        try:
            _a_ = eval(aa.split(' ')[-1][:-1])
            a_.append(_a_[0] if type(_a_) == list else _a_)
        except:
            a_.append(aa.split(' ')[-1][:-1])

    c_ = pd.DataFrame(a_).T
    c_.columns = [aa.split(' ')[0] for aa in arg_list]
    indx = [aa.split(' ')[-1][-13:-1]
            for aa in arg_list if aa.split(' ')[0] == 'output_path']
    c_.index = indx

    if not (indx in list(Collection.index)) or update_flag:
        Collection = Collection.drop(index=indx) if indx in list(
            Collection.index) else Collection
        Collection = pd.concat([Collection, c_])
        storeData.save(
            Collection, 'trained_models_collection.pkl') if export_to_pkl else None
        Collection.to_excel(
            "trained_models_collection.xlsx") if export_to_excel else None
    return Collection


def retrieve_path(seed_str, verbose=False, search_pos=-1, remove_paths=False, confirm_deletion=False):
    # search_pos = [1,0,-1] - beginning, middle, end of string
    # ORIGIN_PATH = "/home/dbendaya/work/ContinualLearning/tinyYolov3_lava/YOLOsdnn/tinyYOLOv3s/"
    # SEARCH_STR = ORIGIN_PATH + '**/' + ('%s*/**/' if search_pos==1 else '') + ('*%s*/**/' if search_pos==0 else '') + ('*%s' if search_pos==-1 else '')
    SEARCH_STR = '**/' + ('%s*/**/' if search_pos == 1 else '') + (
        '*%s*/**/' if search_pos == 0 else '') + ('*%s' if search_pos == -1 else '')
    # SEARCH_STR = "/home/dbendaya/work/ContinualLearning/**/Trained_*%s"
    G_extras = glob(SEARCH_STR % seed_str, recursive=True)
    G_extras.sort(key=os.path.getmtime)
    print('ATTENTION: more than 1 entry for final search string pattern: %s' %
          seed_str) if len(G_extras) > 2 else None
    if G_extras == []:
        print(
            'FAILED: Trained folder not found or wrong search string') if verbose else None
        return []
    print(f'\nPattern {seed_str} can be found here:',
          *G_extras, sep='\n') if verbose else None
    choice = 'y' if confirm_deletion and remove_paths else ''
    if remove_paths:
        yes = {'yes', 'y', 'Y'}
        if not confirm_deletion:
            choice = input('Removing paths [y/N]?').lower()
        if choice in yes:
            o = 1
            while o:
                for g in G_extras:
                    o = os.system("rm %s/ -fr 2>/dev/null" % g)
    return G_extras


def extractValue(str_srch, RL):
    RL = [RL] if type(RL) != list else RL
    for rl in RL:
        if str_srch in rl:
            return rl.split(':')[1][1:-1]
    return None


def extractPath(st):
    for s in {'"', '[', ']', "'"}:
        st = st.replace(s, '')
    return st


def retrieve_info(seedStr, verbose=False, track_loaded_models=False, confirm_deletion=False):
    path = retrieve_path(seedStr, verbose)
    if not path:
        return None
    path = [p for p in path if 'Trained_' in p]
    if not path:
        return None
    else:
        path = path[0]
    model = ''
    paramsAT = "%s/args.txt" % path
    dataAT = "%s/AP@0.5.txt" % path
    file_opened = 0
    try:
        with open(paramsAT) as f:
            RLargs = f.readlines()
            for rl in RLargs:
                if 'model' in rl:
                    model = rl.split(':')[1][1:-1].split('[')[-1].split(']')[0]
                elif 'lr ' in rl:
                    lr = rl.split(':')[1][1:-1]
                elif 'aug_prob' in rl:
                    aug_prob = rl.split(':')[1][1:-1]
                elif 'path' in rl:
                    dataset = rl.split(':')[1][1:-1]
                file_opened = 1
        with open(dataAT) as f:
            RL = f.readlines()
    except Exception as error:
        print(f'*** Error: {type(error).__name__}\n*** {path}: %s file is missing...' %
              ('data' if file_opened else 'args'))
        retrieve_path(seedStr, remove_paths=True,
                      confirm_deletion=confirm_deletion)
        print('=============>|Terminating')
        return None

    nums = {'train': [], 'test': []}
    for k, rl in enumerate(RL[1:]):
        rl_ = rl.split(' ')
        cc = 0
        for nn in rl_:
            try:
                if cc:
                    nums['test'].append(float(nn))
                else:
                    nums['train'].append(float(nn))
                cc += 1
            except ValueError:
                pass
    RLargs.append(f'Epochs : {len(RL)-1}\n')
    RLargs.append(f'train_accuracy : {100*max(nums["train"])}\n')
    RLargs.append(f'test_accuracy : {100*max(nums["test"])}\n')
    RLargs.append(f'output_path : {path}\n')
    if verbose:
        args2print = {'DVSlike', 'Heads', 'b', 'load',
                      'model', 'lr', 'aug_prob', 'sparsity', 'threshold'}
        Collection = toPanda(RLargs)
        print(f'- Model %s on %s\nMax Accuracy after {len(RL)-1}/{extractValue("epoch", RLargs)} training epochs: train: %.2f | test: %.2f' % (
            path.split('/')[-1], dataset, 100*max(nums["train"]), 100*max(nums["test"])))
        print('**** Further info ****')
        print('- Full network path: %s' %
              '/'.join(dataAT.split('/')[:-1])+'/network.pt')
        print('- Args: ', *[a[:-1] for a in RLargs if a.split(' : ')[0] in args2print],
              sep=' | ')
        if track_loaded_models:
            loaded_models = []
            loaded_models.append(os.path.dirname(
                extractPath(extractValue('load', RLargs))))
            accuracy = []
            while loaded_models[-1] is not None:
                search_str = os.path.basename(loaded_models[-1])[-12:]
                # print('***->',search_str)
                RLargs_ = retrieve_info(search_str)
                if RLargs_ != None:
                    RLargs_ = RLargs_[2]
                    loaded_models.append(os.path.dirname(
                        extractPath(extractValue('load', RLargs_))))
                    accuracy.append(f"on {extractValue('dataset', RLargs_)}(dvs={extractValue('DVSlike', RLargs_)},lr={extractValue('lr', RLargs_)},aug={extractValue('aug_prob', RLargs_)})-acc:{'%.2f'%float(extractValue('train_', RLargs_))}/{'%.2f'%float(extractValue('test_', RLargs_))} @ epoch:{extractValue('Epochs', RLargs_)}/{extractValue('epoch', RLargs_)}")
                    # print(search_str, accuracy[-1])
                else:
                    accuracy.append('N/A')
                    break
            # accuracy, loaded_models = accuracy, loaded_models[:-1] if RLargs_ != None else loaded_models
            print('- Tracking previous loaded models:')
            print(*[(l+': '+a) for l, a in zip(loaded_models[::-1],
                  accuracy[::-1])], sep=' \-->\n')

    return '/'.join(dataAT.split('/')[:-1])+'/network.pt', model, RLargs, (Collection if verbose else None)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-output_dir', type=str,   default="Combine_heads_KP",
                        help="folder in which we store predictions+bboxes")
    parser.add_argument('-models', type=str,
                        default='Full', help='[Full, Half]')
    parser.add_argument('-gpu', type=int, default='0', help='gpu #')
    parser.add_argument('-b', type=int, default='1', help='batch #')
    parser.add_argument('-num_workers', type=int, default='1', help='batch #')
    parser.add_argument('-search_str', type=str, default=[],
                        help='search strings based on last patterns on the run', nargs='+')
    parser.add_argument('-inference', default=False,
                        action='store_true', help='run inference')
    parser.add_argument('-y', default=False,
                        action='store_true', help='confirm deletions')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    n_heads = len(args.search_str)
    if n_heads > 1:
        network_path, model = [], []
        for k, srchs in enumerate(args.search_str):
            network_path_, model_, RLargs, _ = retrieve_info(srchs)
            network_path.append(network_path_)
            model.append(model_)
            # args.models = 'Half' if ('short' in model[k] or 'half' in model[k]) else 'Full'
            args.dataset = extractValue('dataset', RLargs)

    if args.inference:
        print('running inference commands:')
        run_inference = f'python train_sdnn_base.py -b {args.b} -dataset {args.dataset} -gpu {args.gpu} -num_workers {args.num_workers} -output_dir %s -strID head%d_ -model %s -load %s'

        [print(run_inference % (args.output_dir+'/'+'_'.join(network_path[k].split('/')[-2].split('_')[1:]),
                                k, model[k], network_path[k])) for k in range(n_heads)]
        [os.system(run_inference % (args.output_dir+'/'+'_'.join(network_path[k].split('/')[-2].split('_')[1:]),
                                    k, model[k], network_path[k])) for k in range(n_heads)]

    # loading head lists from dir
    G = [glob(f'{args.output_dir}/%s/head{k}_*' %
              '_'.join(network_path[k].split('/')[-2].split('_')[1:])) for k in range(n_heads)]
    for k in range(n_heads):
        G[k].sort()

    acc = []
    BB = []
    pbar = tqdm(zip(*G), desc='loading heads predictions')
    predTot = []
    for G0 in pbar:
        pred_ = []
        for g in G0:
            pred0, bb = storeData.load(g)
            pred_.append(pred0.to(torch.device('cpu')).half())
        pred = torch.cat(pred_, dim=1)
        predTot.append(pred)
        BB.append(bb)

    def combined_accuracy(conf_threshold):
        acc = []
        conf_threshold = torch.tensor(conf_threshold)
        for pred, bb in zip(predTot, BB):
            T = pred.shape[-1]
            det = [nms(pred[..., t], conf_threshold=conf_threshold)
                   for t in range(T)]
            acc.append(accuracy(det, bb))
        return np.array(acc).mean()

    initial_guess = 0.3
    bnds = [0.2, 0.55]
    RES = []
    print('Scan results as fn of confidence threshold...')
    while 1:
        conf_th = np.asarray(bnds[0])
        RES.append([combined_accuracy(conf_th), conf_th])
        print(f'{len(RES)}: Accuracy:{RES[-1][0]}mAP @ threshold={conf_th}')
        bnds[0] += 0.05
        if len(RES) > 1:
            if RES[-1][0]-RES[-2][0] < 0:
                print(
                    f'Exiting\nBest Accuracy:{RES[-2+(bnds[0]>bnds[1])][0]}mAP @ threshold={RES[-2+(bnds[0]>bnds[1])][1]}')
                break

    # routines checking for all possible combinations of heads
    if n_heads < 3:  # no need to test combinations
        os.exit()

    import itertools
    Gperm_ind = [comb for comb in itertools.combinations(range(0, n_heads), 3)]
    Gperm_ind += [comb for comb in itertools.combinations(
        range(0, n_heads), 2)]
    for kG in Gperm_ind:
        GG = [G[gi] for gi in kG]
        pbar = tqdm(zip(*GG), desc='loading heads predictions')
        predTot = []
        for G0 in pbar:
            pred_ = []
            for g in G0:
                pred0, bb = storeData.load(g)
                pred_.append(pred0.to(torch.device('cpu')).half())
            pred = torch.cat(pred_, dim=1)
            predTot.append(pred)
            BB.append(bb)

        def combined_accuracy(conf_threshold):
            acc = []
            conf_threshold = torch.tensor(conf_threshold)
            for pred, bb in zip(predTot, BB):
                T = pred.shape[-1]
                det = [nms(pred[..., t], conf_threshold=conf_threshold)
                       for t in range(T)]
                acc.append(accuracy(det, bb))
            return np.array(acc).mean()

        bnds = [0.2, 0.55]
        RES = []
        print(f'Perm={kG}:\nScan results as fn of confidence threshold...')
        while 1:
            conf_th = np.asarray(bnds[0])
            RES.append([combined_accuracy(conf_th), conf_th])
            print(
                f'{len(RES)}: Accuracy:{RES[-1][0]}mAP @ threshold={conf_th}')
            bnds[0] += 0.05
            if len(RES) > 1:
                if RES[-1][0]-RES[-2][0] < 0:
                    print(
                        f'Exiting\nBest Accuracy:{RES[-2+(bnds[0]>bnds[1])][0]}mAP @ threshold={RES[-2+(bnds[0]>bnds[1])][1]}')
                    break
