import argparse
import os
import os.path as osp
from pathlib import Path
import numpy as np
import torch
import time
import utils
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import transforms
import network, loss, IID_losses
from torch.utils.data import DataLoader
from torch.utils.checkpoint import checkpoint
from data_list import ImageList_idx
import random
from scipy.spatial.distance import cdist
from numpy import linalg as LA

def op_copy(optimizer):
    for param_group in optimizer.param_groups:
        param_group['lr0'] = param_group['lr']
    return optimizer

def lr_scheduler(optimizer, iter_num, max_iter, gamma=10, power=0.75):
    decay = (1 + gamma * iter_num / max_iter) ** (-power)
    for param_group in optimizer.param_groups:
        param_group['lr'] = param_group['lr0'] * decay
        param_group['weight_decay'] = 1e-3
        param_group['momentum'] = 0.9
        param_group['nesterov'] = True
    return optimizer

def image_train(resize_size=256, crop_size=224, alexnet=False):
  if not alexnet:
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                   std=[0.229, 0.224, 0.225])
  else:
    normalize = Normalize(meanfile='./ilsvrc_2012_mean.npy')
  return  transforms.Compose([
        transforms.Resize((resize_size, resize_size)),
        transforms.RandomCrop(crop_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize
    ])

def image_test(resize_size=256, crop_size=224, alexnet=False):
  if not alexnet:
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                   std=[0.229, 0.224, 0.225])
  else:
    normalize = Normalize(meanfile='./ilsvrc_2012_mean.npy')
  return  transforms.Compose([
        transforms.Resize((resize_size, resize_size)),
        transforms.CenterCrop(crop_size),
        transforms.ToTensor(),
        normalize
    ])

def data_load(args): 
    ## prepare data
    dsets = {}
    dset_loaders = {}
    train_bs = args.batch_size
    txt_tar = open(args.t_dset_path).readlines()
    txt_test = open(args.test_dset_path).readlines()

    loader_kwargs = dict(num_workers=args.worker, drop_last=False, pin_memory=True)
    if args.worker > 0:
        loader_kwargs['prefetch_factor'] = args.prefetch_factor
    dsets["target"] = ImageList_idx(txt_tar, transform=image_train())
    dset_loaders["target"] = DataLoader(dsets["target"], batch_size=train_bs, shuffle=True, **loader_kwargs)
    dsets['target_'] = ImageList_idx(txt_tar, transform=image_train())
    dset_loaders['target_'] = DataLoader(dsets['target_'], batch_size=train_bs * 3, shuffle=False, **loader_kwargs)
    dsets["test"] = ImageList_idx(txt_test, transform=image_test())
    dset_loaders["test"] = DataLoader(dsets["test"], batch_size=train_bs * 3, shuffle=False, **loader_kwargs)

    return dset_loaders

def weights_torch_load(path):
    """Load source weights on CPU before strict CUDA loading."""
    try:
        return torch.load(path, map_location='cpu', weights_only=True)
    except TypeError:  # PyTorch 1.x / 2.0 compatibility.
        return torch.load(path, map_location='cpu')


def build_backbone_without_download(name, kind):
    """Construct the checkpoint architecture without an unnecessary download."""
    registry = network.res_dict if kind == 'res' else network.vgg_dict
    original_factory = registry[name]

    def no_pretrained(*args, **kwargs):
        kwargs['pretrained'] = False
        return original_factory(*args, **kwargs)

    registry[name] = no_pretrained
    try:
        return network.ResBase(res_name=name) if kind == 'res' else network.VGGBase(vgg_name=name)
    finally:
        registry[name] = original_factory


def backbone_forward(net, inputs, args):
    """Optionally recompute ResNet activations during backward on 24 GB GPUs."""
    if not args.activation_checkpoint or not net.training:
        return net(inputs)
    # Old checkpoint implementations require at least one differentiable input.
    if not inputs.requires_grad:
        inputs = inputs.requires_grad_(True)
    try:
        return checkpoint(net, inputs, use_reentrant=False)
    except TypeError:  # Compatibility with older PyTorch releases.
        return checkpoint(net, inputs)


@torch.no_grad()
def obtain_domain_prior_tiled(all_fea, all_output, args):
    """Compute the original source prior with exact tiled top-k search.

    The original utility materializes an N x N similarity matrix.  This
    preserves its two centroid refinements, self-inclusive top-20 neighbours,
    and alpha formula while bounding temporary memory to one query tile.
    """
    features = torch.from_numpy(np.asarray(all_fea, dtype=np.float32))
    probabilities = torch.softmax(all_output.float().cpu(), dim=1)
    predicted = probabilities.argmax(dim=1).numpy()
    class_num = probabilities.shape[1]
    feature_np = features.numpy()
    affinity = probabilities.numpy()
    centers = affinity.T.dot(feature_np)
    centers /= 1e-8 + affinity.sum(axis=0)[:, None]
    labelset = np.flatnonzero(np.bincount(predicted, minlength=class_num) > args.threshold)
    if len(labelset) == 0:
        raise RuntimeError('No active pseudo-label class is available for source-prior estimation.')
    for _ in range(2):
        distances = cdist(feature_np, centers[labelset], args.distance)
        predicted = labelset[distances.argmin(axis=1)]
        centers = np.eye(class_num, dtype=np.float32)[predicted].T.dot(feature_np)
        centers /= 1e-8 + np.bincount(predicted, minlength=class_num)[:, None]

    search_device = torch.device('cuda' if args.search_device == 'auto' else args.search_device)
    features = F.normalize(features, dim=1).to(search_device)
    labels = torch.from_numpy(predicted).to(search_device)
    count = len(features)
    k = min(20, count)
    if k < 2:
        raise RuntimeError('Source-prior estimation requires at least two target samples.')
    similarity_sum = torch.zeros((), device=search_device, dtype=torch.float64)
    entropy_sum = torch.zeros((), device=search_device, dtype=torch.float64)
    for start in range(0, count, args.prior_query_chunk):
        query = features[start:start + args.prior_query_chunk]
        values, indices = (query @ features.T).topk(k, dim=1)
        similarity_sum += values[:, 1:].double().sum()
        neighbour_labels = labels[indices]
        neighbour_counts = torch.zeros(len(query), class_num, device=search_device, dtype=torch.float64)
        neighbour_counts.scatter_add_(1, neighbour_labels, torch.ones_like(neighbour_labels, dtype=torch.float64))
        neighbour_probabilities = neighbour_counts / k
        # log(clamp(p, eps)) is zero for p=1.  The original log(p + eps)
        # makes a pure neighbourhood have a small *negative* entropy.
        entropy_sum += (-(neighbour_probabilities * neighbour_probabilities.clamp_min(args.epsilon).log()).sum(dim=1)).sum()
    mean_similarity = similarity_sum / (count * (k - 1))
    mean_entropy = entropy_sum / count
    if (not torch.isfinite(mean_similarity) or not torch.isfinite(mean_entropy)
            or mean_similarity <= 0 or mean_entropy <= args.epsilon):
        # A collapsed source has no usable class-neighbour uncertainty.  It
        # contributes zero instead of making all normalized source weights NaN.
        return torch.zeros((), dtype=torch.float32)
    alpha = mean_similarity / mean_entropy
    if not torch.isfinite(alpha) or alpha <= 0:
        return torch.zeros((), dtype=torch.float32)
    return alpha.cpu()


def train_target(args):
    if not torch.cuda.is_available():
        raise RuntimeError('Main_Final_v1.py requires a CUDA GPU.')
    dset_loaders = data_load(args)
    ## set base network
    if args.net[0:3] == 'res':
        netF_list = [build_backbone_without_download(args.net, 'res').cuda() for i in range(len(args.src))]
    elif args.net[0:3] == 'vgg':
        netF_list = [build_backbone_without_download(args.net, 'vgg').cuda() for i in range(len(args.src))]

    netB_list = [network.feat_bottleneck(type=args.classifier, feature_dim=netF_list[i].in_features, bottleneck_dim=args.bottleneck).cuda() for i in range(len(args.src))] 
    netC_list = [network.feat_classifier(type=args.layer, class_num = args.class_num, bottleneck_dim=args.bottleneck).cuda() for i in range(len(args.src))]
    netDC_inter = network.domain_classifier(domain_num = len(args.src), bottleneck_dim=args.bottleneck).cuda()
    #netDC_inter_adv = network.domain_classifier_adv(domain_num = len(args.src), bottleneck_dim=args.bottleneck).cuda()


    param_group = []

    for i in range(len(args.src)):
        modelpath = args.output_dir_src[i] + '/source_F.pt'
        print(modelpath)
        netF_list[i].load_state_dict(weights_torch_load(modelpath), strict=True)
        netF_list[i].eval()
        for k, v in netF_list[i].named_parameters():
            param_group += [{'params':v, 'lr':args.lr * args.lr_decay1}]

        modelpath = args.output_dir_src[i] + '/source_B.pt'
        print(modelpath)
        netB_list[i].load_state_dict(weights_torch_load(modelpath), strict=True)
        netB_list[i].eval()
        for k, v in netB_list[i].named_parameters():
            param_group += [{'params':v, 'lr':args.lr * args.lr_decay2}]

        modelpath = args.output_dir_src[i] + '/source_C.pt'
        print(modelpath)
        netC_list[i].load_state_dict(weights_torch_load(modelpath), strict=True)
        netC_list[i].eval()
        for parameter in netC_list[i].parameters():
            parameter.requires_grad_(False)
    for k, v in netDC_inter.named_parameters():
        param_group += [{'params':v, 'lr':args.lr}]
    optimizer = optim.SGD(param_group)
    optimizer = op_copy(optimizer)

    max_iter = args.max_epoch * len(dset_loaders["target"])
    interval_iter = max_iter // args.interval
    if interval_iter < 1:
        raise ValueError('--interval is larger than the total number of target steps.')
    C = interval_iter
    iter_num = 0
    acc_init = 0
    iter_num_update = 0
    iter_test = iter(dset_loaders["target"])

    args.w = torch.ones(len(args.src), device='cuda')

    while iter_num < max_iter:
        try:
            inputs_test, _, tar_idx = next(iter_test)
        except StopIteration:
            iter_test = iter(dset_loaders["target"])
            inputs_test, _, tar_idx = next(iter_test)

        if inputs_test.size(0) == 1:
            continue
        if iter_num % C == 0:
            iter_num_update += 1
            initc = []
            all_feas = []
            for i in range(len(args.src)):
                netF_list[i].eval()
                netB_list[i].eval()
                #netC_list[i].eval()
                if iter_num == 0:
                    temp1, temp2, alpha = obtain_label_alpha(dset_loaders['target_'], netF_list[i], netB_list[i], netC_list[i], args, obtain_prior=True)
                    args.w[i] = alpha.to(args.w.device)
                else:
                    temp1, temp2 = obtain_label_alpha(dset_loaders['target_'], netF_list[i], netB_list[i], netC_list[i], args, obtain_prior=False)
                temp1 = torch.from_numpy(temp1).cuda()#center points
                temp2 = torch.from_numpy(temp2).cuda()#features
                initc.append(temp1)#
                all_feas.append(temp2)

            _, feas_all, label_confi, _, _ = obtain_label_ts(dset_loaders['test'], netF_list, netB_list,netC_list, netDC_inter, args, iter_num_update)

            for i in range(len(args.src)):
                netF_list[i].train()
                netB_list[i].train()
                #netC_list[i].train()
            netDC_inter.train()
            if iter_num == 0:
                prior_sum = args.w.sum()
                if not torch.isfinite(prior_sum) or prior_sum <= 0:
                    raise RuntimeError('All source priors are invalid; target adaptation cannot start.')
                args.w = args.w / prior_sum
                w = args.w
                print('Initialized weights:', args.w)
            else:
                args.w = w
        inputs_test = inputs_test.cuda()
        outputs_all = torch.zeros(len(args.src), inputs_test.shape[0], args.class_num, device=inputs_test.device)
        outputs_all_N = torch.zeros(len(args.src), inputs_test.shape[0], args.class_num, device=inputs_test.device)
        weights_all = torch.ones(inputs_test.shape[0], len(args.src), device=inputs_test.device)

        features_all = torch.zeros(len(args.src), inputs_test.shape[0], args.bottleneck, device=inputs_test.device)
        features_all_w = torch.zeros(inputs_test.shape[0], args.bottleneck, device=inputs_test.device)

        features_all_F = torch.zeros(len(args.src), inputs_test.shape[0], netF_list[0].in_features, device=inputs_test.device)
        features_all_F_w = torch.zeros(inputs_test.shape[0], netF_list[0].in_features, device=inputs_test.device)


        features_all_N = torch.zeros(len(args.src), inputs_test.shape[0], args.bottleneck, device=inputs_test.device)

        features_all_F_N = torch.zeros(inputs_test.shape[0], args.bottleneck, device=inputs_test.device)

        outputs_all_w = torch.zeros(inputs_test.shape[0], args.class_num, device=inputs_test.device)
        outputs_all_w_N = torch.zeros(inputs_test.shape[0], args.class_num, device=inputs_test.device)

        src_domain_labels = torch.zeros(inputs_test.shape[0]*len(args.src), 1, device=inputs_test.device)

        start_test = True
        start_each_domain = True
        for i in range(len(args.src)):
            features_F = backbone_forward(netF_list[i], inputs_test, args)
            features_test = netB_list[i](features_F)

            outputs_test = netC_list[i](features_test)

            features_all_F[i] = features_F
            features_all[i] = features_test

            weights_numerator = netDC_inter(features_test)
            weights_test = weights_numerator
            softmax_weights = nn.Softmax(dim=1)(weights_test)

            domain_weight = softmax_weights[:, i]
            domain_weight = domain_weight.mean(dim=0)
            outputs_all[i] = outputs_test
            weights_all[:, i] = domain_weight*args.w[i]
            if start_each_domain:
                domain_preidctions = weights_test
                src_domain_labels = torch.full((inputs_test.shape[0], 1), int(i), device=inputs_test.device)
                start_each_domain = False
            else:
                src_domain_labels = torch.cat((src_domain_labels, torch.full((inputs_test.shape[0], 1), int(i), device=inputs_test.device)), 0)
                domain_preidctions = torch.cat((domain_preidctions, weights_test), 0)
        

        z = torch.sum(weights_all, dim=1)
        z = z + 1e-16
        weights_all = torch.transpose(torch.transpose(weights_all,0,1)/z,0,1)
        outputs_all = torch.transpose(outputs_all, 0, 1)
        z_ = weights_all[0:1][0]

        features_all = torch.transpose(features_all, 0, 1)
        features_all_F = torch.transpose(features_all_F, 0, 1)
        #print(z_)

        for i in range(inputs_test.shape[0]):
            outputs_all_w[i] = torch.matmul(torch.transpose(outputs_all[i], 0, 1), weights_all[i])
            features_all_w[i] = torch.matmul(torch.transpose(features_all[i], 0, 1), weights_all[i])
            features_all_F_w[i] = torch.matmul(torch.transpose(features_all_F[i], 0, 1), weights_all[i])

        if start_test:
            all_output = outputs_all_w.float().cpu()
            all_feature = features_all_w.float().cpu()
            all_feature_F = features_all_F_w.float().cpu()
            start_test = False
        else:
            all_output = torch.cat((all_output, outputs_all_w.float().cpu()), 0)
            all_feature = torch.cat((all_feature, features_all_w.float().cpu()), 0)
            all_feature_F = torch.cat((all_feature_F, features_all_F_w.float().cpu()), 0)

        softmax_out = nn.Softmax(dim=1)(outputs_all_w)

        features_test_N, _, _ = obtain_nearest_trace(all_feature_F, feas_all, label_confi)
        # 64 *2048
        features_test_N = features_test_N.cuda()
        #
        for i in range(len(args.src)):
            outputs_test_N = netC_list[i](netB_list[i](features_test_N))
            outputs_all_N[i] = outputs_test_N

        outputs_all_N = torch.transpose(outputs_all_N, 0, 1)

        for i in range(inputs_test.shape[0]):
            outputs_all_w_N[i] = torch.matmul(torch.transpose(outputs_all_N[i], 0, 1), weights_all[i])

        softmax_out_hyper = nn.Softmax(dim=1)(outputs_all_w_N)
        classifier_loss = torch.tensor(0.0).cuda()
        iic_loss = IID_losses.IID_loss(softmax_out, softmax_out_hyper)
        classifier_loss +=  args.iic_par * iic_loss

        if args.cls_par > 0:
            initc_ = torch.zeros(initc[0].size()).cuda()
            temp = all_feas[0]
            all_feas_ = torch.zeros(temp[tar_idx, :].size()).cuda()
            for i in range(len(args.src)):
                initc_ = initc_ + z_[i] * initc[i].float()
                src_fea = all_feas[i]
                all_feas_ = all_feas_ + z_[i] * src_fea[tar_idx, :]
            dd = torch.cdist(all_feas_.float(), initc_.float(), p=2)
            pred_label = dd.argmin(dim=1)
            pred_label = pred_label.int()
            pred = pred_label.long()
            classifier_loss += args.cls_par * nn.CrossEntropyLoss()(outputs_all_w.cuda(), pred.cuda())
        else:
            classifier_loss = torch.tensor(0.0)

        msoftmax = softmax_out.mean(dim=0)
        gentropy_loss = torch.sum(-msoftmax * torch.log(msoftmax + args.epsilon))
        gentropy_loss = gentropy_loss * args.gent_par
        classifier_loss = classifier_loss - gentropy_loss

        if args.dc_loss:
            src_domain_labels = src_domain_labels.view(-1)
            domain_classification_loss = nn.CrossEntropyLoss(weight=args.w)(domain_preidctions.cuda(), src_domain_labels.cuda())
            #domain_classification_loss_adv = nn.CrossEntropyLoss()(domain_preidctions_adv.cuda(), src_domain_labels.cuda())

            #domain_classification_loss = nn.CrossEntropyLoss()(domain_preidctions.cuda(), src_domain_labels.cuda())
            total_domain_loss = args.dc_loss_par * domain_classification_loss #+ 0.4*domain_classification_loss_adv
        else:
            total_domain_loss = torch.zeros((), device=inputs_test.device)


        total_loss =  classifier_loss + total_domain_loss
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        iter_num += 1
        lr_scheduler(optimizer, iter_num=iter_num, max_iter=max_iter)

        if iter_num % interval_iter == 0 or iter_num == max_iter:
            for i in range(len(args.src)):
                netF_list[i].eval()
                netB_list[i].eval()
                netDC_inter.eval()
            acc, _, z_, t_ = cal_acc_multi(dset_loaders['test'], netF_list, netB_list, netC_list, netDC_inter, args)
            log_str = 'Iter:{}/{}; Classification Accuracy = {:.2f}%'.format(iter_num, max_iter, acc)
            print(log_str+'\n')

            args.out_file.write(log_str + '\n')
            args.out_file.flush()

def obtain_label_alpha(loader, netF, netB, netC, args, obtain_prior=True):
    start_test = True
    with torch.no_grad():
        iter_test = iter(loader)
        for _ in range(len(loader)):
            data = next(iter_test)
            inputs = data[0]
            labels = data[1]
            inputs = inputs.cuda()
            feas = netB(netF(inputs.float()))
            feas_uniform = F.normalize(feas)
            outputs = netC(feas)
            if start_test:
                all_fea = feas_uniform.float().cpu()
                all_output = outputs.float().cpu()
                all_label = labels.float()
                start_test = False
            else:
                all_fea = torch.cat((all_fea, feas_uniform.float().cpu()), 0)
                all_output = torch.cat((all_output, outputs.float().cpu()), 0)
                all_label = torch.cat((all_label, labels.float()), 0)
    if obtain_prior:
        alpha = obtain_domain_prior_tiled(all_fea, all_output, args)
    all_output = nn.Softmax(dim=1)(all_output)
    _, predict = torch.max(all_output, 1)
    accuracy = torch.sum(torch.squeeze(predict).float() == all_label).item() / float(all_label.size()[0])
    all_fea = all_fea.float().cpu().numpy()

    K = all_output.size(1)
    aff = all_output.float().cpu().numpy()
    initc = aff.transpose().dot(all_fea)
    initc = initc / (1e-8 + aff.sum(axis=0)[:,None])

    dd = cdist(all_fea, initc, 'cosine')
    pred_label = dd.argmin(axis=1)
    acc = np.sum(pred_label == all_label.float().numpy()) / len(all_fea)

    for round in range(1):
        aff = np.eye(K)[pred_label]
        initc = aff.transpose().dot(all_fea)
        initc = initc / (1e-8 + aff.sum(axis=0)[:,None])
        dd = cdist(all_fea, initc, 'cosine')
        pred_label = dd.argmin(axis=1)
        acc = np.sum(pred_label == all_label.float().numpy()) / len(all_fea)

    log_str = 'Accuracy = {:.2f}% -> {:.2f}%'.format(accuracy*100, acc*100)
    print(log_str+'\n')
    #return pred_label.astype('int')
    if obtain_prior:
        return initc, all_fea, alpha
    else:
        return initc, all_fea

def obtain_label_ts(loader, netF_list, netB_list, netC_list, netDC_inter, args, iter_num_update_f):
    start_test = True
    with torch.no_grad():
        iter_test = iter(loader)
        for _ in range(len(loader)):
            data = next(iter_test)
            inputs = data[0]
            labels = data[1]
            inputs = inputs.cuda()

            outputs_all = torch.zeros(len(args.src), inputs.shape[0], args.class_num, device=inputs.device)

            outputs_all_w = torch.zeros(inputs.shape[0], args.class_num, device=inputs.device)
            features_all = torch.zeros(len(args.src), inputs.shape[0], args.bottleneck, device=inputs.device)
            features_all_w = torch.zeros(inputs.shape[0], args.bottleneck, device=inputs.device)
            features_all_F = torch.zeros(len(args.src), inputs.shape[0], netF_list[0].in_features, device=inputs.device)
            features_all_F_w = torch.zeros(inputs.shape[0], netF_list[0].in_features, device=inputs.device)
            weights_all = torch.ones(inputs.shape[0], len(args.src), device=inputs.device)

            for i in range(len(args.src)):
                features_F = netF_list[i](inputs)
                features = netB_list[i](features_F)
                outputs = netC_list[i](features)

                features_all_F[i] = features_F
                features_all[i] = features
                weights_numerator = netDC_inter(features)
                weights_test = weights_numerator
                softmax_weights = nn.Softmax(dim=1)(weights_test)

                domain_weight = softmax_weights[:, i]
                domain_weight = domain_weight.mean(dim=0)
                outputs_all[i] = outputs
                weights_all[:, i] = domain_weight * args.w[i]


            z = torch.sum(weights_all, dim=1)
            z = z + 1e-16
            weights_all = torch.transpose(torch.transpose(weights_all, 0, 1) / z, 0, 1)
            outputs_all = torch.transpose(outputs_all, 0, 1)
            features_all = torch.transpose(features_all, 0, 1)
            features_all_F = torch.transpose(features_all_F, 0, 1)


            for i in range(inputs.shape[0]):
                outputs_all_w[i] = torch.matmul(torch.transpose(outputs_all[i], 0, 1), weights_all[i])
                features_all_w[i] = torch.matmul(torch.transpose(features_all[i], 0, 1), weights_all[i])
                features_all_F_w[i] = torch.matmul(torch.transpose(features_all_F[i], 0, 1), weights_all[i])

            if start_test:
                # all_fea_F = feas_F.float().cpu()
                # all_fea = feas.float().cpu()
                # all_output = outputs.float().cpu()
                # all_label = labels.float()
                # start_test = False
                all_output = outputs_all_w.float().cpu()
                all_feature = features_all_w.float().cpu()
                all_feature_F = features_all_F_w.float().cpu()
                all_label = labels.float()
                start_test = False
            else:
                # all_fea_F = torch.cat((all_fea_F, feas_F.float().cpu()), 0)
                # all_fea = torch.cat((all_fea, feas.float().cpu()), 0)
                # all_output = torch.cat((all_output, outputs.float().cpu()), 0)
                # all_label = torch.cat((all_label, labels.float()), 0)
                all_output = torch.cat((all_output, outputs_all_w.float().cpu()), 0)
                all_feature = torch.cat((all_feature, features_all_w.float().cpu()), 0)
                all_feature_F = torch.cat((all_feature_F, features_all_F_w.float().cpu()), 0)
                all_label = torch.cat((all_label, labels.float()), 0)

    # all_logis = all_output
    all_output = nn.Softmax(dim=1)(all_output)
    ent = torch.sum(-all_output * torch.log(all_output + args.epsilon), dim=1)
    unknown_weight = 1 - ent / np.log(args.class_num)
    _, predict = torch.max(all_output, 1)

    len_unconfi = int(ent.shape[0]*0.5)
    idx_unconfi = ent.topk(len_unconfi, largest=True)[-1]
    idx_unconfi_list_ent = idx_unconfi.cpu().numpy().tolist()

    accuracy = torch.sum(torch.squeeze(predict).float() == all_label).item() / float(all_label.size()[0])
    if args.distance == 'cosine':
        all_feature = torch.cat((all_feature, torch.ones(all_feature.size(0), 1)), 1)
        all_feature = (all_feature.t() / torch.norm(all_feature, p=2, dim=1)).t()

    all_feature = all_feature.float().cpu().numpy()
    K = all_output.size(1)
    aff = all_output.float().cpu().numpy()
    initc = aff.transpose().dot(all_feature)
    initc = initc / (1e-8 + aff.sum(axis=0)[:,None])

    cls_count = np.eye(K)[predict].sum(axis=0)
    labelset = np.where(cls_count>args.threshold)
    labelset = labelset[0]
    # print(labelset)

    dd = cdist(all_feature, initc[labelset], args.distance)
    pred_label = dd.argmin(axis=1)
    pred_label = labelset[pred_label]

    # --------------------use dd to get confi_idx and unconfi_idx-------------
    dd_min = dd.min(axis = 1)
    dd_min_tsr = torch.from_numpy(dd_min).detach()
    dd_t_confi = dd_min_tsr.topk(int((dd.shape[0]*0.6)), largest = False)[-1]
    dd_confi_list = dd_t_confi.cpu().numpy().tolist()
    dd_confi_list.sort()
    idx_confi = dd_confi_list

    idx_all_arr = np.zeros(shape = dd.shape[0], dtype = np.int64)
    idx_all_arr[idx_confi] = 1
    idx_unconfi_arr = np.where(idx_all_arr == 0)
    idx_unconfi_list_dd = list(idx_unconfi_arr[0])

    idx_unconfi_list = list(set(idx_unconfi_list_dd).intersection(set(idx_unconfi_list_ent)))
    # ------------------------------------------------------------------------
    # idx_unconfi_list = idx_unconfi_list_dd # idx_unconfi_list_dd

    label_confi = np.ones(ent.shape[0], dtype="int64")
    label_confi[idx_unconfi_list] = 0

    acc = np.sum(pred_label == all_label.float().numpy()) / len(all_feature)
    log_str = '{:.1f} AccuracyEpoch = {:.2f}% -> {:.2f}%'.format(iter_num_update_f, accuracy * 100, acc * 100)

    args.out_file.write(log_str + '\n')
    args.out_file.flush()
    print(log_str+'\n')

    return pred_label.astype('int'), all_feature_F, label_confi, all_label, all_output


def obtain_nearest_trace(data_q, data_all, lab_confi):
    data_q_ = data_q.detach()
    data_all_ = data_all.detach()
    data_q_ = data_q_.cpu().numpy()
    data_all_ = data_all_.cpu().numpy()
    num_sam = data_q.shape[0]
    LN_MEM = 70

    flag_is_done = 0  # indicate whether the trace process has done over the target dataset
    ctr_oper = 0  # counter the operation time
    idx_left = np.arange(0, num_sam, 1)
    mtx_mem_rlt = -3 * np.ones((num_sam, LN_MEM), dtype='int64')
    mtx_mem_ignore = np.zeros((num_sam, LN_MEM), dtype='int64')
    is_mem = 0
    mtx_log = np.zeros((num_sam, LN_MEM), dtype='int64')
    indices_row = np.arange(0, num_sam, 1)
    flag_sw_bad = 0
    nearest_idx_last = np.array([-7])

    while flag_is_done == 0:

        nearest_idx_tmp, idx_last_tmp = get_nearest_sam_idx(data_q_, data_all_, is_mem, ctr_oper, mtx_mem_ignore,
                                                            nearest_idx_last)
        is_mem = 1
        nearest_idx_last = nearest_idx_tmp

        if ctr_oper == (LN_MEM - 1):
            flag_sw_bad = 1
        else:
            flag_sw_bad = 0

        mtx_mem_rlt[:, ctr_oper] = nearest_idx_tmp
        mtx_mem_ignore[:, ctr_oper] = idx_last_tmp

        lab_confi_tmp = lab_confi[nearest_idx_tmp]
        idx_done_tmp = np.where(lab_confi_tmp == 1)[0]
        idx_left[idx_done_tmp] = -1

        if flag_sw_bad == 1:
            idx_bad = np.where(idx_left >= 0)[0]
            mtx_log[idx_bad, 0] = 1
        else:
            mtx_log[:, ctr_oper] = lab_confi_tmp

        flag_len = len(np.where(idx_left >= 0)[0])
        # print("{}--the number of left:{}".format(str(ctr_oper), flag_len))

        if flag_len == 0 or flag_sw_bad == 1:
            # idx_nn_tmp = [list(mtx_log[k, :]).index(1) for k in range(num_sam)]
            idx_nn_step = []
            for k in range(num_sam):
                try:
                    idx_ts = list(mtx_log[k, :]).index(1)
                    idx_nn_step.append(idx_ts)
                except:
                    print("ts:", k, mtx_log[k, :])
                    # mtx_log[k, 0] = 1
                    idx_nn_step.append(0)

            idx_nn_re = mtx_mem_rlt[indices_row, idx_nn_step]
            data_re = data_all[idx_nn_re, :]
            flag_is_done = 1
        else:
            data_q_ = data_all_[nearest_idx_tmp, :]
        ctr_oper += 1

    return data_re, idx_nn_re, idx_nn_step  # array


def get_nearest_sam_idx(Q, X, is_mem_f, step_num, mtx_ignore,
                        nearest_idx_last_f):  # Q、X arranged in format of row-vector
    Xt = np.transpose(X)
    Simo = np.dot(Q, Xt)
    nq = np.expand_dims(LA.norm(Q, axis=1), axis=1)
    nx = np.expand_dims(LA.norm(X, axis=1), axis=0)
    Nor = np.dot(nq, nx)
    epsilon = 1e-10
    Sim = 1 - (Simo / (Nor + epsilon ))

    # Sim = cdist(Q, X, "cosine") # too slow
    # print('eeeeee \n', Sim)

    indices_min = np.argmin(Sim, axis=1)
    indices_row = np.arange(0, Q.shape[0], 1)

    idx_change = np.where((indices_min - nearest_idx_last_f) != 0)[0]
    if is_mem_f == 1:
        if idx_change.shape[0] != 0:
            indices_min[idx_change] = nearest_idx_last_f[idx_change]
    Sim[indices_row, indices_min] = 1000
    # Ignore the history elements.
    if is_mem_f == 1:
        for k in range(step_num):
            indices_ingore = mtx_ignore[:, k]
            Sim[indices_row, indices_ingore] = 1000

    indices_min_cur = np.argmin(Sim, axis=1)
    indices_self = indices_min
    return indices_min_cur, indices_self

def cal_acc_multi(loader, netF_list, netB_list, netC_list, netDC_inter, args):
    start_test = True
    with torch.no_grad():
        iter_test = iter(loader)
        for _ in range(len(loader)):
            data = next(iter_test)
            inputs = data[0]
            labels = data[1]
            inputs = inputs.cuda()
            outputs_all = torch.zeros(len(args.src), inputs.shape[0], args.class_num, device=inputs.device)
            weights_all = torch.ones(inputs.shape[0], len(args.src), device=inputs.device)
            transferability = torch.ones(inputs.shape[0], len(args.src), device=inputs.device)
            outputs_all_w = torch.zeros(inputs.shape[0], args.class_num, device=inputs.device)

            for i in range(len(args.src)):
                features = netB_list[i](netF_list[i](inputs))
                outputs = netC_list[i](features)

                weights_numerator = netDC_inter(features)#Numerator and denominator
                #weights_denominator = netG_list(features)
                weights_test = weights_numerator#/weights_denominator
                softmax_weights = nn.Softmax(dim=1)(weights_test)
                domain_weight = softmax_weights[:, i]
                domain_weight = domain_weight.mean(dim=0)
                weights_all[:, i] = domain_weight*args.w[i]
                transferability[:, i] = domain_weight
                outputs_all[i] = outputs
                #weights_all[:, i] = domain_weight.squeeze()

            z = torch.sum(weights_all, dim=1)
            z = z + 1e-16
            #print(weights_all[0:1][0])
            weights_all = torch.transpose(torch.transpose(weights_all,0,1)/z,0,1)
            #print(outputs_all.size())
            outputs_all = torch.transpose(outputs_all, 0, 1)
            #print(outputs_all.size())
            t_ = transferability[0:1][0]
            z_ = weights_all[0:1][0]
            #print(z_)
            for i in range(inputs.shape[0]):
                outputs_all_w[i] = torch.matmul(torch.transpose(outputs_all[i],0,1), weights_all[i])

            if start_test:
                all_output = outputs_all_w.float().cpu()
                all_label = labels.float()
                start_test = False
            else:
                all_output = torch.cat((all_output, outputs_all_w.float().cpu()), 0)
                all_label = torch.cat((all_label, labels.float()), 0)

    _, predict = torch.max(all_output, 1)

    accuracy = torch.sum(torch.squeeze(predict).float() == all_label).item() / float(all_label.size()[0])
    mean_ent = torch.mean(loss.Entropy(nn.Softmax(dim=1)(all_output))).cpu().data.item()
    return accuracy*100, mean_ent, z_, t_
def print_args(args):
    s = "==========================================\n"
    for arg, content in args.__dict__.items():
        s += "{}:{}\n".format(arg, content)
    return s

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='ours')
    parser.add_argument('--gpu_id', type=str, nargs='?', default='0', help="device id to run")
    parser.add_argument('--t', type=int, default=1, help="target") ## Choose which domain to set as target {0 to len(names)-1}
    parser.add_argument('--max_epoch', type=int, default=15, help="max iterations")
    parser.add_argument('--interval', type=int, default=15)
    parser.add_argument('--batch_size', type=int, default=64, help="batch_size")
    parser.add_argument('--worker', type=int, default=4, help="number of workers")
    parser.add_argument('--prefetch_factor', type=int, default=1,
                        help='batches prefetched per worker')
    parser.add_argument('--activation_checkpoint', type=int, choices=[0, 1], default=1,
                        help='1 recomputes backbone activations to fit five source models in 24 GB')
    parser.add_argument('--dset', type=str, default='office-home', choices=['office-31', 'office-home', 'office-caltech','DomainNet'])
    parser.add_argument('--lr', type=float, default=1*1e-2, help="learning rate")
    parser.add_argument('--net', type=str, default='resnet50', help="vgg16, resnet50, res101")
    parser.add_argument('--seed', type=int, default=2021, help="random seed")
    parser.add_argument('--pre_step', type=int, default=50, help="pretrain step of domain classifier")

    parser.add_argument('--gent', type=bool, default=True)
    parser.add_argument('--ent', type=bool, default=True)
    parser.add_argument('--dc_loss', type=bool, default=True)
    parser.add_argument('--threshold', type=int, default=0)



    parser.add_argument('--iic_par', type=float, default=1.0)
    parser.add_argument('--cls_par', type=float, default=0.2)
    parser.add_argument('--gent_par', type=float, default=0.2)
    parser.add_argument('--dc_loss_par', type=float, default=0.1)

    parser.add_argument('--ent_par', type=float, default=0.5)
    parser.add_argument('--lr_decay1', type=float, default=0.1)
    parser.add_argument('--lr_decay2', type=float, default=1.0)
    parser.add_argument('--lr_decay3', type=float, default=0.1)
    # 所选数据的比例
    parser.add_argument('--ratio', type=float, default=0.5, help="the ratio of selected data")
    parser.add_argument('--K', type=int, default=20, help="the number of selected neighbors")

    parser.add_argument('--bottleneck', type=int, default=256)
    parser.add_argument('--epsilon', type=float, default=1e-5)
    parser.add_argument('--layer', type=str, default="wn", choices=["linear", "wn"])
    parser.add_argument('--classifier', type=str, default="bn", choices=["ori", "bn"])
    parser.add_argument('--distance', type=str, default='cosine', choices=["euclidean", "cosine"])  
    parser.add_argument('--output', type=str, default='ckps/adapt_ours_TPDS_sensitive')
    parser.add_argument('--output_src', type=str,
                        default='ckps/source/uda',
                        help='shared source-bank root')
    parser.add_argument('--search_device', choices=['auto', 'cuda', 'cpu'], default='auto',
                        help='device for exact tiled source-prior top-k search')
    parser.add_argument('--prior_query_chunk', type=int, default=256,
                        help='source-prior query rows processed at once')
    args = parser.parse_args()
    
    if args.dset == 'office-home':
        names = ['Art', 'Clipart', 'Product', 'Real_World']
        args.class_num = 65
    if args.dset == 'office-31':
        names = ['amazon', 'dslr' , 'webcam']
        args.class_num = 31
    if args.dset == 'office-caltech':
        names = ['amazon', 'caltech', 'dslr', 'webcam']
        args.class_num = 10
    if args.dset == 'DomainNet':
        names = ['clipart','infograph','painting','quickdraw','real','sketch']
        args.class_num = 345
    if not 0 <= args.t < len(names):
        raise ValueError(f'--t must be in [0, {len(names) - 1}], received {args.t}.')
    if args.prior_query_chunk < 1:
        raise ValueError('--prior_query_chunk must be positive.')
    if args.worker < 0 or args.prefetch_factor < 1:
        raise ValueError('--worker must be non-negative and --prefetch_factor must be positive.')
    args.src = []
    for i in range(len(names)):
        if i == args.t:
            continue
        else:
            args.src.append(names[i])

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    SEED = args.seed
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)

    for i in range(len(names)):
        if i != args.t:
            continue
        folder = './data/'
        args.t_dset_path = folder + args.dset + '/' + names[args.t] + '_list.txt'
        args.test_dset_path = folder + args.dset + '/' + names[args.t] + '_list.txt'
        print(args.t_dset_path)

    args.output_dir_src = []
    for i in range(len(args.src)):
        source_dir = Path(args.output_src) / args.dset / args.src[i][0].upper()
        required = [source_dir / name for name in ('source_F.pt', 'source_B.pt', 'source_C.pt')]
        missing = [str(item) for item in required if not item.is_file()]
        if missing:
            raise FileNotFoundError('Missing shared source checkpoint(s): ' + ', '.join(missing))
        args.output_dir_src.append(str(source_dir))
    print(args.output_dir_src)
    args.output_dir = osp.join(args.output, args.dset, names[args.t][0].upper())

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    ctime = time.localtime()
    year, month, day, hour, miniute = ctime.tm_year, ctime.tm_mon, ctime.tm_mday, ctime.tm_hour, ctime.tm_min
    args.savename = 'cls_' + str(args.cls_par)+ '_gent_' + str(args.gent_par) + '_dcloss_' + str(args.dc_loss_par) + '_' + str(year) + '-' + str(month) + '-' + str(day) + '-' + str(hour) + '-' + str(miniute)

    args.out_file = open(osp.join(args.output_dir, 'log_' + args.savename + '.txt'), 'w')
    args.out_file.write(print_args(args)+'\n')
    args.out_file.flush()

    train_target(args)
