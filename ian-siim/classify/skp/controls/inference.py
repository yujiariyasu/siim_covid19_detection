import cv2
import gc
import glob
import numpy as np
import os, os.path as osp
import pandas as pd
import pickle
import random
import re
import torch
import torch.nn as nn
import torch.nn.functional as F

from omegaconf import OmegaConf
from scipy.stats import rankdata
from torch.nn.parallel import DistributedDataParallel
from tqdm import tqdm
from .datamaker import get_train_val_test_splits, prepend_filepath
from .. import builder


def load_model(cfg, args):
    if cfg.model.name.lower() in ['puzzlenet']:
        cfg.model.params.pretrained = False

    model = builder.build_model(cfg)
    weights = torch.load(cfg.checkpoint, map_location=lambda storage, loc: storage)['state_dict']
    weights = {re.sub(r'^model.', '', k) : v for k,v in weights.items()}
    if cfg.local_rank == 0:
        print(f'Loading checkpoint from {cfg.checkpoint} ...')
    model.load_state_dict(weights)
    if args.accelerator == 'ddp':
        if not args.cpu_inference:
            if isinstance(cfg.local_rank, int): 
                model.to(f'cuda:{cfg.local_rank}')
            else:
                model.cuda()
        model = DistributedDataParallel(model, device_ids=[cfg.local_rank], output_device=cfg.local_rank)
    else:
        if not args.cpu_inference:
            if isinstance(cfg.local_rank, int): 
                model.to(f'cuda:{cfg.local_rank}')
            else:
                model.cuda()
    model.eval()

    return model


def load_pickle(f):
    with open(f, 'rb') as f:
        return pickle.load(f)


def is_tensor(x): 
    return isinstance(x, torch.Tensor)


def _cudaify(x, device):
    dev = f'cuda:{device}'
    if isinstance(x, dict):
        return {k:v.to(dev) if is_tensor(v) else v for k,v in x.items()}

    if isinstance(x, (tuple,list)):
        return type(x)([_.to(dev) if is_tensor(_) else _cudaify(_) for _ in x])

    return x.to(dev)


def cudaify(batch, labels, device): 
    return _cudaify(batch, device), _cudaify(labels, device)


class Ensemble(nn.Module):

    def __init__(self, models):
        super().__init__()
        self.models = nn.ModuleList(models)

    def forward(self, x, act_fn=None, flip=False):
        p = []
        for m in self.models:
            if random.random() > 0.5 and flip:
                out = m(torch.flip(x, dims=(-1,)))
            else:
                out = m(x)
            if act_fn == 'softmax':
                p.append(torch.softmax(out, dim=1))
            elif act_fn == 'sigmoid':
                p.append(torch.sigmoid(out))
            else:
                p.append(out)
        p = torch.mean(torch.stack(p, dim=0), dim=0)
        return p


def setup(cfg, args, mode='predict', return_preprocessor=False):
    cfg.local_rank = cfg.local_rank or 0
    assert args.checkpoint, 'Must specify checkpoint for inference'
    if len(args.checkpoint) == 1:
        cfg.checkpoint = args.checkpoint[0]
        model = load_model(cfg, args)
    else:
        model_list = []
        for ckpt in args.checkpoint:
            cfg.checkpoint = ckpt
            model_list.append(load_model(cfg, args))
        model = Ensemble(model_list)
        del model_list

    if mode == 'test':
        df = pd.read_csv(cfg.data.annotations)
        _,_,test_df = get_train_val_test_splits(cfg, df)
        imgfiles = test_df[cfg.data.input or 'imgfile']
    elif mode == 'extract':
        df = pd.read_csv(cfg.data.annotations)
        imgfiles = df[cfg.data.input or 'imgfile']        
    else:
        with open(args.imgfiles) as f:
            imgfiles = [line.strip() for line in f.readlines()]

    if args.data_dir:
        imgfiles = [osp.join(args.data_dir, i) for i in imgfiles]

    if args.fast_dev_run:
        imgfiles = imgfiles[:args.fast_dev_run]

    if cfg.local_rank == 0:
        print(f'PREDICT : n={len(imgfiles)}')

    cfg.data.dataset.params.return_name = True

    data_info = dict(
        inputs=imgfiles,
        labels=[0]*len(imgfiles)
    )

    if cfg.data.dataset.name == 'SegmentClassify':
        cfg.data.dataset.name = 'ImageDataset'

    if cfg.data.dataset.params.add_invert_label:
        cfg.data.dataset.params.add_invert_label = False

    dataset = builder.build_dataset(cfg, 
        data_info=data_info,
        mode='predict')
    loader = builder.build_dataloader(cfg, dataset=dataset, mode='predict',
        dist_inference=args.accelerator == 'ddp')

    # For segmentation, `args.save_preds_file` is a directory
    # where each segmentation will be saved to a uint8 numpy array
    if cfg.local_rank == 0:
        if args.mode == 'get_cams':
            print(f'Saving CAMs to {args.save_cams_dir} ...')
            save_dir = osp.dirname(args.save_cams_dir)
            if save_dir != '':
                if not osp.exists(save_dir): os.makedirs(save_dir)

        if args.mode == 'segment':
            save_dir = osp.dirname(args.save_preds_file)
            if save_dir != '':
                if not osp.exists(save_dir): os.makedirs(save_dir)

        if args.mode == 'pseudoseg':
            save_dir = osp.dirname(args.save_pseudo_segs)
            if save_dir != '':
                if not osp.exists(save_dir): os.makedirs(save_dir)

    if cfg.local_rank == 0:
        iterator = tqdm(enumerate(loader), total=len(loader))
    else:
        iterator = enumerate(loader)

    if return_preprocessor:
        return model, iterator, loader.dataset.preprocess

    return model, iterator


def extract_features(cfg, args):
    model, iterator = setup(cfg, args, mode='extract')

    if not osp.exists(args.save_features_dir):
        os.makedirs(args.save_features_dir)

    y_pred, names = [],[]
    for _ind, data in iterator:
        batch, labels, _names = data
        batch, labels = cudaify(batch, labels, device=cfg.local_rank)
        B,C,H,W = batch.size()
        with torch.no_grad():
            for i in range(B):
                features = model.extract_features(batch[i].unsqueeze(1))
                np.save(osp.join(args.save_features_dir, osp.basename(_names[i])), features.cpu().numpy())


def pseudoseg(cfg, args, mode='predict'):
    model, iterator = setup(cfg, args, mode=mode)

    y_pred, names = [],[]
    for _ind, data in iterator:
        batch, labels, _names = data
        batch, labels = cudaify(batch, labels, device=cfg.local_rank)
        with torch.no_grad():
            output = model(batch)

        outseg, outcls = output
        outseg = torch.sigmoid(outseg).cpu().numpy()
        outcls = torch.sigmoid(outcls).cpu().numpy()


        y_pred += list(outcls)
        names.extend(list(_names))

        for ii, o in enumerate(outseg):
            tmp_name = _names[ii].split('/')[-1]
            cv2.imwrite(osp.join(args.save_pseudo_segs, tmp_name), 
                (outseg[ii]*255).astype('uint8').transpose(1,2,0))

    y_pred = np.asarray(y_pred)
    imgfiles = names
    
    assert 'csv' in args.save_preds_file or 'pkl' in args.save_preds_file, \
        f'Predictions must be saved as `csv` or `pkl`'
    
    if args.accelerator == 'ddp': 
        with open(f'.tmp_preds_{cfg.local_rank}.pkl', 'wb') as f:
            pickle.dump(dict(imgfile=names, p=y_pred), f)
        with open(f'.done_rank{cfg.local_rank}.txt', 'w') as f:
            _ = f.write('done')

    if cfg.local_rank == 0:
        if not osp.exists(osp.dirname(args.save_preds_file)):
            print(f'Creating directory {args.save_preds_file} ...')
            os.makedirs(osp.dirname(args.save_preds_file))

        if args.accelerator == 'ddp':
            while 1:
                if len(glob.glob('.done_rank*txt')) == cfg.world_size:
                    break
            predictions = glob.glob('.tmp_preds_*pkl')
            predictions = [load_pickle(f) for f in predictions]
            imgfiles = predictions[0]['imgfile']
            for pred in predictions[1:]:
                imgfiles.extend(pred['imgfile'])
            y_pred = np.concatenate([pred['p'] for pred in predictions])
            os.system('rm .tmp_preds*pkl ; rm .done_rank*txt')
        if 'csv' in args.save_preds_file:
            save_df = pd.DataFrame(dict(imgfile=imgfiles))
            for c in range(y_pred.shape[-1]):
                save_df[f'p{c}'] = y_pred[:,c]
            save_df.to_csv(args.save_preds_file, index=False)
        elif 'pkl' in args.save_preds_file:
            with open(args.save_preds_file, 'wb') as f:
                pickle.dump(dict(imgfile=imgfiles, p=y_pred), f)


def predict(cfg, args, mode='predict'):
    model, iterator = setup(cfg, args, mode=mode)

    y_pred, names = [],[]
    for _ind, data in iterator:
        batch, labels, _names = data
        batch, labels = cudaify(batch, labels, device=cfg.local_rank)
        with torch.no_grad():
            output = model(batch)

        if isinstance(output, tuple):
            maxseg = output[0].view(batch.size(0), -1).max(1)[0].unsqueeze(1)
            output = output[1] # (N, 5)
            output = torch.cat([output, maxseg], dim=1)

        if args.act_fn == 'softmax':
            output = torch.softmax(output, dim=1).cpu().numpy()
        elif args.act_fn == 'sigmoid':
            output = torch.sigmoid(output).cpu().numpy()
        else:
            output = output.cpu().numpy()

        y_pred += list(output)
        names.extend(list(_names))

    y_pred = np.asarray(y_pred)
    imgfiles = names
    
    assert 'csv' in args.save_preds_file or 'pkl' in args.save_preds_file, \
        f'Predictions must be saved as `csv` or `pkl`'
    
    if args.accelerator == 'ddp': 
        with open(f'.tmp_preds_{cfg.local_rank}.pkl', 'wb') as f:
            pickle.dump(dict(imgfile=names, p=y_pred), f)
        with open(f'.done_rank{cfg.local_rank}.txt', 'w') as f:
            _ = f.write('done')

    if cfg.local_rank == 0:
        if not osp.exists(osp.dirname(args.save_preds_file)):
            print(f'Creating directory {args.save_preds_file} ...')
            os.makedirs(osp.dirname(args.save_preds_file))

        if args.accelerator == 'ddp':
            while 1:
                if len(glob.glob('.done_rank*txt')) == cfg.world_size:
                    break
            predictions = glob.glob('.tmp_preds_*pkl')
            predictions = [load_pickle(f) for f in predictions]
            imgfiles = predictions[0]['imgfile']
            for pred in predictions[1:]:
                imgfiles.extend(pred['imgfile'])
            y_pred = np.concatenate([pred['p'] for pred in predictions])
            os.system('rm .tmp_preds*pkl ; rm .done_rank*txt')
        if 'csv' in args.save_preds_file:
            save_df = pd.DataFrame(dict(imgfile=imgfiles))
            for c in range(y_pred.shape[-1]):
                save_df[f'p{c}'] = y_pred[:,c]
            save_df.to_csv(args.save_preds_file, index=False)
        elif 'pkl' in args.save_preds_file:
            with open(args.save_preds_file, 'wb') as f:
                pickle.dump(dict(imgfile=imgfiles, p=y_pred), f)


def test(cfg, args):
    predict(cfg, args, mode='test')


def convert_output_to_original_size(output, imsize, resize_ignore=False):
    if resize_ignore:
        return torch.nn.functional.interpolate(output.unsqueeze(0), size=tuple(imsize), mode='bilinear').squeeze(0)
    # output.shape = (C, H, W)
    # imsize = (H, W)
    # 1- Find out which side (shorter) was padded
    padded = np.argmin(imsize)
    # 2- Find out scale factor (using unpadded side)
    scale_factor = np.max(imsize) / output.shape[int(2-padded)]
    # 3- Rescale
    output = torch.nn.functional.interpolate(output.unsqueeze(0), scale_factor=scale_factor, mode='bilinear').squeeze(0)
    # 4- Determine padding
    padding = output.shape[int(padded+1)] - np.min(imsize)
    pad_a = padding // 2
    pad_b = padding - pad_a
    # 5- Removing padding
    if padded == 0:
        output = output[:,pad_a:output.shape[1]-pad_b,:]
    elif padded == 1:
        output = output[:,:,pad_a:output.shape[2]-pad_b]
    assert tuple(output.shape[1:]) == tuple(imsize), f'Output size {output.shape[1:]} does not equal original image size {imsize}'
    return output 


def segment(cfg, args, mode='predict'):
    model, iterator = setup(cfg, args, mode=mode)

    y_pred, names = [],[]
    for _ind, data in iterator:
        batch, labels, _names = data
        batch, labels = cudaify(batch, labels, device=cfg.local_rank)
        with torch.no_grad():
            output = model(batch)

        if args.act_fn == 'softmax':
            output = torch.softmax(output, dim=1)
        elif args.act_fn == 'sigmoid':
            output = torch.sigmoid(output)

        output = output.cpu().numpy()
        output = (output>=0.5).astype('float')

        # imsizes = torch.stack(imsizes).transpose(1, 0).cpu().numpy()
        # output = [
        #     convert_output_to_original_size(output[ii], imsizes[ii], 
        #         resize_ignore=cfg.transform.resize.name == 'resize_ignore').numpy() 
        #     for ii in range(output.shape[0])
        # ]

        for ii, o in enumerate(output):
            tmp_name = _names[ii].split('/')[-1].replace('dcm','png')
            cv2.imwrite(osp.join(args.save_preds_file, tmp_name), 
                (output[ii]*255).astype('uint8').transpose(1,2,0))


def segment_test(cfg, args):
    segment(cfg, args, mode='test')


def load_single_model_or_ensemble(config_list, weight_list, args):
    # if len(config_list) == 1:
    #     cfg = OmegaConf.load(config_list[0])
    #     cfg.checkpoint = weight_list[0]
    #     model = load_model(cfg, args)
    # else:
    print(f'Loading {len(config_list)}-model ensemble ...')
    model_list = []
    for ind, each_cfg in enumerate(config_list):
        cfg = OmegaConf.load(each_cfg)
        cfg.checkpoint = weight_list[ind]
        model_list.append(load_model(cfg, args))
    model = Ensemble(model_list) 
    return model


def get_configs_and_weights(group):
    configs, weights = group.configs, group.weights
    assert len(configs) == len(weights)
    return configs, weights


def full_inference(cfg, args):
    cfg.local_rank = cfg.local_rank or 0
    seg_model = None
    lid_model = None
    cls_model = None
    csg_model = None
    cs2_model = None
    # Load in segmentation models 
    if cfg.segmentation_models:
        seg_configs, seg_weights = get_configs_and_weights(cfg.segmentation_models) 
        seg_model = load_single_model_or_ensemble(seg_configs, seg_weights, args)
    # Load in line detection models
    if cfg.lid_models:
        lid_configs, lid_weights = get_configs_and_weights(cfg.lid_models)
        lid_model = load_single_model_or_ensemble(lid_configs, lid_weights, args)
    # Load in classification models
    if cfg.classification_models:
        cls_configs, cls_weights = get_configs_and_weights(cfg.classification_models) 
        cls_model = load_single_model_or_ensemble(cls_configs, cls_weights, args)
    # Load in classification models that also take segmentation as input
    if cfg.classification_models_with_seg:
        csg_configs, csg_weights = get_configs_and_weights(cfg.classification_models_with_seg) 
        csg_model = load_single_model_or_ensemble(csg_configs, csg_weights, args)
    # Load in classification models that take segmentation as input differently
    if cfg.classification_models_with_seg2:
        cs2_configs, cs2_weights = get_configs_and_weights(cfg.classification_models_with_seg2) 
        cs2_model = load_single_model_or_ensemble(cs2_configs, cs2_weights, args)

    with open(args.imgfiles, 'r') as f:
        imgfiles = [l.strip() for l in f.readlines()]

    if args.data_dir:
        imgfiles = [osp.join(args.data_dir, i) for i in imgfiles]

    if args.fast_dev_run:
        imgfiles = imgfiles[:args.fast_dev_run]

    if cfg.local_rank == 0:
        print(f'PREDICT : n={len(imgfiles)}')

    if seg_model:
        seg_dataset = builder.build_dataset(OmegaConf.load(seg_configs[0]), data_info=dict(inputs=[0], labels=[0]), mode='predict')
    if lid_model:
        lid_dataset = builder.build_dataset(OmegaConf.load(lid_configs[0]), data_info=dict(inputs=[0], labels=[0], segs=[0]), mode='predict')
    if cls_model:
        cls_dataset = builder.build_dataset(OmegaConf.load(cls_configs[0]), data_info=dict(inputs=[0], labels=[0]), mode='predict')
    if csg_model:
        csg_dataset = builder.build_dataset(OmegaConf.load(csg_configs[0]), data_info=dict(inputs=[0], labels=[0], segs=[0]), mode='predict')
    if cs2_model:
        cs2_dataset = builder.build_dataset(OmegaConf.load(cs2_configs[0]), data_info=dict(inputs=[0], labels=[0], segs=[0]), mode='predict')

    if cfg.local_rank == 0:
        print(f'Saving predictions to {args.save_preds_file} ...')
        save_dir = osp.dirname(args.save_preds_file)
        if save_dir != '':
            if not osp.exists(save_dir): os.makedirs(save_dir)

    predictions = []
    for imfi in tqdm(imgfiles, total=len(imgfiles)): 
        #1- Load image
        img = cv2.imread(imfi, 0)
        img = np.expand_dims(img, axis=-1)
        img3 = np.repeat(img, 3, axis=-1)
        final_pred = []
        if seg_model:
            #2- Process with dataset (for segmentation)
            seg_img = seg_dataset.process_image_no_mask(img)
            #3- Transform to torch tensor and move to GPU
            seg_img = torch.tensor(seg_img).unsqueeze(0).float().cuda()
            #4- Get segmentation
            with torch.no_grad(): seg_pred = seg_model(seg_img)
            seg_pred = torch.sigmoid(seg_pred)*255
            seg_pred = seg_pred.squeeze(0).cpu().numpy().transpose(1,2,0).astype('uint8')
            del seg_img
            # seg_pred.shape = (1024, 1024, 3)
        if lid_model:
            lid_img = lid_dataset.process_image(img3, seg_pred)
            lid_img = torch.tensor(lid_img).unsqueeze(0).float().cuda() 
            with torch.no_grad(): lid_pred = lid_model(lid_img, act_fn='sigmoid')
            cvc,ett,ngt = (255*lid_pred.cpu().numpy()[0]).astype('int')
            del lid_img
            #Use line detection predictions to adjust predicted segmentation
            seg_pred[...,0][seg_pred[...,0] > cvc] = cvc
            seg_pred[...,1][seg_pred[...,1] > ngt] = ngt
            seg_pred[...,2][seg_pred[...,2] > ett] = ett                        
        if cls_model:
            #5- Process image with dataset (for classification, no seg input)
            cls_img = cls_dataset.process_image(img3)
            cls_img = torch.tensor(cls_img).unsqueeze(0).float().cuda() 
            with torch.no_grad(): cls_pred = cls_model(cls_img, act_fn='sigmoid', flip=cfg.random_flip_within_ensemble)
            cls_pred = cls_pred.cpu().numpy()[0]
            del cls_img
            final_pred += [cls_pred]
        # cls_pred.shape = (11,)
        #6- Jointly process image and segmentation
        if csg_model:
            csg_img = csg_dataset.process_image(img3, seg_pred)
            csg_img = torch.tensor(csg_img).unsqueeze(0).float().cuda()
            with torch.no_grad(): csg_pred = csg_model(csg_img, act_fn='sigmoid', flip=cfg.random_flip_within_ensemble)
            csg_pred = csg_pred.cpu().numpy()[0]
            del csg_img
            final_pred += [csg_pred]
        if cs2_model:
            cs2_img = cs2_dataset.process_image(img, np.expand_dims(seg_pred.max(axis=-1), axis=-1))
            cs2_img = torch.tensor(cs2_img).unsqueeze(0).float().cuda()
            with torch.no_grad(): cs2_pred = cs2_model(cs2_img, act_fn='sigmoid', flip=cfg.random_flip_within_ensemble)
            cs2_pred = cs2_pred.cpu().numpy()[0]
            del cs2_img
            final_pred += [cs2_pred]

        if not args.rank_average:
            final_pred = np.mean(np.asarray(final_pred), axis=0)
        predictions.append(final_pred)
        gc.collect()

    if args.rank_average:
        pred_dict = {}
        # Each element in predictions is list of numpy arrays
        num_preds = len(final_pred)
        for pred_ind in range(num_preds):
            tmp_pred = np.asarray([p[pred_ind] for p in predictions])
            tmp_pred = np.concatenate([np.expand_dims(rankdata(tmp_pred[:,col]), axis=-1) for col in range(tmp_pred.shape[1])], axis=1)
            pred_dict[pred_ind] = tmp_pred
        predictions = pred_dict[0]
        for pred_ind in range(1, num_preds):
            predictions += pred_dict[pred_ind]

    predictions = np.asarray(predictions)
    df = pd.DataFrame(dict(StudyInstanceUID=[i.split('/')[-1].replace('.jpg','') for i in imgfiles]))
    for i in range(predictions.shape[1]):
        df[f'p{i}'] = predictions[:,i]

    df.to_csv(args.save_preds_file, index=False)

