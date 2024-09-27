#!/usr/bin/env python3
"""
Custom inference for timm library
"""
from argparse import ArgumentParser
import json
import logging
import os
import time
from contextlib import suppress
from functools import partial

import numpy as np
import pandas as pd
import torch

from timm.data import create_dataset, create_loader, ImageNetInfo, infer_imagenet_subset
from timm.layers import apply_test_time_pool
from timm.models import create_model
from timm.data.transforms_factory import create_transform
from timm.utils import AverageMeter, setup_default_logging, set_jit_fuser, ParseKwargs


import shutil
from pathlib import Path

from tqdm import tqdm

import torch
import torch.nn.functional as func
from torch.utils.data import DataLoader

import matplotlib.pyplot as plt


_FMT_EXT = {
    'json': '.json',
    'json-record': '.json',
    'json-split': '.json',
    'parquet': '.parquet',
    'csv': '.csv',
}

torch.backends.cudnn.benchmark = True
_logger = logging.getLogger('inference')


def read_class_map(filename: str):
    """ Read class map file, return two dicts and list of names"""
    class_to_idx = {}
    idx_to_class = {}
    target_names = []
    with open(filename, 'r', encoding='utf-8') as f:
        for (idx, line) in enumerate(f.readlines()):
            class_name = line.strip()

            class_to_idx[class_name] = idx
            idx_to_class[idx] = class_name
            target_names.append(class_name)

    return class_to_idx, idx_to_class, target_names


def get_arg_parser() -> ArgumentParser:
    parser = ArgumentParser(description='Pytorch inference')

    parser.add_argument('--data-dir', type=str, help='path to dataset root dir')
    parser.add_argument('--class-map', type=str, help='path to class to idx mapping file')

    parser.add_argument('--model-name', type=str, help='model architecture')
    parser.add_argument('--checkpoint', type=str, help='path to model checkpoint')
    parser.add_argument('--num-classes', type=int, help='Number of classes')


    parser.add_argument('--batch-size', default=64, type=int, help='batch size')
    parser.add_argument('--device', type=str, default='cuda', help="Device to use")
    parser.add_argument('--num-gpu', type=int, default=1, help='Number of GPUS to use')


    parser.add_argument('--input-size', default=None, nargs=3, type=int,
                        help='Input image dims (C H W), model default if empty')
    parser.add_argument('--crop-pct', default=1.0, type=float,
                        help='Input image center crop percent')
    parser.add_argument('--crop-mode', default='center', type=str, 
                        choices=['center', 'squash', 'border'],
                        help='Input image crop mode (squash, border, center)')
    parser.add_argument('--pad-mode', default='reflect', type=str,
                        choices=['reflect', 'constant', 'edge', 'symmetric'])
    parser.add_argument('--interpolation', default='bilinear', type=str,
                        help='Interpolation is transform')
    parser.add_argument('--mean', type=float, nargs='+', default=None, metavar='MEAN',
                        help='Override mean pixel value of dataset')
    parser.add_argument('--std', type=float,  nargs='+', default=None, metavar='STD',
                        help='Override std deviation of of dataset')


    parser.add_argument('--threshold', default='0.75', type=float,
                        help='Threshold for classification (default 0.75)')


    parser.add_argument('--move-files', action='store_true', default=False,
                        help='Move files or copy them to folder')
    parser.add_argument('--show-stats', action='store_true', default=False,
                        help='Print statistics at the end')
    parser.add_argument('--single-folder', action='store_true', default=False,
                        help='Create only one <more> threshold folder')
    parser.add_argument('--only-this-class', type=str, default=None,
                        help='Sort only this class images')



    parser.add_argument('-j', '--workers', default=2, type=int, metavar='N',
                        help='number of data loading workers (default: 2)')

    parser.add_argument('--use-train-size', action='store_true', default=False,
                        help='force use of train input size, even when test size is specified in pretrained cfg')


    parser.add_argument('--log-freq', default=10, type=int,
                        metavar='N', help='batch logging frequency (default: 10)')

    parser.add_argument('--results-dir', type=str, default=None,
                        help='folder for output results')
    parser.add_argument('--results-file', type=str, default=None,
                        help='results filename (relative to results-dir)')
    parser.add_argument('--results-format', type=str, nargs='+', default=['csv'],
                        help='results format (one of "csv", "json", "json-split", "parquet")')
    parser.add_argument('--results-separate-col', action='store_true', default=False,
                        help='separate output columns per result index.')
    parser.add_argument('--topk', default=1, type=int,
                        metavar='N', help='Top-k to output to CSV')
    parser.add_argument('--fullname', action='store_true', default=False,
                        help='use full sample name in output (not just basename).')
    parser.add_argument('--filename-col', type=str, default='filename',
                        help='name for filename / sample name column')
    parser.add_argument('--index-col', type=str, default='index',
                        help='name for output indices column(s)')
    parser.add_argument('--label-col', type=str, default='label',
                        help='name for output indices column(s)')
    parser.add_argument('--output-col', type=str, default=None,
                        help='name for logit/probs output column(s)')
    parser.add_argument('--output-type', type=str, default='prob',
                        help='output type colum ("prob" for probabilities, "logit" for raw logits)')
    parser.add_argument('--label-type', type=str, default='description',
                        help='type of label to output, one of  "none", "name", "description", "detailed"')
    parser.add_argument('--include-index', action='store_true', default=False,
                        help='include the class index in results')
    parser.add_argument('--exclude-output', action='store_true', default=False,
                        help='exclude logits/probs from results, just indices. topk must be set !=0.')


def get_dataset(root_dir: str, input_size: tuple, interpolation: str,
                crop_pct: float, crop_mode: str, padding_mode: str):
    """ Create dataset with specific timm transform. Structure: root/cls1/some_file.png"""
    dataset = create_dataset(
        name='',
        split='',
        root=root_dir
    )
    dataset.transform = create_transform(  # type: ignore
        input_size,
        is_training=False,
        interpolation=interpolation,
        crop_pct=crop_pct,
        crop_mode=InferenceCropMode[crop_mode.upper()],
        padding_mode=PaddingMode[padding_mode.upper()],
        crop_border_pixels=0,
        use_prefetcher=False,
        normalize=True
        # mean=mean,
        # std=std
    )
    return dataset


def inference(data_dir: str, class_map: str, checkpoint: str, model_name: str, device: str,
              num_classes: int, input_size: tuple, crop_pct: float, crop_mode: str,
              interpolation: str, batch_size: int, threshold: float, move_files: bool,
              show_stats: bool, single_folder: bool, only_this_class: str | None = None,
              num_gpu: int = 1):
    setup_default_logging()

    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.benchmark = True

    # create dataset with transforms
    dataset = get_dataset(
        root_dir=data_dir,
        input_size=input_size,
        interpolation=interpolation,
        crop_pct=crop_pct,
        crop_mode=crop_mode,
        padding_mode='constant',
    )
    _logger.info(f'Read dataset from {data_dir}: {len(dataset)} samples')

    # different mappings
    class2idx, idx2class, target_names = read_class_map(class_map)
    # create loader
    loader = DataLoader(dataset, batch_size=batch_size, drop_last=False, shuffle=False)
    # set device and create model
    device = torch.device(device) # type: ignore
    model = create_model(model_name, pretrained=True, num_classes=num_classes,
                         checkpoint_path=checkpoint).to(device)
    model.eval()
    n_params = sum([m.numel() for m in model.parameters()])
    _logger.info(f'Model {model_name} created, #params: {n_params}')

    if num_gpu > 1:
        model = torch.nn.DataParallel(model, device_ids=list(range(num_gpu)))

    workers = 1 if 'tfds' in args.dataset or 'wds' in args.dataset else args.workers


    with torch.no_grad():
        for (batch_idx, (batch_data, _)) in tqdm(enumerate(loader), total=len(loader), ncols=50):
            logits = model(batch_data.to(device))
            prob_per_class = func.softmax(logits, dim=-1)
            prob_values, labels = torch.topk(prob_per_class, k=1, dim=-1)

            # first create pandas data frame, then postprocessing
            # (filename, class_label, prob, logit)
            # save embeddings, save logits

            for local_idx in range(batch_data.shape[0]):
                pred_prob = prob_values[local_idx].cpu().item()
                class_label = idx2class[labels[local_idx].cpu().item()]
                img_idx = batch_idx * batch_size + local_idx

                src_filename = Path(data_dir) / dataset.filename(img_idx)  # type: ignore

                if only_this_class is not None and class_label == only_this_class:
                    prob_idx = [i for (i, x) in enumerate(probs) if pred_prob > x][0]
                    new_folder = Path(folders_lst[prob_idx])
                elif only_this_class is not None:
                    continue
                else:
                    if pred_prob >= threshold:
                        new_folder = Path(class_label + more_thre)
                    elif single_folder:
                        new_folder = Path('no_label')
                    else:
                        new_folder = Path(class_label + less_thre)

                dst_filename = Path(data_dir) / new_folder / src_filename.name

                if move_files:
                    src_filename.rename(dst_filename)
                else:
                    shutil.copy(src_filename, dst_filename)



    top_k = min(args.topk, args.num_classes)
    batch_time = AverageMeter()
    end = time.time()
    all_indices = []
    all_labels = []
    all_outputs = []
    use_probs = args.output_type == 'prob'
    with torch.no_grad():
        for batch_idx, (input, _) in enumerate(loader):

            with amp_autocast():
                output = model(input)

            if use_probs:
                output = output.softmax(-1)

            if top_k:
                output, indices = output.topk(top_k)
                np_indices = indices.cpu().numpy()
                if args.include_index:
                    all_indices.append(np_indices)
                if to_label is not None:
                    np_labels = to_label(np_indices)
                    all_labels.append(np_labels)

            all_outputs.append(output.cpu().numpy())

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if batch_idx % args.log_freq == 0:
                _logger.info('Predict: [{0}/{1}] Time {batch_time.val:.3f} ({batch_time.avg:.3f})'.format(
                    batch_idx, len(loader), batch_time=batch_time))

    all_indices = np.concatenate(all_indices, axis=0) if all_indices else None
    all_labels = np.concatenate(all_labels, axis=0) if all_labels else None
    all_outputs = np.concatenate(all_outputs, axis=0).astype(np.float32)
    filenames = loader.dataset.filenames(basename=not args.fullname)

    output_col = args.output_col or ('prob' if use_probs else 'logit')
    data_dict = {args.filename_col: filenames}
    if args.results_separate_col and all_outputs.shape[-1] > 1:
        if all_indices is not None:
            for i in range(all_indices.shape[-1]):
                data_dict[f'{args.index_col}_{i}'] = all_indices[:, i]
        if all_labels is not None:
            for i in range(all_labels.shape[-1]):
                data_dict[f'{args.label_col}_{i}'] = all_labels[:, i]
        for i in range(all_outputs.shape[-1]):
            data_dict[f'{output_col}_{i}'] = all_outputs[:, i]
    else:
        if all_indices is not None:
            if all_indices.shape[-1] == 1:
                all_indices = all_indices.squeeze(-1)
            data_dict[args.index_col] = list(all_indices)
        if all_labels is not None:
            if all_labels.shape[-1] == 1:
                all_labels = all_labels.squeeze(-1)
            data_dict[args.label_col] = list(all_labels)
        if all_outputs.shape[-1] == 1:
            all_outputs = all_outputs.squeeze(-1)
        data_dict[output_col] = list(all_outputs)

    df = pd.DataFrame(data=data_dict)

    results_filename = args.results_file
    if results_filename:
        filename_no_ext, ext = os.path.splitext(results_filename)
        if ext and ext in _FMT_EXT.values():
            # if filename provided with one of expected ext,
            # remove it as it will be added back
            results_filename = filename_no_ext
    else:
        # base default filename on model name + img-size
        img_size = data_config["input_size"][1]
        results_filename = f'{args.model}-{img_size}'

    if args.results_dir:
        results_filename = os.path.join(args.results_dir, results_filename)

    for fmt in args.results_format:
        save_results(df, results_filename, fmt)

    print(f'--result')
    print(df.set_index(args.filename_col).to_json(orient='index', indent=4))


def save_results(df, results_filename, results_format='csv', filename_col='filename'):
    results_filename += _FMT_EXT[results_format]
    if results_format == 'parquet':
        df.set_index(filename_col).to_parquet(results_filename)
    elif results_format == 'json':
        df.set_index(filename_col).to_json(results_filename, indent=4, orient='index')
    elif results_format == 'json-records':
        df.to_json(results_filename, lines=True, orient='records')
    elif results_format == 'json-split':
        df.to_json(results_filename, indent=4, orient='split', index=False)
    else:
        df.to_csv(results_filename, index=False)


if __name__ == '__main__':
    main()
