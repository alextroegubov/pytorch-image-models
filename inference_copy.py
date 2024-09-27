#!/usr/bin/env python3
"""
Custom inference for timm library
"""
from argparse import ArgumentParser
import logging
from typing import Optional
from pathlib import Path

from tqdm import tqdm
import numpy as np
import pandas as pd

import torch
import torch.nn.functional as func
from torch.utils.data import DataLoader

from timm.data import create_dataset
from timm.data.transforms_factory import create_transform
from timm.data.transforms import InferenceCropMode, PaddingMode
from timm.models import create_model
from timm.utils import setup_default_logging

torch.backends.cudnn.benchmark = True
_logger = logging.getLogger("inference")


def read_class_map(filename: str):
    """Read class map file, return two dicts and list of names"""
    class_to_idx = {}
    idx_to_class = {}
    target_names = []
    with open(filename, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f.readlines()):
            class_name = line.strip()

            class_to_idx[class_name] = idx
            idx_to_class[idx] = class_name
            target_names.append(class_name)

    return class_to_idx, idx_to_class, target_names


def get_arg_parser() -> ArgumentParser:
    parser = ArgumentParser(description='Pytorch inference')

    parser.add_argument('--data-dir', type=str,
                        help='path to dataset root dir')
    parser.add_argument('--class-map', type=str,
                        help='path to class to idx mapping file')

    parser.add_argument('--model-name', type=str, help='model architecture')
    parser.add_argument('--checkpoint', type=str,
                        help='path to model checkpoint')
    parser.add_argument('--num-classes', type=int, help='Number of classes')

    parser.add_argument('--batch-size', default=64,
                        type=int, help='batch size')
    parser.add_argument('--device', type=str,
                        default='cuda', help="Device to use")
    parser.add_argument('--num-gpu', type=int, default=1,
                        help='Number of GPUS to use')

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
    parser.add_argument('--std', type=float, nargs='+', default=None, metavar='STD',
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
    return parser


def get_dataset(
    root_dir: str,
    input_size: tuple,
    interpolation: str,
    crop_pct: float,
    crop_mode: str,
    padding_mode: str,
):
    """Create dataset with specific timm transform. Structure: root/cls1/some_file.png"""
    dataset = create_dataset(name="", split="", root=root_dir)
    dataset.transform = create_transform(  # type: ignore
        input_size,
        is_training=False,
        interpolation=interpolation,
        crop_pct=crop_pct,
        crop_mode=InferenceCropMode[crop_mode.upper()],
        padding_mode=PaddingMode[padding_mode.upper()],
        crop_border_pixels=0,
        use_prefetcher=False,
        normalize=True,
        # mean=mean,
        # std=std
    )
    return dataset


def process_target_class(
    save_dir: str | Path,
    filenames: list[str],
    target_class: str,
    all_probs: np.ndarray,
    all_labels: list[str],
):
    # set different thresholds
    thresholds = (0.99, 0.97, 0.95, 0.90, 0.85, 0.80, 0.70, 0.1)
    # create folders
    folder_names = [
        Path(save_dir) / f"{target_class}_split" / Path(f"more_{prob:.2f}") for prob in thresholds
    ]
    _ = [folder.mkdir(parents=True, exist_ok=True) for folder in folder_names]

    for idx, file in enumerate(filenames):
        if all_labels[idx] == target_class:
            prob_idx = [i for (i, x) in enumerate(thresholds) if all_probs[idx] > x][0]
            Path(file).rename(folder_names[prob_idx] / Path(file).name)

    # delete empty folders
    _ = [folder.rmdir() for folder in folder_names if len(list(folder.rglob("*"))) == 0]
    res_folders = [folder for folder in folder_names if folder.exists()]

    return res_folders


def process_split_classes(
    save_dir: str | Path,
    filenames: list[str],
    class_names: list[str],
    all_labels: list[str],
):
    # create folders:
    save_path = Path(save_dir)
    folder_names: list[Path] = []
    for cls in class_names:
        save_path.joinpath(cls).mkdir(parents=True, exist_ok=True)
        folder_names.append(save_path / cls)

    for idx, file in enumerate(filenames):
        label = all_labels[idx]
        Path(file).rename(save_path / label / Path(file).name)

    return folder_names
    # FIXME: delete empty folders


def process_split_classes_on_threshold(
    save_dir: str | Path,
    filenames: list[str],
    class_names: list[str],
    all_labels: list[str],
    threshold: float,
    all_probs: np.ndarray,
):
    # create folders:
    save_path = Path(save_dir)
    thre_str = f"_{threshold:.2f}"
    folder_names = [save_path.joinpath(cls + thre_str) for cls in class_names]
    folder_names.append(save_path / "no_label")

    for folder in folder_names:
        folder.mkdir(parents=True, exist_ok=True)

    # split files
    for idx, file in enumerate(filenames):
        label = all_labels[idx]
        prob = all_probs[idx]

        new_parent = label + thre_str if prob >= threshold else "no_label"
        Path(file).rename(save_path / new_parent / Path(file).name)

    # delete empty folders
    _ = [folder.rmdir() for folder in folder_names if len(list(folder.rglob("*"))) == 0]
    res_folders = [folder for folder in folder_names if folder.exists()]

    return res_folders


def inference(
    data_dir: str,
    class_map: str,
    checkpoint: str,
    model_name: str,
    device: str,
    num_classes: int,
    input_size: tuple,
    crop_pct: float,
    crop_mode: str,
    interpolation: str,
    batch_size: int,
    threshold: float,
    show_stats: bool,
    save_dir: Optional[str | Path] = None,
    num_gpu: int = 1,
    save_tsv: bool = False,
    only_target_class: Optional[str] = None,
    split_classes: bool = False,
    split_classes_on_threshold: bool = False,
):

    setup_default_logging()

    if [only_target_class is not None, split_classes, split_classes_on_threshold].count(True) > 1:
        _logger.error(
            "Options <only_target_class>, <split_classes>, <split_classes_on_threshold> "
            "are mutually exclusive"
        )
        return
    # save in data_dir if save_dir is None
    save_dir = Path(save_dir or data_dir)

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
        padding_mode="constant",
    )
    _logger.info(f"Read dataset from {data_dir}: {len(dataset)} samples")

    # different mappings
    _, idx2class, target_names = read_class_map(class_map)
    # create loader
    loader = DataLoader(dataset, batch_size=batch_size, drop_last=False, shuffle=False)
    # set device and create model
    device = torch.device(device)  # type: ignore
    model = create_model(
        model_name, pretrained=True, num_classes=num_classes, checkpoint_path=checkpoint
    ).to(device)
    model.eval()
    n_params = sum([m.numel() for m in model.parameters()])
    _logger.info(f"Model {model_name} created, #params: {n_params}")

    if num_gpu > 1:
        model = torch.nn.DataParallel(model, device_ids=list(range(num_gpu)))
    _logger.info(f"Inference on {num_gpu} {device}, batch_size = {batch_size}")

    # compute class indices and probabilities
    all_filenames: list[str] = dataset.filenames(absolute=True)  # type: ignore
    all_idx = []
    all_probs = []

    with torch.no_grad():
        for batch_data, _ in tqdm(loader, total=len(loader), ncols=50):
            logits = model(batch_data.to(device))
            prob_per_class = func.softmax(logits, dim=-1)
            prob_values, labels_idx = torch.topk(prob_per_class, k=1, dim=-1)

            all_idx.append(labels_idx.squeeze().cpu().numpy())
            all_probs.append(prob_values.squeeze().cpu().numpy())

    # postprocessing
    all_idx = np.concatenate(all_idx, axis=0)
    all_labels = list(map(lambda x: idx2class[x], all_idx))
    all_probs = np.concatenate(all_probs, axis=0)

    folders = []

    if only_target_class is not None and only_target_class in target_names:
        folders = process_target_class(
            save_dir=save_dir,
            target_class=only_target_class,
            filenames=all_filenames,
            all_labels=all_labels,
            all_probs=all_probs,
        )

    elif split_classes:
        folders = process_split_classes(
            save_dir=save_dir,
            filenames=all_filenames,
            all_labels=all_labels,
            class_names=target_names,
        )

    elif split_classes_on_threshold:
        folders = process_split_classes_on_threshold(
            save_dir=save_dir,
            threshold=threshold,
            filenames=all_filenames,
            all_labels=all_labels,
            class_names=target_names,
            all_probs=all_probs,
        )

    if save_tsv:
        filenames = list(map(lambda x: Path(x).name, all_filenames))
        parents = list(map(lambda x: Path(x).parent, all_filenames))
        df = pd.DataFrame.from_dict(
            {
                "dataset_index": np.arange(len(all_filenames), dtype=np.int64),
                "parent": parents,
                "filename": filenames,
                "label": all_labels,
                "probability": all_probs,
            }
        )
        df.to_csv(save_dir / "inf_result.tsv", index=False, sep="\t")

    if show_stats:
        print(f"Sorted {len(dataset)} files in {data_dir}")
        for folder in folders:
            total = len(dataset)
            n_folder = len(list(folder.glob("*")))

            print(
                f"\t--{folder.parent.name}/{folder.name} has {n_folder} images"
                + f"({n_folder / total * 100:.1f}%)"
            )


if __name__ == "__main__":
    args = get_arg_parser().parse_args()
    inference(**dict(args._get_kwargs()))
