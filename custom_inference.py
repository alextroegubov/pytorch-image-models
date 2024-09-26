""" Custom inference for timm library"""
import argparse
import shutil
from pathlib import Path

from tqdm import tqdm

import torch
import torch.nn.functional as func
from torch.utils.data import DataLoader

from timm import create_model
from timm.data.transforms_factory import create_transform
from timm.data.transforms import InferenceCropMode, PaddingMode
from timm.data import create_dataset

import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(description='Pytorch Image Model Inference')

parser.add_argument('--data-dir', metavar='DIR', type=str,
                    help='path to dataset root dir')
parser.add_argument('--class-map', default='', type=str, metavar='FILENAME',
                    help='path to class to idx mapping file (default: "")')
parser.add_argument('--checkpoint', default='', type=str, metavar='PATH',
                    help='path to model checkpoint (default: none)')

parser.add_argument('--model-name', metavar='MODEL', default='resnet50',
                    help='model architecture (default: resnet50)')
parser.add_argument('--device', default='cuda', type=str,
                    help="Device (accelerator) to use.")
parser.add_argument('--num-classes', type=int, default=None,
                    help='Number classes in dataset')

parser.add_argument('--input-size', default=None, nargs=3, type=int,
                    metavar='N N N', help='Input image dimensions (d h w, e.g. \
                        --input-size 3 224 224), uses model default if empty')
parser.add_argument('--crop-pct', default=1.0, type=float,
                    metavar='N', help='Input image center crop pct')
parser.add_argument('--crop-mode', default='center', type=str,
                    choices=['squash', 'border', 'center'],
                    help='Input image crop mode (squash, border, center). Default is center.')
parser.add_argument('--interpolation', default='bilinear', type=str,
                    help='Interpolation for transform')

parser.add_argument('--batch-size', default=64, type=int,
                    metavar='N', help='batch size (default: 64)')
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


def get_dataset(root_dir: str, input_size: tuple, interpolation: str,
                crop_pct: float, crop_mode: str, padding_mode: str):
    """ Create dataset with timm transform"""
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


def create_new_folders_multi_class(root_dir: str, target_names: list[str],
                                   threshold: float, single_folder: bool = False):
    """ Create new folders and return folders list"""
    more_thre: str = f'_more_{threshold:.2f}/'
    less_thre: str = f'_less_{threshold:.2f}/'

    folders_lst: list[Path] = []

    for cls in target_names:
        cls_path = Path(root_dir) / (cls + more_thre)
        cls_path.mkdir(parents=True, exist_ok=True)
        folders_lst.append(cls_path)

        if not single_folder:
            cls_path = Path(root_dir) / (cls + less_thre)
            cls_path.mkdir(parents=True, exist_ok=True)
            folders_lst.append(cls_path)

    if single_folder:
        cls_path = Path(root_dir) / 'no_label'
        cls_path.mkdir(parents=True, exist_ok=True)
        folders_lst.append(cls_path)
    return more_thre, less_thre, folders_lst


def create_new_folders_single_class(root_dir: str, cls_name: str, probs: list[float]):
    """ Create new folders """
    root_path = Path(root_dir)

    folders_lst: list[Path] = []

    for prob in probs:
        conf = f'_more_{prob:.2f}'
        root_path.joinpath(cls_name + conf).mkdir(parents=True, exist_ok=True)
        folders_lst.append(root_path.joinpath(cls_name + conf))

    return folders_lst


def main(data_dir: str, class_map: str, checkpoint: str, model_name, device,
         num_classes: int, input_size: tuple, crop_pct: float, crop_mode: str,
         interpolation: str, batch_size: int, threshold: float, move_files: bool,
         show_stats: bool, single_folder: bool, only_this_class: str | None = None) -> None:
    """ Main function"""

    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.benchmark = True

    device = torch.device(device)

    dataset = get_dataset(
        root_dir=data_dir,
        input_size=input_size,
        interpolation=interpolation,
        crop_pct=crop_pct,
        crop_mode=crop_mode,
        padding_mode='constant',
    )
    _, idx2class, target_names = read_class_map(class_map)

    loader = DataLoader(dataset, batch_size=batch_size, drop_last=False, shuffle=False)

    model = create_model(model_name=model_name, pretrained=True,
                         num_classes=num_classes, checkpoint_path=checkpoint).to(device)
    model.eval()

    probs = [0.99, 0.97, 0.95, 0.90, 0.85, 0.80, 0.70]

    more_thre, less_thre = None, None

    if only_this_class:
        folders_lst = create_new_folders_single_class(
            data_dir, only_this_class, probs)
    else:
        more_thre, less_thre, folders_lst = create_new_folders_multi_class(
            data_dir, target_names, threshold, single_folder)

    with torch.no_grad():
        for (batch_idx, (batch_data, _)) in tqdm(enumerate(loader), total=len(loader), ncols=50):

            model_output = model(batch_data.to(device))
            print(model_output.shape)
            prob_per_class = func.softmax(model_output, dim=-1)
            prob_values, labels = torch.topk(prob_per_class, k=1, dim=-1)

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

    if show_stats:
        print(f"Sorted {len(dataset)} files in {data_dir}")
        for folder in folders_lst:
            total = len(dataset)
            n_folder = len(list(folder.glob('*')))

            print(f"\t--{folder.parent / folder.name} has {n_folder} images" +
                  f"({n_folder / total * 100:.1f}%)")


if __name__ == '__main__':
    args = parser.parse_args()
    main(**dict(args._get_kwargs()))
