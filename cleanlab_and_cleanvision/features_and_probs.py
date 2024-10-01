#!/usr/bin/env python3
"""
Calculating features and probabilities for use in cleanlab inspection
"""
import logging
from typing import Optional
from pathlib import Path

from typing import Sequence, Iterator

from tqdm import tqdm
import numpy as np

import torch
import torch.nn.functional as func
from torch.utils.data import DataLoader, Sampler

from timm.data import create_dataset
from timm.data.transforms_factory import create_transform
from timm.data.transforms import InferenceCropMode, PaddingMode
from timm.models import create_model
from timm.utils import setup_default_logging

from utils import read_filenames_from_file

torch.backends.cudnn.benchmark = True
_logger = logging.getLogger("Features and Probabilities")


class SubsetSequentialSampler(Sampler[int]):
    r"""Samples elements sequentially from a given list of indices, without replacement.

    Args:
        indices (sequence): a sequence of indices
    """

    indices: Sequence[int]

    def __init__(self, indices: Sequence[int]) -> None:
        self.indices = indices

    def __iter__(self) -> Iterator[int]:
        for i in self.indices:
            yield i

    def __len__(self) -> int:
        return len(self.indices)


def get_indices_for_sampler(timm_dataset, filename: str | Path):
    target_filenames_lst = read_filenames_from_file(filename)
    all_filenames_lst: list[str] = timm_dataset.filenames(basename=True)
    # indices should be arranged in the same order as files in target_filenames_lst
    sampler_idx = [all_filenames_lst.index(name) for name in target_filenames_lst]
    return sampler_idx


def compute_features(model, device, dataloader):
    """
    Changes head.fc to nn.Identity and computes features

    Args:
        model: pytorch model
        device: device for computation
        dataloader: pytorch DataLoader
    Returns:
        np.ndarray with shape (n_samples, n_features)
    """
    model = model.to(device)
    # FIXME: if isinstance(model, ...):
    model.head.fc = torch.nn.Identity()
    model.eval()

    all_features = []
    with torch.no_grad():
        for batch_data, _ in tqdm(dataloader, "Features", total=len(dataloader), ncols=50):
            features = model(batch_data.to(device))
            all_features.append(features.squeeze().cpu().numpy())

    all_features = np.vstack(all_features)
    return all_features


def compute_probabilities(model, device, dataloader):
    """
    Args:
        model: pytorch model that returns logits
        device: device for computation
        dataloader: pytorch DataLoader
    Returns:
        np.ndarray with shape (n_samples, n_classes)
    """
    model = model.to(device)
    model.eval()

    all_probs = []
    with torch.no_grad():
        for batch_data, _ in tqdm(dataloader, "Probabilities", total=len(dataloader), ncols=50):
            logits = model(batch_data.to(device))
            probs = func.softmax(logits, dim=-1)
            all_probs.append(probs.squeeze().cpu().numpy())

    all_probs = np.vstack(all_probs)
    return all_probs


def features_and_probabilities(
    data_dir: str,
    model_name: str,
    num_classes: int,
    checkpoint: str,
    device: str,
    input_size: tuple[int, int, int],
    crop_pct: float,
    crop_mode: InferenceCropMode,
    padding_mode: PaddingMode,
    interpolation: str,
    batch_size: int,
    file_with_samples_filenames: Optional[str | Path] = None,
    save_features: bool = True,
    save_probs: bool = True,
    save_dir: Optional[str | Path] = None,
):
    """Computes features and probabilities for a dataset

    Args:
        data_dir: root directory with images
        model_name: timm model name
        num_classes: number of classes in the model
        checkpoint: model weights
        device: device for computation
        input_size: image input size (C, H, W), e.g. (3, 224, 224)
        crop_pct: crop percentage, 0.0-1.0
        crop_mode: one of the possible inference crop modes
        padding_mode: one of the possible padding mode
        interpolation: linear, bilinear, bicubic
        batch_size: size of the batch
        file_with_samples_filenames: allows to get subset of images in data_dir
        save_features: wheater to save features
        save_probs: wheather to save probabilities
        save_dir: where to save features and probs
    """

    setup_default_logging()

    # save in data_dir if save_dir is None
    save_dir = Path(save_dir or data_dir)

    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.benchmark = True

    # create dataset with transforms: structure: root/cls1/file.png
    dataset = create_dataset(name="", split="", root=data_dir)
    dataset.transform = create_transform(  # type: ignore
        input_size,
        is_training=False,
        interpolation=interpolation,
        crop_pct=crop_pct,
        crop_mode=crop_mode,
        padding_mode=padding_mode,
        crop_border_pixels=0,
        use_prefetcher=False,
        normalize=True,
    )
    _logger.info("Read dataset from %s: %s samples", data_dir, len(dataset))

    # create loader with sampler
    sampler = None
    if file_with_samples_filenames is not None:
        sampler_idx = get_indices_for_sampler(dataset, file_with_samples_filenames)
        sampler = SubsetSequentialSampler(sampler_idx)
    loader = DataLoader(
        dataset, batch_size=batch_size, sampler=sampler, drop_last=False, shuffle=False
    )
    _logger.info(
        "Create dataloader: %s",
        (
            "using all files"
            if file_with_samples_filenames is None
            else f"using only subset of {len(sampler_idx)} samples"
        ),
    )

    # set device and create model
    device = torch.device(device)  # type: ignore

    if save_features:
        model = create_model(
            model_name, pretrained=True, num_classes=num_classes, checkpoint_path=checkpoint
        )
        features = compute_features(model, device, loader)
        np.save(save_dir / "features", features)

    if save_probs:
        model = create_model(
            model_name, pretrained=True, num_classes=num_classes, checkpoint_path=checkpoint
        )
        probs = compute_probabilities(model, device, loader)
        np.save(save_dir / "probabilities", probs)
