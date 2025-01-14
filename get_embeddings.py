""" Getting embeddings"""
import argparse
from pathlib import Path
from tqdm import tqdm
import numpy as np
import pandas as pd

import torch
from torch.utils.data import DataLoader

from timm.data.transforms_factory import create_transform
from timm.data import create_dataset

from timm.models import ConvNeXt
import torch.nn as nn

DATA_DIR = "/home/user/Documents/data/vehicle_classification_raw_284k_8_cls_after_cv_jpg"
CHECKPOINT = "/home/user/Documents/weights/classification/vehicle_type/convnextv2_pico/model.pt"
SAVE_DIR = "/home/user/Documents/repos/pytorch-image-models/output/dataset_features"

MODEL_NAME = "convnextv2_pico.fcmae_ft_in1k"
DEVICE = 'cuda'

CROP_PCT = 1.0
CROP_MODE = 'center'
INTERPOLATION = 'bicubic'
INPUT_SIZE = (3, 288, 288)

BATCH_SIZE = 256
NUM_CLASSES = 8


class TimmNet(ConvNeXt):
    def __init__(self, model):
        super(TimmNet, self).__init__()
        self.model = model
        self.model.head.add_module("softmax", nn.Softmax(dim=1))
        self.model.head.softmax.requires_grad_(False)

    def forward(self, x):
        x = self.model(x)
        # x = self.softmax(x)
        x = self.model.head.softmax(x)
        return x


def create_argparser():
    parser = argparse.ArgumentParser(description='Pytorch Image Model Inference')

    parser.add_argument('--data-dir', default=DATA_DIR, type=str, help='path to dataset root dir')
    parser.add_argument('--checkpoint', default=CHECKPOINT, type=str, help='path to model checkpoint (default: none)')
    parser.add_argument('--save-dir', default=SAVE_DIR, type=str, help='path to save results')
    parser.add_argument('--model-name', default=MODEL_NAME, type=str, help='model architecture')
    parser.add_argument('--device', default=DEVICE, type=str, help="Device (accelerator) to use.")
    parser.add_argument('--input-size', default=INPUT_SIZE, nargs=3, type=int, help='Input image dimensions (c h w)')
    parser.add_argument('--crop-pct', default=CROP_PCT, type=float, help='Input image center crop pct')
    parser.add_argument('--crop-mode', default=CROP_MODE, type=str, help='Input image crop mode (squash, border, center)')
    parser.add_argument('--interpolation', default=INTERPOLATION, type=str, help='Interpolation for transform')
    parser.add_argument('--batch-size', default=BATCH_SIZE, type=int, help='batch size')
    parser.add_argument('--num-classes', default=NUM_CLASSES, type=int, help='Number of classes')

    return parser


def get_dataset(root_dir: str, input_size: tuple, interpolation: str,
                crop_pct: float, crop_mode: str):
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
        crop_mode=crop_mode,
        crop_border_pixels=0,
        use_prefetcher=False,
        normalize=True,
        # mean=mean,
        # std=std
    )
    return dataset

# def compute_and_save_probs_per_class(model, device, loader):
#     all_probs_per_class = []

#     with torch.no_grad():
#         for batch_data, _ in tqdm(loader, total=len(loader), ncols=50):
#             logits = model(batch_data.to(device))
#             probs_per_class = func.softmax(logits, dim=-1)
#             all_probs_per_class.append(probs_per_class.cpu().numpy())

#     all_probs_per_class = np.vstack(all_probs_per_class)

#     return all_probs_per_class


def main(data_dir: str, checkpoint: str, model_name, device,
         num_classes: int, input_size: tuple, crop_pct: float, crop_mode: str,
         interpolation: str, batch_size: int, save_dir: str) -> None:
    """ Main function"""

    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.benchmark = True

    device = torch.device(device)
    Path(save_dir).mkdir(parents=True, exist_ok=True)

    dataset = get_dataset(
        root_dir=data_dir,
        input_size=input_size,
        interpolation=interpolation,
        crop_pct=crop_pct,
        crop_mode=crop_mode,
    )

    loader = DataLoader(dataset, batch_size=batch_size, drop_last=False, shuffle=False)

    df_lst = []
    for i in range(len(dataset)):
        df_lst.append([i, dataset.filename(i, absolute=True), dataset.filename(i, basename=True)])

    df = pd.DataFrame(df_lst, columns=['ds_index', 'filepath', 'filename'])

    df.to_csv(Path(save_dir) / "idx_name.tsv", sep='\t', index=False, header=True, encoding='utf-8')

    model_emb = torch.load(checkpoint).to(device)
    model_emb.model.head.fc = torch.nn.Identity()
    model_emb.model.head.softmax = torch.nn.Identity()
    model_emb.eval()

    all_embeddings = []
    all_idx = []

    with torch.no_grad():
        for (batch_idx, (batch_data, _)) in tqdm(enumerate(loader), total=len(loader), ncols=50):
            embeds = model_emb(batch_data.to(device)).cpu().numpy()

            all_embeddings.append(embeds)
            all_idx.append(np.arange(0, batch_data.shape[0]) + batch_idx * batch_size)

    all_embeddings = np.vstack(all_embeddings)
    all_idx = np.hstack(all_idx)

    np.save(Path(save_dir) / "features.npy", all_embeddings)
    np.save(Path(save_dir) / "all_idx.npy", all_idx)

    print(all_embeddings.shape, all_idx.shape, len(loader), len(dataset))


if __name__ == '__main__':
    args = create_argparser().parse_args()
    main(**dict(args._get_kwargs()))
