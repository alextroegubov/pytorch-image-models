import argparse
import time
from pathlib import Path
import json

import torch
import torch.nn.functional as func

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from timm import create_model
from timm.data import create_dataset
from timm.data import create_loader
from timm.utils import ParseKwargs


from sklearn.metrics import classification_report


parser = argparse.ArgumentParser(description='Pytorch Image Model Validation')
parser.add_argument('--data-dir', metavar='DIR',
                    help='path to dataset root dir')
parser.add_argument('--split', metavar='SPLIT', default='val',
                    help='dataset split (val or train)')
parser.add_argument('--model', metavar='MODEL', default='resnet50',
                    help='model architecture (default: resnet50)')
parser.add_argument('--batch-size', default=256, type=int,
                    metavar='N', help='batch size (default: 256)')

parser.add_argument('--input-size', default=None, nargs=3, type=int,
                    metavar='N N N', help='Input image dimensions (d h w, e.g. \
                    --input-size 3 224 224), uses model default if empty')
parser.add_argument('--crop-pct', default=1.0, type=float,
                    metavar='N', help='Input image center crop pct')
parser.add_argument('--crop-mode', default='center', type=str,
                    metavar='N', help='Input image crop mode (squash, border, center). \
                    Default is center.')
parser.add_argument('--interpolation', default='bilinear', type=str,
                    help='Interpolation')

parser.add_argument('--num-classes', type=int, default=None,
                    help='Number classes in dataset')
parser.add_argument('--class-map', default='', type=str, metavar='FILENAME',
                    help='path to class to idx mapping file (default: "")')
parser.add_argument('--batch-log-step', default=5, type=int,
                    metavar='N', help='batch logging frequency (default: 10)')
parser.add_argument('--checkpoint', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')

parser.add_argument('--device', default='cuda', type=str,
                    help="Device (accelerator) to use.")
parser.add_argument('--model-kwargs', nargs='*', default={}, action=ParseKwargs)


parser.add_argument('--results-dir', type=str, default=None,
                    help='folder for output results')
parser.add_argument('--save-text', action='store_true', default=False,
                    help='Save mispredicted images as text dataframe')
parser.add_argument('--save-pics', action='store_true', default=False,
                    help='Save mistpredicted images as images')
parser.add_argument('--save-report', action='store_true', default=False,
                    help='Save classification report in json format')


def save_mispreds_text(dataset, true_labels, pred_labels, pred_prob, idx2class, folder):
    mispred_idx = np.arange(len(dataset))[true_labels != pred_labels]

    df_data = []
    columns = ['filename', 'true_class', 'pred_class', 'prob']
    for idx in mispred_idx:
        df_data.append([
            dataset.filename(idx),
            idx2class[true_labels[idx]],
            idx2class[pred_labels[idx]],
            pred_prob[idx]
        ])

    df = pd.DataFrame(df_data, columns=columns)
    df.to_csv(f'{folder}/mispred.tsv', sep='\t', header=True, index=False)


def save_mispreds_pics(dataset, true_labels, pred_labels, pred_prob, idx2class, folder):
    mispred_idx = np.arange(len(dataset))[true_labels != pred_labels]

    for idx in mispred_idx:
        true_label = idx2class[true_labels[idx]]
        pred_label = idx2class[pred_labels[idx]]
        prob = pred_prob[idx]
        plt.figure(figsize=(8, 8))
        plt.title(f'True: {true_label}, Pred: {pred_label} ({prob*100:.2f}%)')
        plt.imshow(dataset[idx][0].transpose(1, 2, 0))
        filename = Path(dataset.filename(idx)).name

        pics_path = Path(folder) / 'pics'
        Path(pics_path).mkdir(parents=True, exist_ok=True)
        plt.savefig(str(pics_path / filename))
        plt.close()


def save_classification_report(true_labels, pred_labels, target_names, folder):
    report = classification_report(
        true_labels,
        pred_labels,
        target_names=target_names,
        digits=3,
        output_dict=True
    )
    with open(f'{folder}/cls_report.json', 'w', encoding='utf-8') as file:
        json.dump(report, file)


def main() -> None:
    args = parser.parse_args()

    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.benchmark = True

    device = torch.device(args.device)

    dataset = create_dataset(
        name='',
        root=args.data_dir,
        split=args.split,
        class_map=args.class_map
    )

    idx2class = {value: key for (key, value) in dataset.reader.class_to_idx.items()}  # type: ignore
    target_names = [idx2class[i] for i in range(len(idx2class))]

    data_loader = create_loader(
        dataset,  # type: ignore
        batch_size=args.batch_size,
        use_prefetcher=True,
        is_training=False,
        device=device,
        input_size=tuple(args.input_size),
        crop_pct=args.crop_pct,
        crop_mode=args.crop_mode,
        interpolation=args.interpolation
    )

    model = create_model(
        model_name=args.model,
        pretrained=True,
        num_classes=args.num_classes,
        checkpoint_path=args.checkpoint,
        **args.model_kwargs
    ).to(device)
    model.eval()

    all_true_labels = []
    all_pred_labels = []
    all_pred_probs = []

    start_time = time.time()
    batch_time = time.time()

    with torch.no_grad():
        for (batch_idx, (batch_data, batch_labels)) in enumerate(data_loader):

            model_output = model(batch_data)  # (batch_sz x num_cls)
            prob_per_class = func.softmax(model_output, dim=-1)  # (batch_sz x num_cls)
            prob_values, labels = torch.topk(prob_per_class, k=1, dim=-1)

            all_true_labels.append(batch_labels.cpu().numpy().squeeze())
            all_pred_labels.append(labels.cpu().numpy().squeeze())
            all_pred_probs.append(prob_values.cpu().numpy().squeeze())

            if batch_idx % args.batch_log_step == 0:
                print(f'Batch [{batch_idx}/{len(data_loader)}] Time {time.time() - batch_time:.3f}')
                batch_time = time.time()

        all_true_labels = np.concatenate(all_true_labels)
        all_pred_labels = np.concatenate(all_pred_labels)
        all_pred_probs = np.concatenate(all_pred_probs)

    print(f'Inference finished in {time.time() - start_time:.2f}')

    report = classification_report(
        all_true_labels,
        all_pred_labels,
        target_names=target_names,
        digits=3
    )

    print(report)

    Path(args.results_dir).mkdir(parents=True, exist_ok=True)

    if args.save_text and args.results_dir is not None:
        save_mispreds_text(dataset, all_true_labels, all_pred_labels,
                           all_pred_probs, idx2class, args.results_dir)

    if args.save_pics and args.results_dir is not None:
        save_mispreds_pics(dataset, all_true_labels, all_pred_labels,
                           all_pred_probs, idx2class, args.results_dir)

    if args.save_report and args.results_dir is not None:
        save_classification_report(all_true_labels, all_pred_labels,
                                   target_names, args.results_dir)


if __name__ == '__main__':
    main()
