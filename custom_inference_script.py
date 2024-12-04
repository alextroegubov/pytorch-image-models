from pathlib import Path
import custom_inference

CLASS_MAP = "/home/user/Documents/data/plates/license_plates_readability/class_map.txt"
#CHECKPOINT = "/home/user/Documents/repos/pytorch-image-models/output/convnext_pico.d1_in1k-convnext_pico_color_aug_updata_v4/model_best.pth.tar"

CHECKPOINT = "/home/user/Documents/repos/pytorch-image-models/output/convnext_pico.d1_in1k-50_100_pretrained_pad_around_3_cls-v7/model_best.pth.tar"
#CHECKPOINT = "/home/user/Documents/repos/pytorch-image-models/output/convnext_pico.d1_in1k-50_100_pretrained_pad_around_2_cls-v1/model_best.pth.tar"

MODEL_NAME = "convnext_pico.d1_in1k"
DEVICE = 'cuda'
NUM_CLASSES = 3

CROP_PCT = 1.0
CROP_MODE = 'pad_around'
PAD_MODE = 'constant'
INTERPOLATION = 'bilinear'
INPUT_SIZE = (3, 50, 100)

BATCH_SIZE = 1000
THRESHOLD = 0.97

ROOT = "/home/user/Documents/data/plates/test_data_plates"

#SUB_DIRS = ['good', 'bad', 'border']
SUB_DIRS = ['']

for cls_dir in SUB_DIRS:
    custom_inference.inference(
        data_dir=str(Path(ROOT) / cls_dir),
        class_map=CLASS_MAP,
        checkpoint=CHECKPOINT,
        model_name=MODEL_NAME,
        device=DEVICE,
        num_classes=NUM_CLASSES,
        crop_pct=CROP_PCT,
        crop_mode=CROP_MODE,
        pad_mode=PAD_MODE,
        interpolation=INTERPOLATION,
        batch_size=BATCH_SIZE,
        input_size=INPUT_SIZE,
        threshold=THRESHOLD,
        only_target_class=None,
        show_stats=True,
        split_classes_on_threshold=False,
        split_classes=True,
        save_tsv=False,
        save_dir=str(Path(ROOT) / cls_dir)

    )
