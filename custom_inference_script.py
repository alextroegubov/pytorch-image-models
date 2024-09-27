from pathlib import Path
import inference_copy

CLASS_MAP = "/home/user/Documents/data/plates/license_plates_readability/class_map.txt"
CHECKPOINT = "/home/user/Documents/repos/pytorch-image-models/output/convnext_pico.d1_in1k-convnext_pico_color_aug_full_dataset-v1/model_best.pth.tar"

MODEL_NAME = "convnext_pico.d1_in1k"
DEVICE = 'cuda'
NUM_CLASSES = 2

CROP_PCT = 1.0
CROP_MODE = 'border'
INTERPOLATION = 'bicubic'
INPUT_SIZE = (3, 224, 224)

BATCH_SIZE = 10
THRESHOLD = 0.75

ROOT = "/home/user/Documents/data/plates/numbers_15_crop"
SUB_DIRS = ['']

SHOW_STATS = True
MOVE_FILES = True
SINGLE_FOLDER = True


for cls_dir in SUB_DIRS:
    inference_copy.inference(
        data_dir=str(Path(ROOT) / cls_dir),
        class_map=CLASS_MAP,
        checkpoint=CHECKPOINT,
        model_name=MODEL_NAME,
        device=DEVICE,
        num_classes=NUM_CLASSES,
        crop_pct=CROP_PCT,
        crop_mode=CROP_MODE,
        interpolation=INTERPOLATION,
        batch_size=BATCH_SIZE,
        input_size=INPUT_SIZE,

        threshold=THRESHOLD,
        show_stats=SHOW_STATS,
        only_target_class=None,
        split_classes_on_threshold=False,
        split_classes=True,
        save_tsv=True,
        save_dir=str(Path(ROOT) / cls_dir)

    )
