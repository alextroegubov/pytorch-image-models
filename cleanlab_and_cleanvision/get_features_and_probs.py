import features_and_probs
import torch


DATA_DIR = ""
MODEL_NAME = ""
NUM_CLASSES = 8
CHECKPOINT = ""
DEVICE = torch.device('cuda')
INPUT_SIZE = (3, 224, 224)
CROP_PCT = 1
CROP_MODE = 'center'
PADDING_MODE = 'reflect'
INTERPOLATION = 'bicubic'
BATCH_SIZE = 128
SAVE_FEATURES = True
SAVE_PROBS = True
SAVE_DIR = 'embeddings'

features_and_probs.features_and_probabilities(
    data_dir=DATA_DIR,
    model_name=MODEL_NAME,
    num_classes=NUM_CLASSES,
    checkpoint=CHECKPOINT,
    device=DEVICE,
    input_size=INPUT_SIZE,
    crop_pct=1,
    crop_mode=CROP_MODE,
    padding_mode=PADDING_MODE,
    interpolation=INTERPOLATION,
    batch_size=BATCH_SIZE,
    save_features=True,
    save_probs=True,
    save_dir=SAVE_DIR,
)