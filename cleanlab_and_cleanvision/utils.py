from pathlib import Path
from sklearn import model_selection


def write_filenames_to_file(save_dir: str | Path, filename: str, names_lst: list[str]):
    with open(Path(save_dir) / filename, mode='w', encoding='utf-8') as f:
        f.writelines(names_lst)


def split_k_fold_files(root: str | Path, save_dir: str | Path, n_folds: int = 5,
                       pattern: str = '*', seed: int = 0, has_train_val_split: bool = False):
    """ Assume root/(train, val, test)/(class_1, class_2, etc) is has_train_val_split = True,
        else root/(class_1, class_2, etc)
    """
    abs_filenames = [x for x in Path(root).rglob(pattern) if x.is_file()]
    filenames = [x.name for x in abs_filenames]
    labels = [x.parent.name for x in abs_filenames]

    k_folds = model_selection.StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)

    for fold_idx, (train_filenames, test_filenames) in enumerate(k_folds.split(filenames, labels)):
        write_filenames_to_file(save_dir, f'train_{fold_idx}_fold.txt', train_filenames)
        write_filenames_to_file(save_dir, f'test_{fold_idx}_fold.txt', test_filenames)


def read_filenames_from_file(filename: str | Path) -> list[str]:
    with open(filename, mode='r', encoding='utf-8') as f:
        names = f.readlines()
    return names



