from pathlib import Path

import numpy as np
from sklearn.model_selection import RepeatedKFold

import ignnition
import utils
from models import qm9


def main(
    source_dir, training_dir, validation_dir, ignnition_dir, log_dir,
    num_folds=11, num_repeats=1, random_seed=42
):
    # Params
    rng = np.random.RandomState(random_seed)
    # Get files and splits
    files = np.array(list(source_dir.glob("*.json")))
    rng.shuffle(files)
    splits = RepeatedKFold(
        n_splits=num_folds, n_repeats=num_repeats, random_state=rng
    ).split(files)
    # Iterate over splits and train
    for train_idx, validation_idx in splits:
        # Join files into JSON datasets
        utils.graph.join_graphs_into_dataset(files[train_idx], training_dir, empty_dirs=True)
        utils.graph.join_graphs_into_dataset(files[validation_idx], validation_dir, empty_dirs=True)
        # Ignnition
        model = ignnition.create_model(model_dir=ignnition_dir.absolute())
        model.train_and_validate()
        # GNN
        qm9.main(log_dir=log_dir, training_dir=training_dir, validation_dir=validation_dir)


if __name__ == "__main__":
    main(
        source_dir=Path("data/qm9/raw"),
        training_dir=Path("data/qm9/train"),
        validation_dir=Path("data/qm9/validation"),
        ignnition_dir=Path("ignnition/qm9"),
        log_dir=Path("logs"),
    )
