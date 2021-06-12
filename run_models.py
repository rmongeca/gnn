from datetime import datetime
from pathlib import Path
from time import time

import numpy as np
from sklearn.model_selection import RepeatedKFold, KFold

import json

import ignnition
import utils
from models import qm9, radio_resource_management as rrm


def main(
    source_dir: Path, training_dir: Path, validation_dir: Path, ignnition_dir: Path, log_dir: Path,
    gnn_model, num_folds=5, num_repeats=0, random_seed=42
):
    assert source_dir.exists()
    assert training_dir.exists()
    assert validation_dir.exists()
    assert ignnition_dir.exists()
    assert log_dir.exists()
    # Params
    rng = np.random.RandomState(random_seed)
    # Get files and splits
    files = np.array(list(source_dir.glob("*.json")))
    rng.shuffle(files)
    kfold = (
        RepeatedKFold(n_splits=num_folds, n_repeats=num_repeats, random_state=rng)
        if num_repeats > 0 else KFold(n_splits=num_folds, random_state=rng, shuffle=True)
    )
    splits = kfold.split(files)
    run_times = {
        "ignnition": [],
        "gnn": []
    }
    # Iterate over splits and train
    for train_idx, validation_idx in splits:
        # Join files into JSON datasets
        utils.graph.join_graphs_into_dataset(files[train_idx], training_dir, empty_dirs=True)
        utils.graph.join_graphs_into_dataset(files[validation_idx], validation_dir, empty_dirs=True)
        # Ignnition
        start = time()
        model = ignnition.create_model(model_dir=ignnition_dir.absolute())
        model.train_and_validate()
        run_times["ignnition"].append(time() - start)
        # GNN
        start = time()
        gnn_model.main(log_dir=log_dir, training_dir=training_dir, validation_dir=validation_dir)
        run_times["gnn"].append(time() - start)
    # Save time executions
    with open(log_dir / f"run_model_times_{datetime.now():%Y%m%d%H%M%S}.json", "w") as fp:
        json.dump(run_times, fp)


if __name__ == "__main__":
    # main(
    #     source_dir=Path("ignnition/qm9/data/raw"),
    #     training_dir=Path("ignnition/qm9/data/train"),
    #     validation_dir=Path("ignnition/qm9/data/validation"),
    #     ignnition_dir=Path("ignnition/qm9"),
    #     log_dir=Path("ignnition/qm9/logs"),
    #     gnn_model=qm9,
    #     random_seed=20210506
    # )
    main(
        source_dir=Path("ignnition/radio-resource-management/data/raw"),
        training_dir=Path("ignnition/radio-resource-management/data/train"),
        validation_dir=Path("ignnition/radio-resource-management/data/validation"),
        ignnition_dir=Path("ignnition/radio-resource-management"),
        log_dir=Path("ignnition/radio-resource-management/logs"),
        gnn_model=rrm,
        random_seed=20210221
    )
