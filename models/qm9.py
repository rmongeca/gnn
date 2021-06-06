"""Example GNN for QM9"""
import json
import tensorflow as tf
from datetime import datetime
from pathlib import Path

from gnn import GNN, get_dataset_from_files
from gnn.initial import PadInitializer
from gnn.message_passing import FeedForwardMessage
from gnn.readout import GatedReadout
from gnn.update import GRUUpdate


def main(
    log_dir,
    training_dir,
    validation_dir,
):
    # Constants
    node_feature_names = [
        "acceptor",
        "aromatic",
        "atomic_number",
        "donor",
        "element_c",
        "element_f",
        "element_h",
        "element_n",
        "element_o",
        "hybridization_null",
        "hybridization_sp",
        "hybridization_sp2",
        "hybridization_sp3",
        "hydrogen_count",
    ]
    edge_feature_names = [
        "distance", "order_1", "order_1_5", "order_2", "order_3"
    ]
    target = "dipole_moment"
    # Training params
    batch_size = 1
    n_epochs = 20
    train_step_per_epochs = 1000
    valid_step_per_epoch = 100
    validation_freq = 1
    learning_schedule_params = {
        "initial_learning_rate": 1.935e-4,
        "decay_steps": 20000,
        "end_learning_rate": 1.84e-4,
        "power": 1.0
    }
    # Files
    training_fn = list(Path(training_dir).glob("*.json"))
    validation_fn = list(Path(validation_dir).glob("*.json"))
    # Optimizer Loss Metric
    learning_schedule = tf.keras.optimizers.schedules.PolynomialDecay(**learning_schedule_params)
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_schedule)
    loss = tf.keras.losses.MeanSquaredError()
    metrics = [tf.keras.metrics.MeanAbsoluteError()]
    # Model
    message_passing_args = {
        "aggregation_fn": tf.math.reduce_sum,
        "activation": "relu",
        "layer": tf.keras.layers.Dense,
        "num_layers": 4,
        "units": 50,
    }
    readout_args = {
        "activation": "relu",
        "gate_activation": "sigmoid",
        "layer": tf.keras.layers.Dense,
        "num_layers": 3,
        "units": 50,
    }
    model = GNN(hidden_state_size=25, message_size=25, message_passing_iterations=4,
                output_size=1, initializer=PadInitializer, message_passing=FeedForwardMessage,
                update=GRUUpdate, readout=GatedReadout, message_passing_args=message_passing_args,
                readout_args=readout_args)
    model.compile(
        optimizer=optimizer,
        loss=loss,
        metrics=metrics
    )
    # Callbacks
    log_name = f"gnn_qm9_{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    log_subdir = log_dir / log_name
    callbacks = [
        tf.keras.callbacks.TensorBoard(
            log_dir=log_subdir, update_freq='epoch', write_images=False, histogram_freq=1)
    ]
    # Datasets
    training = get_dataset_from_files(
        training_fn, node_feature_names, edge_feature_names, target, batch_size=batch_size)
    validation = get_dataset_from_files(
        validation_fn, node_feature_names, edge_feature_names, target, batch_size=1)
    # Fit History
    loss = model.fit(
        training, epochs=n_epochs, steps_per_epoch=train_step_per_epochs,
        validation_data=validation, validation_freq=validation_freq,
        validation_steps=valid_step_per_epoch, callbacks=callbacks, use_multiprocessing=True)
    json.dump(loss.history, open(Path(log_subdir) / "history.json", "w"))


if __name__ == "__main__":
    # Paths
    training_dir = Path("data/qm9/train")
    validation_dir = Path("data/qm9/validation")
    log_dir = Path("data/logs")
    main(
        log_dir=log_dir,
        training_dir=training_dir,
        validation_dir=validation_dir,
    )
