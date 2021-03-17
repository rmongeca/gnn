"""Example GNN for QM9"""
import numpy as np
import os
import tensorflow as tf
from datetime import datetime
from pathlib import Path
from time import time

from gnn import GNN, GNNInput
from gnn.initial import PadInitializer
from gnn.message_passing import EdgeNetMessage
from gnn.readout import GatedReadout
from gnn.update import GRUUpdate

tf.keras.backend.clear_session()
# tf.config.experimental_run_functions_eagerly(True)

np.random.seed(42)

# Paths
log_dir = Path("data/logs")
test_dir = Path("data/qm9/test")
training_dir = Path("data/qm9/training")

# Constants
node_feature_names = [
    "element_c", "element_f", "element_h", "element_n", "element_o",
    "aromatic", "acceptor", "donor", "atomic_number",
    "hybridization_null", "hybridization_sp", "hybridization_sp2",
    "hybridization_sp3", "hydrogen_count"
]
edge_feature_names = [
    "distance", "order_1", "order_1_5", "order_2", "order_3"
]
target = "dipole_moment"

# Files
test_fn = [test_dir / fn for fn in os.listdir(test_dir) if not fn.startswith(".")]
training_fn = [training_dir / fn for fn in os.listdir(training_dir) if not fn.startswith(".")]
print(f"Training samples: {len(training_fn)}")
print(f"Test samples: {len(test_fn)}")

# Logs
log_name = f"gnn_qm9_{datetime.now().strftime('%Y%m%d-%H%M%S')}"
log_subdir = log_dir / log_name
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_subdir)

# TF Datasets
training = tf.data.Dataset.from_generator(
    **GNNInput.get_data_generator(training_fn, node_feature_names, edge_feature_names, target))
test = tf.data.Dataset.from_generator(
    **GNNInput.get_data_generator(test_fn, node_feature_names, edge_feature_names, target))

# Training params
n_epochs = 10
batch_size = 10
train_step_per_epochs = len(training_fn) // batch_size
valid_step_per_epoch = len(test_fn)
learning_schedule = tf.keras.optimizers.schedules.PolynomialDecay(
    initial_learning_rate=1.935e-4, decay_steps=900, end_learning_rate=1.84e-4, power=1.0
)
optimizer = tf.keras.optimizers.Adam(learning_rate=learning_schedule)
loss = tf.keras.losses.MeanSquaredError()
metrics = [tf.keras.metrics.MeanAbsoluteError()]


# GNN Model
model = GNN(hidden_state_size=20, message_size=20, message_passing_iterations=4,
            output_size=1, initializer=PadInitializer, message_passing=EdgeNetMessage,
            update=GRUUpdate, readout=GatedReadout)
model.compile(
    optimizer=optimizer,
    loss=loss,
    metrics=metrics
)

# Model fit
start = time()
loss = model.fit(
    training.padded_batch(batch_size), epochs=n_epochs, steps_per_epoch=train_step_per_epochs,
    validation_data=test.padded_batch(1), validation_freq=1, validation_steps=valid_step_per_epoch,
    callbacks=[tensorboard_callback], use_multiprocessing=True)
elapsed = time() - start
print(f"Fit ended after {elapsed} seconds")
