"""Example GNN for QM9"""
import numpy as np
import os
import tensorflow as tf
from datetime import datetime
from pathlib import Path
from time import time

from gnn import GNN, GNNInput
from gnn.initial import PadInitializer
from gnn.message_passing import FeedForwardMessage
from gnn.readout import GatedReadout
from gnn.update import GRUUpdate

tf.keras.backend.clear_session()
tf.autograph.set_verbosity(1)
# tf.config.experimental_run_functions_eagerly(True)

np.random.seed(42)

# Paths
log_dir = Path("data/logs")
validation_dir = Path("data/qm9/validation")
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
validation_fn = [validation_dir / fn for fn in os.listdir(validation_dir) if not fn.startswith(".")]
training_fn = [training_dir / fn for fn in os.listdir(training_dir) if not fn.startswith(".")]
print(f"Training samples: {len(training_fn)}")
print(f"Validation samples: {len(validation_fn)}")

# Logs
log_name = f"gnn_qm9_{datetime.now().strftime('%Y%m%d-%H%M%S')}"
log_subdir = log_dir / log_name
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_subdir)

# TF Datasets
training = tf.data.Dataset.from_generator(
    **GNNInput.get_data_generator(training_fn, node_feature_names, edge_feature_names, target))
validation = tf.data.Dataset.from_generator(
    **GNNInput.get_data_generator(validation_fn, node_feature_names, edge_feature_names, target))

# GNN Model
model = GNN(hidden_state_size=20, message_size=45, message_passing_iterations=3,
            output_size=1, initializer=PadInitializer, message_passing=FeedForwardMessage,
            update=GRUUpdate, readout=GatedReadout)

# Training params
batch_size = 2
n_epochs = 1
train = training.padded_batch(batch_size)
train_step_per_epochs = len(training_fn) // batch_size
valid = validation.padded_batch(1)
valid_step_per_epoch = len(validation_fn)
validation_freq = 1

# Model compile parameters
learning_schedule = tf.keras.optimizers.schedules.PolynomialDecay(
    initial_learning_rate=1.935e-4, decay_steps=n_epochs, end_learning_rate=1.84e-4, power=1.0
)
optimizer = tf.keras.optimizers.Adam(learning_rate=learning_schedule)
loss = tf.keras.losses.MeanSquaredError()
metrics = [tf.keras.metrics.MeanAbsoluteError()]

model.compile(
    optimizer=optimizer,
    loss=loss,
    metrics=metrics
)

# Model fit
start = time()
loss = model.fit(
    train, epochs=n_epochs, steps_per_epoch=train_step_per_epochs,
    validation_data=valid, validation_freq=validation_freq, validation_steps=valid_step_per_epoch,
    callbacks=[tensorboard_callback], use_multiprocessing=True)
elapsed = time() - start
print(f"Fit ended after {elapsed} seconds")
