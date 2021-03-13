"""Example GNN for QM9"""
import numpy as np
import os
import tensorflow as tf
from datetime import datetime
from pathlib import Path
from time import time

from gnn import GNN, GNNInput
from gnn.initial import DenseInitializer
from gnn.message_passing import EdgeNetMessagePassing
from gnn.readout import GatedReadout
from gnn.update import GRUUpdate

tf.keras.backend.clear_session()
# tf.config.experimental_run_functions_eagerly(True)

np.random.seed(42)

# Paths
log_dir = Path("/home/rmonge/UPC/TFM/gnn/data/logs")
test_dir = Path("/home/rmonge/UPC/TFM/gnn/data/qm9/test")
training_dir = Path("/home/rmonge/UPC/TFM/gnn/data/qm9/training")

# Constants
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
    **GNNInput.get_data_generator(training_fn, target))
test = tf.data.Dataset.from_generator(
    **GNNInput.get_data_generator(test_fn, target))

# Training params
n_epochs = 5
batch_size = 32
train_step_per_epochs = len(training_fn) // batch_size + 1
valid_step_per_epoch = len(test_fn)
learning_schedule = tf.keras.optimizers.schedules.PolynomialDecay(
    initial_learning_rate=1.935e-4, decay_steps=900, end_learning_rate=1.84e-4, power=1.0
)
optimizer = tf.keras.optimizers.Adam(learning_rate=learning_schedule)
loss = tf.keras.losses.MeanSquaredError()
metrics = [tf.keras.metrics.MeanAbsoluteError()]


# GNN Model
model = GNN(hidden_state_size=15, message_size=20, message_passing_iterations=3,
            output_size=len(target), initializer=DenseInitializer,
            message_passing=EdgeNetMessagePassing, update=GRUUpdate, readout=GatedReadout)
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
