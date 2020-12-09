"""Example GNN for QM9"""
import json
import numpy as np
import os
import tensorflow as tf
from datetime import datetime
from time import time

from gnn import GNN, GNNInput, DenseInitializer

tf.keras.backend.clear_session()
# tf.config.experimental_run_functions_eagerly(True)

np.random.seed(42)

training_dir = "/home/rmonge/UPC/TFM/gnn-qm9/data/training"
training_fn = [os.path.join(training_dir, fn) for fn in os.listdir(training_dir)
               if not fn.startswith(".")]
test_dir = "/home/rmonge/UPC/TFM/gnn-qm9/data/test"
test_fn = [os.path.join(test_dir, fn) for fn in os.listdir(test_dir) if not fn.startswith(".")]
target = "dipole_moment"

print(f"Training samples: {len(training_fn)}")
print(f"Test samples: {len(test_fn)}")

training = tf.data.Dataset.from_generator(
    **GNNInput.get_data_generator(training_fn, target))
test = tf.data.Dataset.from_generator(
    **GNNInput.get_data_generator(test_fn, target))

graph, y = next(training.take(1).as_numpy_iterator())

model = GNN(hidden_state_size=15, message_size=20, message_passing_iterations=3, output_size=1)

learning_schedule = tf.keras.optimizers.schedules.PolynomialDecay(
    initial_learning_rate=1.935e-4, decay_steps=900, end_learning_rate=1.84e-4, power=1.0
)
optimizer = tf.keras.optimizers.Adam(learning_rate=learning_schedule)
loss = tf.keras.losses.MeanSquaredError()
metrics = [tf.keras.metrics.MeanAbsoluteError()]

model.compile(
    optimizer=optimizer,
    loss=loss,
    metrics=metrics
)

log_name = f"gnn_{datetime.now().strftime('%Y%m%d-%H%M%S')}"
log_dir = f"/home/rmonge/UPC/TFM/ignnition_data/CheckPoint/{log_name}"
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir)

start = time()
loss = model.fit(
    training, epochs=10, steps_per_epoch=1000,
    validation_data=test, validation_freq=1, validation_steps=1000,
    callbacks=[tensorboard_callback], use_multiprocessing=True)
elapsed = time() - start
print(f"Fit ended after {elapsed} seconds")

