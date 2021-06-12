"""Example GNN for Radio Resource Management"""
import json
import tensorflow as tf
from datetime import datetime
from pathlib import Path

from gnn import GNN, get_dataset_from_files
from gnn.initial import NormalizeInitializer
from gnn.message_passing import FeedForwardMessage
from gnn.readout import FeedForwardReadout
from gnn.update import FeedForwardUpdate


noise_power = tf.constant(6.294627058970857e-15)


@tf.function()
def compute_sum_rate(power, loss, weights, N):
    """Compute Sum Rate from power allocation and channle loss matrix."""
    batch_shape = tf.shape(power)[:1]
    # Prepare power tensor
    power_tiled = tf.tile(power, [1, N, 1])
    rx_power = tf.math.square(tf.multiply(loss, power_tiled))
    # Prepare masks diagonal and off-diagonal masks
    mask = tf.eye(N, batch_shape=batch_shape)
    mask_inverse = tf.ones_like(mask) - mask
    # Compute valid power/interferences per transciever-reciever-pair
    valid_rx_power = tf.reduce_sum(tf.multiply(rx_power, mask), axis=-1)
    interference = tf.reduce_sum(tf.multiply(rx_power, mask_inverse), axis=-1)
    interference += tf.tile([[noise_power]], tf.shape(interference))
    # Compute SINR rates
    sinr = tf.ones_like(interference) + tf.divide(valid_rx_power, interference)
    sum_rate = tf.divide(tf.math.log(sinr), tf.math.log(tf.constant(2, dtype=tf.float32)))
    weighted_sum_rate = tf.multiply(weights, sum_rate)
    return tf.reduce_mean(tf.reduce_sum(weighted_sum_rate, -1))


@tf.function()
def sum_rate_loss(y_true, y_pred):
    """SINR sum rate loss.

    Loss function for Radio Resource Management example, computing the expected sum rate value.
    Inputs are batched tensors with shape (b, n, ?) where b is batch_size, n is the number of
    nodes in the graph an ? changes depending on input.

    Parameters
    ----------
    y_true : tf.Tensor
        Batched tensor with with shape (b, n, n) containing for each node the path losses from the
        node (transceiver-reciever-pair) to all others, include itself.
    y_pred : tf.Tensor
        Batched tensor with GNN output, containing a hidden state with shape (b, n, 4), where second
        last element is the allocated power to transceiver-reciever-pair and last element is the
        reference wmmse allocated power aproximation to compare with computed.
    """
    N = tf.shape(y_pred)[1]
    weights = y_pred[:, :, -2]
    power = tf.expand_dims(y_pred[:, :, -3], axis=0)
    sum_rate = compute_sum_rate(power, y_true, weights, N)
    return tf.negative(sum_rate)


@tf.function()
def sum_rate_metric(y_true, y_pred):
    """WMMSE ratio metric.

    Metric function for Radio Resource Management example, computing the sum rate normalized by the
    wmmse sum rate. Inputs are batched tensors with shape (b, n, ?) where b is batch_size, n is the
    number of nodes in the graph an ? changes depending on input.

    Parameters
    ----------
    y_true : tf.Tensor
        Batched tensor with with shape (b, n, n) containing for each node the path losses from the
        node (transceiver-reciever-pair) to all others, include itself.
    y_pred : tf.Tensor
        Batched tensor with GNN output, containing a hidden state with shape (b, n, 4), where second
        last element is the allocated power to transceiver-reciever-pair and last element is the
        reference wmmse allocated power aproximation to compare with computed.
    """
    N = tf.shape(y_pred)[1]
    power_wmmse = tf.expand_dims(y_pred[:, :, -1], axis=0)
    weights = y_pred[:, :, -2]
    power = tf.expand_dims(y_pred[:, :, -3], axis=0)
    sum_rate_wmmse = compute_sum_rate(power_wmmse, y_true, weights, N)
    sum_rate = compute_sum_rate(power, y_true, weights, N)
    return tf.multiply(tf.divide(sum_rate, sum_rate_wmmse), tf.constant(100, dtype=tf.float32))


def main(
    log_dir,
    training_dir,
    validation_dir,
):
    # Constants
    node_feature_names = [
        "receiver_distance",
        "channel_loss",
        "power",
    ]
    edge_feature_names = [
        "transceiver_receiver_loss"
    ]
    additional_inputs_names = [
        "weights",
        "wmmse_power",
    ]
    target = "path_loss"
    target_shapes = tf.TensorShape([None, None])
    # Training params
    batch_size = 1
    n_epochs = 10
    train_step_per_epochs = 2000
    valid_step_per_epoch = 500
    validation_freq = 1
    learning_rate = 0.001
    # Files
    training_fn = list(Path(training_dir).glob("*.json"))
    validation_fn = list(Path(validation_dir).glob("*.json"))
    # Optimizer Loss Metric
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    loss = sum_rate_loss
    metrics = [sum_rate_metric]
    # Model
    message_passing_args = {
        "aggregation_fn": tf.math.reduce_max,
        "activation": "relu",
        "layer": tf.keras.layers.Dense,
        "num_layers": 3,
        "units": [5, 32, 32],
        "output_activation": "relu",
    }
    update_args = {
        "activation": "relu",
        "layer": tf.keras.layers.Dense,
        "num_layers": 3,
        "units": [35, 16],
        "output_activation": "sigmoid",
    }
    readout_args = {
        "layer": tf.keras.layers.Activation,
        "num_layers": 1,
        "output_activation": "relu",
        "trainable": False,
    }
    model = GNN(hidden_state_size=3, message_size=3, message_passing_iterations=3,
                output_size=3, initializer=NormalizeInitializer, message_passing=FeedForwardMessage,
                update=FeedForwardUpdate, readout=FeedForwardReadout,
                message_passing_args=message_passing_args, update_args=update_args,
                readout_args=readout_args)
    model.compile(
        optimizer=optimizer,
        loss=loss,
        metrics=metrics
    )
    # Callbacks
    log_name = f"gnn_radio_{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    log_subdir = log_dir / log_name
    callbacks = [
        tf.keras.callbacks.TensorBoard(
            log_dir=log_subdir, update_freq='epoch', write_images=False, histogram_freq=1)
    ]
    # Datasets
    training = get_dataset_from_files(
        training_fn, node_feature_names, edge_feature_names, target, additional_inputs_names,
        target_shapes=target_shapes, batch_size=batch_size, local=True)
    validation = get_dataset_from_files(
        validation_fn, node_feature_names, edge_feature_names, target, additional_inputs_names,
        target_shapes=target_shapes, batch_size=batch_size, local=True)
    # Fit History
    loss = model.fit(
        training, epochs=n_epochs, steps_per_epoch=train_step_per_epochs,
        validation_data=validation, validation_freq=validation_freq,
        validation_steps=valid_step_per_epoch, callbacks=callbacks, use_multiprocessing=True)
    json.dump(loss.history, open(Path(log_subdir) / "history.json", "w"))


if __name__ == "__main__":
    # Paths
    training_dir = Path("data/radio-resource-management/train")
    validation_dir = Path("data/radio-resource-management/validation")
    log_dir = Path("logs")
    main(
        log_dir=log_dir,
        training_dir=training_dir,
        validation_dir=validation_dir,
    )
