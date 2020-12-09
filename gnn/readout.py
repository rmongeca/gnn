"""Readout Layer for GNN implementation."""
import tensorflow as tf

from gnn.input import ReadoutInput


class GatedReadout(tf.keras.layers.Layer):
    """Readout function layer with Gated NN for GNN model."""

    def __init__(self, hidden_state_size=10, output_size=1,
                 gate_activation=tf.keras.activations.sigmoid, *args, **kwargs):
        super(GatedReadout, self).__init__(*args, **kwargs)
        self.hidden_state_size = hidden_state_size
        self.output_size = output_size
        self.gate_activation = gate_activation
        # Init inner layers
        self.gate = tf.keras.layers.Dense(units=output_size, name="readout-gate-hidden-initial")
        self.state = tf.keras.layers.Dense(units=output_size, name="readout-state-hidden")

    def call(self, inputs: ReadoutInput, training=None):
        # Get input info
        batch_size = tf.shape(inputs.hidden)[0]
        # Handle batches independently
        return tf.map_fn(
            fn=lambda batch: self._handle_batch(batch, inputs, training=training),
            elems=tf.range(batch_size),
            fn_output_signature=tf.TensorSpec([self.output_size], float)
        )

    def _handle_batch(self, batch, inputs: ReadoutInput, training=None):
        hidden = inputs.hidden[batch, :, :]
        hidden_initial = inputs.hidden_initial[batch, :, :]
        hidden_concat = tf.concat([hidden, hidden_initial], axis=-1)
        gate = self.gate_activation(self.gate(hidden_concat, training=training))
        state = self.state(hidden, training=training)
        return tf.math.reduce_sum(tf.math.multiply(gate, state), axis=0)
