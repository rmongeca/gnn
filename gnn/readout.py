"""Readout Layer for GNN implementation."""
import tensorflow as tf
from abc import ABC, abstractmethod

from .input import ReadoutInput


class ReadoutLayer(tf.keras.layers.Layer, ABC):
    """Base Readout Layer abstract class to be inherited by Readout Layer implementations.

    Abstract class to define handling of Batched ReadoutInput for readout layer to perform Readout.
    Child classes must implement the readout method, which takes as argument a non-batched
    ReadoutInput.
    """

    def call(self, inputs: ReadoutInput, training=None):
        batch_size = tf.shape(inputs.hidden)[0]
        return tf.map_fn(
            fn=lambda batch: self._handle_batch(batch, inputs, training=training),
            elems=tf.range(batch_size),
            fn_output_signature=tf.TensorSpec([self.output_size], float)
        )

    def _handle_batch(self, batch, inputs: ReadoutInput, training=None):
        batch_inputs = ReadoutInput(
            hidden=inputs.hidden[batch, :, :],
            hidden_initial=inputs.hidden_initial[batch, :, :],
        )
        return self.readout(batch_inputs, training=training)

    @abstractmethod
    def readout(self, inputs: ReadoutInput, training=None):
        pass


class GatedReadout(ReadoutLayer):
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

    def build(self, input_shapes):
        gate_shape = tf.TensorShape([None, None, input_shapes.hidden[2]*2])
        self.gate.build(gate_shape)
        self.state.build(input_shapes.hidden)
        super(ReadoutLayer, self).build([])

    def readout(self, inputs: ReadoutInput, training=None):
        hidden = inputs.hidden
        hidden_initial = inputs.hidden_initial
        hidden_concat = tf.concat([hidden, hidden_initial], axis=-1)
        gate_in = self.gate(hidden_concat, training=training)
        gate = self.gate_activation(gate_in)
        state = self.state(hidden, training=training)
        return tf.math.reduce_sum(tf.math.multiply(gate, state), axis=0)
