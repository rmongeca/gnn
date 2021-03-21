"""Readout Layer for GNN implementation."""
import tensorflow as _tf
from abc import ABC as _ABC, abstractmethod as _abstractmethod

from .input import ReadoutInput
from .mlp import MLP


class ReadoutLayer(_tf.keras.layers.Layer, _ABC):
    """Base Readout Layer abstract class to be inherited by Readout Layer implementations.

    Abstract class to define handling of Batched ReadoutInput for readout layer to perform Readout.
    Child classes must implement the readout method, which takes as argument a non-batched
    ReadoutInput.
    """

    def call(self, inputs: ReadoutInput, training=None):
        batch_size = _tf.shape(inputs.hidden)[0]
        return _tf.map_fn(
            fn=lambda batch: self._handle_batch(batch, inputs, training=training),
            elems=_tf.range(batch_size),
            fn_output_signature=_tf.TensorSpec([self.output_size], float)
        )

    def _handle_batch(self, batch, inputs: ReadoutInput, training=None):
        batch_inputs = ReadoutInput(
            hidden=inputs.hidden[batch, :, :],
            hidden_initial=inputs.hidden_initial[batch, :, :],
        )
        return self._readout(batch_inputs, training=training)

    @_tf.function
    def _readout(self, inputs: ReadoutInput, training=None):
        """Wrapper around readout method with tf.function decorator."""
        return self.readout(inputs=inputs, training=training)

    @_abstractmethod
    def readout(self, inputs: ReadoutInput, training=None):
        """Readout method to apply to a certain batch's ReadoutInput."""
        pass


class GatedReadout(ReadoutLayer):
    """Readout function layer with Gated NN for GNN model."""

    def __init__(
        self, hidden_state_size=10, output_size=1, nn_layers=3, nn_units=50, nn_activation="relu",
        gate_activation=_tf.keras.activations.sigmoid, *args, **kwargs
    ):
        super(GatedReadout, self).__init__(*args, **kwargs)
        self.hidden_state_size = hidden_state_size
        self.output_size = output_size
        self.nn_units = nn_units
        # Init inner layers
        self.gate = MLP(
            activation=nn_activation, name="readout-gate", num_layers=nn_layers, units=nn_units,
            output_activation=gate_activation, **kwargs)
        self.state = MLP(
            activation=nn_activation, name="readout-state", num_layers=nn_layers, units=nn_units,
            **kwargs)
        self.graph = MLP(
            activation=nn_activation, name="readout-state", num_layers=nn_layers, units=nn_units,
            output_units=output_size, **kwargs)

    def build(self, input_shapes):
        gate_shape = _tf.TensorShape([None, input_shapes.hidden[2]*2])
        state_shape = _tf.TensorShape([None, input_shapes.hidden[2]])
        graph_shape = _tf.TensorShape([None, self.nn_units])
        self.gate.build(gate_shape)
        self.state.build(state_shape)
        self.graph.build(graph_shape)
        super(ReadoutLayer, self).build([])

    def readout(self, inputs: ReadoutInput, training=None):
        hidden = inputs.hidden
        hidden_initial = inputs.hidden_initial
        hidden_concat = _tf.concat([hidden, hidden_initial], axis=-1)
        gate = self.gate(hidden_concat, training=training)
        state = self.state(hidden, training=training)
        reduced = _tf.math.reduce_sum(_tf.math.multiply(gate, state), axis=0, keepdims=True)
        return _tf.squeeze(self.graph(reduced), axis=0)
