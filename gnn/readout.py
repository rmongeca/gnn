"""Readout Layer for GNN implementation."""
import tensorflow as _tf
from abc import ABC as _ABC, abstractmethod as _abstractmethod

from .input import ReadoutInput
from .mlp import MLP


class ReadoutLocalLayer(_tf.keras.layers.Layer, _ABC):
    """Base Readout Local Layer abstract class to be inherited by Readout Layer implementations.

    Abstract class to define handling of Batched ReadoutInput for readout layer to perform Local or
    node-level Readout. Child classes must implement the readout method, which takes as argument a
    non-batched ReadoutInput and return a Tensor of shape (None, output_size) where None represents
    the number of nodes.
    """

    def __init__(self, hidden_state_size=10, output_size=1, *args, **kwargs):
        """Constructor for Readout Layer abstract class."""
        super(ReadoutLocalLayer, self).__init__(*args, **kwargs)
        self.output_size = output_size

    def call(self, inputs: ReadoutInput, training=None):
        batch_size = _tf.shape(inputs.hidden)[0]
        return _tf.map_fn(
            fn=lambda batch: self._handle_batch(batch, inputs, training=training),
            elems=_tf.range(batch_size),
            fn_output_signature=_tf.TensorSpec([None, self.output_size], float)
        )

    def _handle_batch(self, batch, inputs: ReadoutInput, training=None):
        batch_inputs = ReadoutInput(
            hidden=inputs.hidden[batch, :, :],
            hidden_initial=inputs.hidden_initial[batch, :, :],
        )
        return self.readout(batch_inputs, training=training)

    @_abstractmethod
    def readout(self, inputs: ReadoutInput, training=None):
        """Readout method to apply to a certain batch's ReadoutInput."""
        pass


class ReadoutGlobalLayer(_tf.keras.layers.Layer, _ABC):
    """Base Readout Global Layer abstract class to be inherited by Readout Layer implementations.

    Abstract class to define handling of Batched ReadoutInput for readout layer to perform Global or
    graph-level Readout. Child classes must implement the readout method, which takes as argument a
    non-batched ReadoutInput and return a Tensor of shape (output_size,).
    """

    def __init__(self, hidden_state_size=10, output_size=1, *args, **kwargs):
        """Constructor for Readout Layer abstract class."""
        super(ReadoutGlobalLayer, self).__init__(*args, **kwargs)
        self.output_size = output_size

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
        return self.readout(batch_inputs, training=training)

    @_abstractmethod
    def readout(self, inputs: ReadoutInput, training=None):
        """Readout method to apply to a certain batch's ReadoutInput."""
        pass


class FeedForwardReadout(ReadoutLocalLayer):
    """Readout function layer with a Feed Forward NN for GNN model."""

    def __init__(
        self, output_size=1, activation="relu", layer=None, num_layers=4, output_activation=None,
        units=50, *args, **kwargs
    ):
        super(FeedForwardReadout, self).__init__(output_size=output_size, *args, **kwargs)
        self.mlp = MLP(
            activation=activation, layer=layer, name="readout-ff-net", num_layers=num_layers,
            output_activation=output_activation, output_units=output_size, units=units, **kwargs)

    def build(self, input_shapes):
        self.mlp.build(_tf.TensorShape([None, input_shapes.hidden[2]]))
        super(FeedForwardReadout, self).build([])

    def readout(self, inputs: ReadoutInput, training=None):
        hidden = inputs.hidden
        return self.mlp(hidden, training=training)


class GatedReadout(ReadoutGlobalLayer):
    """Readout function layer with Gated NN for GNN model."""

    def __init__(
        self, hidden_state_size=10, output_size=1, activation="relu", layer=None, num_layers=3,
        gate_activation=_tf.keras.activations.sigmoid, units=50, *args, **kwargs
    ):
        super(GatedReadout, self).__init__(output_size=output_size, *args, **kwargs)
        self.hidden_state_size = hidden_state_size
        self.output_size = output_size
        self.units = units
        # Init inner layers
        self.gate = MLP(
            activation=activation, layer=layer, name="readout-gate", num_layers=num_layers,
            output_activation=gate_activation, units=units, **kwargs)
        self.state = MLP(
            activation=activation, layer=layer, name="readout-state", num_layers=num_layers,
            units=units, **kwargs)
        self.graph = MLP(
            activation=activation, layer=layer, name="readout-state", num_layers=num_layers,
            output_units=output_size, units=units, **kwargs)

    def build(self, input_shapes):
        gate_shape = _tf.TensorShape([None, input_shapes.hidden[2]*2])
        state_shape = _tf.TensorShape([None, input_shapes.hidden[2]])
        graph_shape = _tf.TensorShape([None, self.units])
        self.gate.build(gate_shape)
        self.state.build(state_shape)
        self.graph.build(graph_shape)
        super(GatedReadout, self).build([])

    def readout(self, inputs: ReadoutInput, training=None):
        hidden = inputs.hidden
        hidden_initial = inputs.hidden_initial
        hidden_concat = _tf.concat([hidden, hidden_initial], axis=-1)
        gate = self.gate(hidden_concat, training=training)
        state = self.state(hidden, training=training)
        reduced = _tf.math.reduce_sum(_tf.math.multiply(gate, state), axis=0, keepdims=True)
        return _tf.squeeze(self.graph(reduced), axis=0)
