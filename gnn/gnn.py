"""GNN model implementing a Message Passing Neural Network."""
import tensorflow as _tf

from .initial import PadInitializer
from .input import GNNInput, MessagePassingInput, ReadoutInput, UpdateInput
from .message_passing import ConcatenationMessage
from .readout import GatedReadout
from .update import GRUUpdate


class GNN(_tf.keras.Model):
    """General class for Message Passing Neural Network.

    Parameters
    ----------
    hidden_state_size : int
        Size of the hidden state vector for each node of the net. Defaults to 10.
    message_size : int
        Size of the messages generated through message_passing layer.
    message_passing_iterations : int
        Number of iterations for message passing phase of the net. Defaults to 3.
    output_size : int
        Dimension of output version, for graph-level regression. Defaults to 1.
    initializer : _tf.keras.layers.Layer
        Initializer layer for GNN. Defaults to gnn.initial.PadInitializer.
    message_passing : _tf.keras.layers.Layer
        Message Passing layer for GNN, subclass of gnn.message_passing.MessagePassingLayer.
        Defaults to gnn.message_passing.ConcatenationMessage.
    update : _tf.keras.layers.Layer
        Update layer for GNN. Defaults to gnn.update.GRUUpdate.
    readout : _tf.keras.layers.Layer
        Readout layer for GNN, subclass of gnn.readout.ReadoutLayer. Defaults to
        gnn.readout.GatedReadout.
    """

    def __init__(self, hidden_state_size=10, message_size=10, message_passing_iterations=3,
                 output_size=1, initializer=PadInitializer, message_passing=ConcatenationMessage,
                 update=GRUUpdate, readout=GatedReadout, *args, **kwargs):
        super(GNN, self).__init__(*args, **kwargs)
        # Record arguments
        self.hidden_state_size = hidden_state_size
        self.message_passing_iterations = message_passing_iterations
        self.message_size = message_size
        self.output_size = output_size
        self.initializer = initializer
        self.message_passing = message_passing
        self.update = update
        self.readout = readout
        self.kwargs = kwargs
        # Set inner layers
        self.init = self.initializer(hidden_state_size=self.hidden_state_size, **kwargs)
        self.mp = self.message_passing(hidden_state_size=self.hidden_state_size,
                                       message_size=self.message_size, **kwargs)
        self.up = self.update(hidden_state_size=self.hidden_state_size,
                              message_size=self.message_size, **kwargs)
        self.ro = self.readout(hidden_state_size=self.hidden_state_size,
                               output_size=self.output_size, **kwargs)

    def build(self, input_shape: GNNInput):
        edge_features = input_shape.edge_features
        node_features = input_shape.node_features
        num_batches = node_features[0]  # Should be None
        num_nodes = node_features[1]  # Should be None for variable size graphs
        num_edges = edge_features[1]  # Should be None for variable size graphs
        hidden_shape = _tf.TensorShape([num_batches, num_nodes, self.hidden_state_size])
        messages_shape = _tf.TensorShape([num_batches, num_nodes, self.message_size])
        source_shape = target_shape = _tf.TensorShape([num_batches, num_edges])
        self.init.build(node_features)
        self.mp.build(
            MessagePassingInput(
                edge_features=edge_features, edge_sources=source_shape,
                edge_targets=target_shape, hidden=hidden_shape))
        self.up.build(
            UpdateInput(hidden=hidden_shape, hidden_initial=hidden_shape, messages=messages_shape))
        self.ro.build(
            ReadoutInput(hidden=hidden_shape, hidden_initial=hidden_shape))
        super(GNN, self).build([])

    def call(self, inputs: GNNInput, training=None, mask=None):
        hidden = self.init(inputs.node_features, training=training)
        hidden_initial = _tf.identity(hidden)
        for _ in _tf.range(self.message_passing_iterations):
            messages = self.mp(
                MessagePassingInput(
                    edge_features=inputs.edge_features, edge_sources=inputs.edge_sources,
                    edge_targets=inputs.edge_targets, hidden=hidden),
                training=training)
            hidden = self.up(
                UpdateInput(hidden=hidden, hidden_initial=hidden_initial, messages=messages),
                training=training)
        y = self.ro(ReadoutInput(hidden=hidden, hidden_initial=hidden_initial), training=training)
        return y

    def get_config(self):
        return {**{
            "hidden_state_size": self.hidden_state_size,
            "message_passing_iterations": self.message_passing_iterations,
            "message_size": self.message_size,
            "output_size": self.output_size,
            "initializer": self.initializer,
            "message_passing": self.message_passing,
            "update": self.update,
            "readout": self.readout,
        }, **self.kwargs}
