"""Message Passing Layer for GNN implementation."""
import tensorflow as _tf
from abc import ABC as _ABC, abstractmethod as _abstractmethod

from .input import MessagePassingInput, MessageFunctionInput
from .mlp import MLP


class MessagePassingLayer(_tf.keras.layers.Layer, _ABC):
    """Abstract Message Passing layer for GNN model.

    Abstract class to define handling of Batched MessagePassingInput for message passing layers.
    Child classes must implement the message method, which takes as argument a non-batched
    MessagePassingInput and generates messages, and the aggregate method, which takes non-batched
    messages tensor and returns the aggregated messages.
    """

    def __init__(self, hidden_state_size=10, message_size=100, *args, **kwargs):
        super(MessagePassingLayer, self).__init__(*args, **kwargs)
        self.hidden_state_size = hidden_state_size
        self.message_size = message_size

    def call(self, inputs: MessagePassingInput, training=False):
        batch_size = _tf.shape(inputs.hidden)[0]
        return _tf.map_fn(
            fn=lambda batch: self._handle_batch(batch, inputs=inputs, training=training),
            elems=_tf.range(batch_size),
            fn_output_signature=_tf.TensorSpec([None, self.message_size], float)
        )

    def _handle_batch(self, batch, inputs: MessagePassingInput, training=False):
        num_nodes = _tf.shape(inputs.hidden)[1]
        inputs = MessagePassingInput(
            edge_features=inputs.edge_features[batch, :, :],
            edge_sources=inputs.edge_sources[batch, :],
            edge_targets=inputs.edge_targets[batch, :],
            hidden=inputs.hidden[batch, :, :]
        )
        # Iterate over all nodes for message passing phase
        return _tf.map_fn(
            fn=lambda node: self._message_passing(node, inputs, training=training),
            elems=_tf.range(num_nodes),
            fn_output_signature=_tf.TensorSpec([self.message_size], float)
        )

    def _message_passing(self, node_id, inputs: MessagePassingInput, training=False):
        # Get edges for message passing
        edge_list = _tf.where(inputs.edge_targets == node_id)
        edges = _tf.gather_nd(params=inputs.edge_features, indices=edge_list,
                              name="mp-node-edges")
        # Get neighbours for message passing
        source_nodes = _tf.gather(params=inputs.edge_sources, indices=edge_list)
        neighbours = _tf.gather_nd(params=inputs.hidden, indices=source_nodes,
                                   name="mp-node-neighbours")
        # Get node for message passing
        node = _tf.gather_nd(params=inputs.hidden, indices=[[node_id]])
        messages = self.message(
            MessageFunctionInput(edges=edges, neighbours=neighbours, node=node), training=training)
        return self.aggregation(messages, training=training)

    @_abstractmethod
    def message(self, inputs: MessageFunctionInput, training=False):
        """Abstract Message creation function from the edge and neighbour information for a message
        passing phase."""
        pass

    @_abstractmethod
    def aggregation(self, messages, training=False):
        """Abstract Aggregation function to merge all messages for a message passing phase."""
        pass


class ConcatenationMessage(MessagePassingLayer):
    """Concatenation Message Passing layer for GNN model.

    This Message Passing Layer builds a message for node n_v from another node n_w, with edge e_vw
    by concatenating (n_w, e_vw). Then all messages from n_v's neighbours are aggregated by
    summing the messages.
    """

    def message(self, inputs: MessageFunctionInput, training=False):
        return _tf.concat([inputs.neighbours, inputs.edges], axis=-1)

    def aggregation(self, messages, training=False):
        return _tf.math.reduce_sum(messages, axis=0)


class ConcatenateWithDesinationMessage(MessagePassingLayer):
    """Concatenation With Source Message Passing layer for GNN model.

    This Message Passing Layer builds a message for node n_v from another node n_w, with edge e_vw
    by concatenating (n_v, n_w, e_vw). Then all messages from n_v's neighbours are aggregated by
    summing the messages.
    """

    def message(self, inputs: MessageFunctionInput, training=False):
        node = _tf.repeat(
            inputs.node, _tf.shape(inputs.neighbours)[0], axis=0)
        return _tf.concat([node, inputs.neighbours, inputs.edges], axis=-1)

    def aggregation(self, messages, training=False):
        return _tf.math.reduce_sum(messages, axis=0)


class EdgeNetMessage(MessagePassingLayer):
    """Edge Network Message Passing layer for GNN model.

    This Message Passing Layer builds a message for node n_v from another node n_w, with edge e_vw
    by feeding the edge to an Edge Neural Network A which outputs a matrix of size message_size x
    hidden_size that is later applied to the hidden state of n_w, h_w. That is:
    m_vw = (A·e_vw)·h_w
    then aggregates the messages for n_v summing over them: sum(m_vw for w in N(v)).
    """

    def __init__(
        self, hidden_state_size=10, message_size=10, num_layers=4, units=50, activation="relu",
        *args, **kwargs
    ):
        super(EdgeNetMessage, self).__init__(hidden_state_size, message_size, *args, **kwargs)
        self.edge_net = MLP(
            activation=activation, name="mp-edge-net", num_layers=num_layers, units=units,
            output_units=(message_size * hidden_state_size), **kwargs)
        self.edge_net.add(_tf.keras.layers.Reshape(
            target_shape=(message_size, hidden_state_size), name="mp-edge-net-reshape", **kwargs
        ))

    def message(self, inputs: MessageFunctionInput, training=False):
        edge_network = self.edge_net(inputs.edges, training=training)
        return _tf.linalg.matvec(edge_network, inputs.neighbours)

    def aggregation(self, messages, training=False):
        return _tf.math.reduce_sum(messages, axis=0)


class FeedForwardMessage(MessagePassingLayer):
    """Feed Forward Message Passing layer for GNN model.

    This Message Passing Layer builds a message for node n_v from another node n_w, with edge e_vw
    by feeding to a feed-forward neural network the concatenation of the source and target hidden
    state along with the edge feature vector; then aggregates the messages for n_v summing over
    them: sum(m_vw for w in N(v)).
    """

    def __init__(
        self, hidden_state_size=10, message_size=10, activation="relu", layer=None,
        num_layers=4, output_activation=None, units=50, aggregation_fn=None, *args, **kwargs
    ):
        super(FeedForwardMessage, self).__init__(
            hidden_state_size=hidden_state_size, message_size=message_size, *args, **kwargs)
        self.hidden_state_size = hidden_state_size
        self.ff_net = MLP(
            activation=activation, layer=layer, name="mp-ff-net", num_layers=num_layers,
            output_activation=output_activation, output_units=message_size, units=units, **kwargs)
        self.aggregation_fn = _tf.math.reduce_sum if aggregation_fn is None else aggregation_fn

    def build(self, input_shapes):
        concatenated_dims = self.hidden_state_size * 2 + input_shapes.edge_features[-1]
        ff_shape = _tf.TensorShape([None, concatenated_dims])
        self.ff_net.build(ff_shape)
        super(FeedForwardMessage, self).build([])

    def message(self, inputs: MessageFunctionInput, training=False):
        node = _tf.repeat(
            inputs.node, _tf.shape(inputs.neighbours)[0], axis=0)
        ff_inputs = _tf.concat([node, inputs.neighbours, inputs.edges], axis=-1)
        return self.ff_net(ff_inputs, training=training)

    def aggregation(self, messages, training=False):
        return self.aggregation_fn(messages, axis=0)
