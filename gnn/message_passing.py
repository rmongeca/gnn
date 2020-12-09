"""Message Passing Layer for GNN implementation."""
import tensorflow as tf
from abc import ABC, abstractmethod

from gnn.input import MessagePassingInput, MessageFunctionInput


class MessagePassingLayer(tf.keras.layers.Layer, ABC):
    """Abstract Message Passing layer for GNN model.

    In order to use this abstract class, the user has to define the aggregation and message
    methods.
    """

    def __init__(self, hidden_state_size=10, message_size=100, *args, **kwargs):
        super(MessagePassingLayer, self).__init__(*args, **kwargs)
        self.hidden_state_size = hidden_state_size
        self.message_size = message_size

    def call(self, inputs: MessagePassingInput, training=False):
        batch_size = tf.shape(inputs.hidden)[0]
        return tf.map_fn(
            fn=lambda batch: self._handle_batch(batch, inputs=inputs, training=training),
            elems=tf.range(batch_size),
            fn_output_signature=tf.TensorSpec([None, self.message_size], float)
        )

    def _handle_batch(self, batch, inputs: MessagePassingInput, training=False):
        num_nodes = tf.shape(inputs.hidden)[1]
        inputs = MessagePassingInput(
            edge_features=inputs.edge_features[batch, :, :],
            edge_sources=inputs.edge_sources[batch, :],
            edge_targets=inputs.edge_targets[batch, :],
            hidden=inputs.hidden[batch, :, :]
        )
        # Iterate over all nodes for message passing phase
        return tf.map_fn(
            fn=lambda node: self._message_passing(node, inputs, training=training),
            elems=tf.range(num_nodes),
            fn_output_signature=tf.TensorSpec([self.message_size], float)
        )

    def _message_passing(self, node_id, inputs: MessagePassingInput, training=False):
        # Get edges for message passing
        edge_list = tf.where(inputs.edge_targets == node_id)
        edges = tf.gather_nd(params=inputs.edge_features, indices=edge_list,
                             name="message-passing-node-edges")
        if tf.rank(edges) < 2:  # Ensure tensor rank is 2
            tf.expand_dims(edges, axis=0)
        # Get neighbours for message passing
        source_nodes = tf.gather(inputs.edge_sources, indices=edge_list)
        neighbours = tf.gather_nd(params=inputs.hidden, indices=source_nodes,
                                  name="message-passing-node-neighbours")
        if tf.rank(neighbours) < 2:  # Ensure tensor rank is 2
            tf.expand_dims(neighbours, axis=0)
        # Get node for message passing
        node = tf.gather_nd(params=inputs.hidden, indices=[node_id])
        messages = self.message(
            MessageFunctionInput(edges=edges, neighbours=neighbours, node=node), training=training)
        return self.aggregation(messages, training=training)

    @abstractmethod
    def message(self, inputs: MessageFunctionInput, training=False):
        """Abstract Message creation function from the edge and neighbour information for a message
        passing phase."""
        pass

    @abstractmethod
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
        return tf.concat([inputs.neighbours, inputs.edges], axis=-1)

    def aggregation(self, messages, training=False):
        return tf.math.reduce_sum(messages, axis=0)


class EdgeNetMessagePassing(MessagePassingLayer):
    """Edge Network Message Passing layer for GNN model.

    This Message Passing Layer builds a message for node n_v from another node n_w, with edge e_vw
    by feeding the edge to an Edge Neural Network A which outputs a matrix of size message_size x
    hidden_size that is later applied to the hidden state of n_w, h_w. That is:
    m_vw = (A·e_vw)·h_w
    then aggregates the messages for n_v summing over them: sum(m_vw for w in N(v)).
    """

    def __init__(self, hidden_state_size=10, message_size=100,
                 edge_num_layers=4, edge_hidden_dimension=50, edge_activation="relu",
                 *args, **kwargs):
        super(EdgeNetMessagePassing, self).__init__(hidden_state_size, message_size,
                                                    *args, **kwargs)
        self.edge_net_output_size = message_size * hidden_state_size
        self.edge_net = tf.keras.Sequential(name="message-passing-edge-network")
        for num in range(edge_num_layers):
            self.edge_net.add(tf.keras.layers.Dense(units=edge_hidden_dimension,
                                                    activation=edge_activation,
                                                    name=f"edge-network-dense-{num + 1}"))
        self.edge_net.add(tf.keras.layers.Dense(units=self.edge_net_output_size,
                                                activation=edge_activation,
                                                name="edge-network-dense-output"))
        self.reshape = tf.keras.layers.Reshape(
            target_shape=(message_size, hidden_state_size),
            name="message-passing-edge-state-reshape"
        )

    def message(self, inputs: MessageFunctionInput, training=False):
        edge_network = self.edge_net(inputs.edges, training=training)
        edge_network = self.reshape(edge_network, training=training)
        return tf.linalg.matvec(edge_network, inputs.neighbours)

    def aggregation(self, messages, training=False):
        return tf.math.reduce_sum(messages, axis=0)
