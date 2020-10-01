"""Message Passing Neural Network implementation"""
import tensorflow as tf
from typing import Dict, Tuple, NamedTuple


class GraphInput(NamedTuple):
    """Input named tuple for the MessagePassingNet."""

    node_features: tf.Tensor
    edge_features: tf.Tensor
    adjacency_lists: Tuple[tf.Tensor, ...]
    edge_to_idx: Dict


class GraphInputShapes(NamedTuple):
    """Input Shape named tuple for the MessagePassingNet."""

    node_features: tf.TensorShape
    edge_features: tf.TensorShape


class MessageLayer(tf.keras.layers.Layer):
    """Default Message function layer for Message Passing Network."""

    def __init__(self, hidden_state_size=10, message_size=100,
                 aggregation=lambda x: tf.math.reduce_sum(x, axis=0)):
        super(MessageLayer, self).__init__()
        self.hidden_state_size = hidden_state_size
        self.message_size = message_size
        self.aggregation = aggregation
        self.edge_net = tf.keras.layers.Dense(units=message_size*hidden_state_size)

    def call(self, inputs: GraphInput, hidden: tf.Tensor, initial: tf.Tensor, training=None):
        num_nodes = inputs.node_features.shape[0]
        return tf.map_fn(
            fn=lambda x: self._call(x, inputs, hidden, training=training),
            elems=tf.range(num_nodes, dtype=float))

    def _call(self, node_idx, inputs: GraphInput, hidden: tf.Tensor, training=None):
        node_idx = int(node_idx)
        edge_to_idx = inputs.edge_to_idx
        adjacency_list = inputs.adjacency_lists[node_idx]
        edges = tf.gather(params=inputs.edge_features,
                          indices=self._get_edge_idx(node_idx, adjacency_list, edge_to_idx))
        neighbours = tf.gather(params=hidden, indices=adjacency_list)
        return self.aggregation(self._mp(neighbours, edges, training=training))

    def call_edge(self, edge: tf.Tensor, neighbour: tf.Tensor):
        _edge = tf.reshape(edge, (1, edge.shape[0]))
        _neighbour = tf.reshape(neighbour, (self.hidden_state_size,))
        edge_mat = tf.reshape(self.edge_net(_edge), (self.message_size, self.hidden_state_size))
        return tf.linalg.matvec(edge_mat, _neighbour)

    def _get_edge_idx(self, node_idx, adjacency_list: tf.Tensor, edge_to_idx: Dict):
        return tf.map_fn(
            fn=lambda i: edge_to_idx["{}-{}".format(*sorted([node_idx, int(i)]))],
            elems=adjacency_list
        )

    def _mp(self, neighbours: tf.Tensor, edges: tf.Tensor, training=None) -> tf.Tensor:
        num_edges = edges.shape[0]
        return tf.map_fn(
            fn=lambda i: self.call_edge(edges[i], neighbours[i]),
            elems=tf.range(num_edges),
            fn_output_signature=float
        )


class UpdateLayer(tf.keras.layers.Layer):
    """Default Update function layer for Message Passing Network."""

    def __init__(self, hidden_state_size=10, message_size=100):
        super(UpdateLayer, self).__init__()
        self.hidden_state_size = hidden_state_size
        self.message_size = message_size
        self.up = tf.keras.layers.GRU(units=self.hidden_state_size)

    def build(self, input_shapes: GraphInputShapes):
        input_shape = self.hidden_state_size + self.message_size
        self.up.build(input_shape=(1, input_shape))

    def call(self, messages, hidden, initial, training=None):
        _inputs = tf.concat([hidden, messages], axis=1)
        up_inputs = tf.reshape(_inputs, (_inputs.shape[0], _inputs.shape[1], 1))
        return self.up(up_inputs, training=training)


class ReadoutLayer(tf.keras.layers.Layer):
    """Readout Message function layer for Message Passing Network."""

    def __init__(self, hidden_state_size=10, outputs=1):
        super(ReadoutLayer, self).__init__()
        self.hidden_state_size = hidden_state_size
        self._i = tf.keras.layers.Dense(units=outputs)
        self._j = tf.keras.layers.Dense(units=outputs)

    def build(self, input_shapes: GraphInputShapes):
        self._i.build(input_shape=self.hidden_state_size*2)
        self._j.build(input_shape=self.hidden_state_size)

    def call(self, hidden: tf.Tensor, hidden0: tf.Tensor, training=None):
        inputs = tf.concat([hidden, hidden0], axis=1)
        i = self._i(inputs, training=training)
        j = self._j(hidden, training=training)
        ij = tf.math.multiply(i, j)
        return tf.math.reduce_sum(ij, axis=0)


class Initial(tf.keras.layers.Layer):
    """Initial hidden state layer for Message Passing Network."""

    def __init__(self, hidden_state_size=10):
        super(Initial, self).__init__()
        self.hidden_state_size = hidden_state_size
        self.init = tf.keras.layers.Dense(units=self.hidden_state_size)

    def build(self, input_shapes: GraphInputShapes):
        node_features = input_shapes.node_features[1]
        self.init.build(input_shape=node_features)

    def call(self, inputs: GraphInput, training=None):
        node_features = inputs.node_features
        return self.init(node_features, training=training)


class MessagePassingNet(tf.keras.Model):
    """General class for Message Passing Neural Network.

    Parameters
    ----------
    hidden_state_size : int
        Size of the hidden state vector for each node of the net. Defaults to 10.
    mp_iterations : int
        Number of iterations for message passing phase of the net. Defaults to 3.
    mp_agregation: tf.function
        Function to use for message aggregation in message passing phase. Defaults to
        tf.math.reduce_sum.
    message_layer : tf.keras.layers.Layer
    message_params : dict
    update_layer : tf.keras.layers.Layer
    update_params : dict
    readout_layer : tf.keras.layers.Layer
    readout_params : dict
    """

    def __init__(self, hidden_state_size=10, message_size=20, mp_iterations=3,
                 mp_aggregation=lambda x: tf.math.reduce_sum(x, axis=0),
                 message_layer=MessageLayer, update_layer=UpdateLayer, readout_layer=ReadoutLayer,
                 initial=Initial, **kwargs):
        super(MessagePassingNet, self).__init__()
        # Record arguments
        self.hidden_state_size = hidden_state_size
        self.message_size = message_size
        self.mp_iterations = mp_iterations
        self.mp_aggregation = mp_aggregation
        # Inner layer class setting
        self.message_layer = message_layer
        self.update_layer = update_layer
        self.readout_layer = readout_layer
        self.initial = initial
        # Inner layers
        self.init = self.initial(hidden_state_size=self.hidden_state_size)
        self.mp = self.message_layer(hidden_state_size=self.hidden_state_size,
                                     message_size=self.message_size,
                                     aggregation=self.mp_aggregation)
        self.up = self.update_layer(hidden_state_size=self.hidden_state_size,
                                    message_size=self.message_size)
        self.ro = self.readout_layer(hidden_state_size=self.hidden_state_size)

    def build(self, input_shapes: GraphInputShapes):
        self.init.build(input_shapes)
        # self.mp.build(input_shapes)
        self.up.build(input_shapes)
        self.ro.build(input_shapes)

    def call(self, inputs: GraphInput, training=None, mask=None) -> tf.Tensor:
        hidden = self.init(inputs)
        hidden0 = tf.identity(hidden)
        for _ in range(self.mp_iterations):
            messages = self.mp(inputs, hidden, hidden0)
            hidden = self.up(messages, hidden, hidden0)
        y = self.ro(hidden, hidden0)
        return y

    def train_step(self, data):
        # Unpack the data. Its structure depends on your model and
        # on what you pass to `fit()`.
        graph, y = data

        with tf.GradientTape() as tape:
            y_pred = self(graph, training=True)  # Forward pass
            # Compute the loss value
            # (the loss function is configured in `compile()`)
            loss = self.compiled_loss(y, y_pred, regularization_losses=self.losses)

        # Compute gradients
        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)
        # Update weights
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))
        # Update metrics (includes the metric that tracks the loss)
        self.compiled_metrics.update_state(y, y_pred)
        # Return a dict mapping metric names to current value
        return {m.name: m.result() for m in self.metrics}

