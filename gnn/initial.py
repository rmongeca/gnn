"""Initial layer for GNN to construct hidden states."""
import tensorflow as tf


class DenseInitializer(tf.keras.layers.Layer):
    """Initial hidden state layer for Message Passing Network."""

    def __init__(self, hidden_state_size=10, *args, **kwargs):
        super(DenseInitializer, self).__init__(*args, **kwargs)
        self.hidden_state_size = hidden_state_size
        # Init inner layers
        self.init = tf.keras.layers.Dense(units=hidden_state_size, name="initializer-dense")

    def build(self, input_shape):
        self.init.build(input_shape)
        super(DenseInitializer, self).build([])

    def call(self, node_features, training=None):
        # Flatten batch-nodes
        batch_size = tf.shape(node_features)[0]
        num_nodes = tf.shape(node_features)[1]
        num_dims = tf.shape(node_features)[2]
        nodes = tf.reshape(node_features, [batch_size*num_nodes, num_dims])
        # Init hidden states
        hidden = self.init(nodes, training=training)
        # Restore batches
        return tf.reshape(hidden, [batch_size, num_nodes, self.hidden_state_size])
