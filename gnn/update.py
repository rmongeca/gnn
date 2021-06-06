"""Update Layer for GNN implementation."""
import tensorflow as _tf
from abc import ABC as _ABC, abstractmethod as _abstractmethod

from .input import UpdateInput
from .mlp import MLP


class UpdateLayer(_tf.keras.layers.Layer, _ABC):
    """Base Update Layer abstract class to be inherited by Update Layer implementations.

    Abstract class to define handling of Batched UpdateInput for update layer to perform Update.
    Child classes must implement the update method, which takes as argument a non-batched flattened
    UpdateInput.
    """
    def __init__(self, hidden_state_size=10, message_size=10, *args, **kwargs):
        """Update Layer abstract class constructor."""
        super(UpdateLayer, self).__init__(*args, **kwargs)
        self.hidden_state_size = hidden_state_size
        self.message_size = message_size

    def call(self, inputs: UpdateInput, training=None):
        # Flatten batch-nodes
        batch_size = _tf.shape(inputs.hidden)[0]
        num_nodes = _tf.shape(inputs.hidden)[1]
        flattened_size = batch_size * num_nodes
        messages = _tf.reshape(inputs.messages, [flattened_size, self.message_size])
        hidden = _tf.reshape(inputs.hidden, [flattened_size, self.hidden_state_size])
        hidden_initial = _tf.reshape(
            inputs.hidden_initial, [flattened_size, self.hidden_state_size])
        # Call update function
        new_hidden = self.update(
            UpdateInput(
                hidden=hidden,
                hidden_initial=hidden_initial,
                messages=messages,
            ), training=training
        )
        # Restore batches
        return _tf.reshape(new_hidden, [batch_size, num_nodes, self.hidden_state_size])

    @_abstractmethod
    def update(self, inputs: UpdateInput, training=None):
        """Update method to apply to a certain flattened UpdateInput."""
        pass


class FeedForwardUpdate(UpdateLayer):
    """Update function layer with a Feed Forward NN for GNN model."""

    def __init__(
        self, hidden_state_size=10, message_size=10, activation="relu", layer=None,
        num_layers=4, output_activation=None, units=50, *args, **kwargs
    ):
        super(FeedForwardUpdate, self).__init__(
            hidden_state_size=hidden_state_size, message_size=message_size, *args, **kwargs)
        self.mlp = MLP(
            activation=activation, layer=layer, name="update-ff-net", num_layers=num_layers,
            output_activation=output_activation, output_units=hidden_state_size, units=units,
            **kwargs)

    def build(self, input_shapes):
        build_shapes = input_shapes.hidden[-1] + input_shapes.messages[-1]
        self.mlp.build(_tf.TensorShape([None, build_shapes]))
        super(UpdateLayer, self).build([])

    def update(self, inputs: UpdateInput, training=None):
        hidden = inputs.hidden
        messages = inputs.messages
        _input = _tf.concat([hidden, messages], axis=-1)
        return self.mlp(_input, training=training)


class GRUUpdate(UpdateLayer):
    """Update function layer with GRU for GNN model."""

    def __init__(self, hidden_state_size=10, message_size=10, *args, **kwargs):
        super(GRUUpdate, self).__init__(
            hidden_state_size=hidden_state_size, message_size=message_size, *args, **kwargs)
        self.gru = _tf.keras.layers.GRUCell(units=hidden_state_size, name="update-gru")

    def update(self, inputs: UpdateInput, training=None):
        hidden = inputs.hidden
        messages = inputs.messages
        new_hidden, _ = self.gru(messages, states=hidden, training=training)
        return new_hidden
