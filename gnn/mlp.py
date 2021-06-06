"""Multi Layer Perceptron base layer for use in GNN layers."""
import tensorflow as _tf


def MLP(
    activation="relu", layer=None, name="mlp", num_layers=2, output_activation=None,
    output_layer=None, output_units=None, units=50, **kwargs
):
    """Create a (Sequential) Multi Layer Perceptron with dense layers."""
    assert isinstance(num_layers, int) and num_layers > 0
    if layer is None:
        layer = _tf.keras.layers.Dense

    # If layer is a string, repeat for each non-output layer
    if not isinstance(layer, list):
        layer = [layer] * (num_layers - 1) if num_layers > 1 else [layer]
    if output_layer is None:
        output_layer = layer[-1]

    # If activation is a string, repeat for each non-output layer
    if not isinstance(activation, list):
        activation = [activation] * (num_layers - 1) if num_layers > 1 else [activation]

    # If units is a number, repeat for each non-output layer
    if not isinstance(units, list):
        units = [units] * (num_layers - 1) if num_layers > 1 else [units]
    if output_units is None:
        output_units = units[-1]

    model = _tf.keras.Sequential(name=name)
    for i in range(num_layers-1):
        model.add(
            layer[i](
                units=units[i], activation=activation[i], name=f"{name}-dense-{i}", **kwargs
            )
        )
    model.add(
        output_layer(units=output_units, activation=output_activation, name=f"{name}-output")
        if output_layer != _tf.keras.layers.Activation else
        output_layer(activation=output_activation, name=f"{name}-output")
    )
    return model
