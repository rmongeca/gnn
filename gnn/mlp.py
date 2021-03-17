"""Multi Layer Perceptron base layer for use in GNN layers."""
import tensorflow as _tf


def MLP(
    activation="relu", layer=None, name="mlp", num_layers=2, output_units=None,
    output_activation=None, units=50, **kwargs
):
    """Create a (Sequential) Multi Layer Perceptron with dense layers."""
    if layer is None:
        layer = _tf.keras.layers.Dense
    if output_units is None:
        output_units = units
    model = _tf.keras.Sequential(name=name)
    for i in range(1, num_layers):
        model.add(layer(units=units, activation=activation, name=f"{name}-dense-{i}", **kwargs))
    model.add(layer(units=output_units, activation=output_activation, name=f"{name}-output"))
    return model
