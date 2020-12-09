"""Graph Input named tuple implementation"""
import json
import tensorflow as tf
from typing import NamedTuple, Union


class GNNInput(NamedTuple):
    """Input named tuple for the GNN."""
    edge_features: Union[tf.Tensor, tf.TensorShape]
    edge_sources: Union[tf.Tensor, tf.TensorShape]
    edge_targets: Union[tf.Tensor, tf.TensorShape]
    node_features: Union[tf.Tensor, tf.TensorShape]

    @classmethod
    def get_data_generator(cls, files, target):
        """Define a class generator to form a tf.data.Dataset, and return generator and output
        types and shapes.
        """
        with open(files[0], "r") as _fp:
            _graph = json.load(_fp)
        _graph.pop("edge_sources")
        _graph.pop("edge_targets")
        num_edge_features = len([key for key in _graph.keys() if "edge_" in key])
        num_node_features = len([key for key in _graph.keys() if "node_" in key])
        output_types = (GNNInput(tf.float32, tf.int32, tf.int32, tf.float32), tf.float32)
        output_shapes = (
            GNNInput(
                tf.TensorShape([None, num_edge_features]),
                tf.TensorShape([None]), tf.TensorShape([None]),
                tf.TensorShape([None, num_node_features]),
            ), tf.TensorShape([1])
        )

        def get_data():
            """Data generator for graph JSON files into GNNInput named tuples together with target
            tensor.
            """

            for fn in files:
                with open(fn, "r") as fp:
                    graph = json.load(fp)
                edge_sources = tf.squeeze(tf.constant(graph.pop("edge_sources")))
                edge_targets = tf.squeeze(tf.constant(graph.pop("edge_targets")))
                edge_features = tf.squeeze(tf.stack(
                    [tf.constant(values, dtype=float) for key, values in graph.items() if
                     "edge_" in key],
                    axis=1
                ))
                node_features = tf.squeeze(tf.stack(
                    [tf.constant(values, dtype=float) for key, values in graph.items() if
                     "node_" in key],
                    axis=1
                ))
                data = GNNInput(
                    edge_features=edge_features,
                    edge_sources=edge_sources,
                    edge_targets=edge_targets,
                    node_features=node_features,
                )
                y = tf.constant([graph[target]])
                yield data, y
        return {
            "generator": get_data,
            "output_types": output_types,
            "output_shapes": output_shapes
        }


class MessagePassingInput(NamedTuple):
    """Input named tuple for the MessagePassing layers."""
    edge_features: Union[tf.Tensor, tf.TensorShape]
    edge_sources: Union[tf.Tensor, tf.TensorShape]
    edge_targets: Union[tf.Tensor, tf.TensorShape]
    hidden: Union[tf.Tensor, tf.TensorShape]


class MessageFunctionInput(NamedTuple):
    """Input named tuple for the Message function inside MessagePassing layers."""
    edges: Union[tf.Tensor, tf.TensorShape]
    neighbours: Union[tf.Tensor, tf.TensorShape]
    node: Union[tf.Tensor, tf.TensorShape]


class UpdateInput(NamedTuple):
    """Input named tuple for the Update layers."""
    hidden: Union[tf.Tensor, tf.TensorShape]
    hidden_initial: Union[tf.Tensor, tf.TensorShape]
    messages: Union[tf.Tensor, tf.TensorShape]


class ReadoutInput(NamedTuple):
    """Input named tuple for the Readout layers."""
    hidden: Union[tf.Tensor, tf.TensorShape]
    hidden_initial: Union[tf.Tensor, tf.TensorShape]
