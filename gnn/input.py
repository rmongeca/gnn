"""Graph Input named tuple implementation"""
import json as _json
import networkx as _nx
import numpy as _np
import tensorflow as _tf
from os import PathLike as _PathLike
from typing import List as _List, NamedTuple as _NamedTuple, Union as _Union


class GNNInput(_NamedTuple):
    """Input named tuple for the GNN."""
    edge_features: _Union[_tf.Tensor, _tf.TensorShape]
    edge_sources: _Union[_tf.Tensor, _tf.TensorShape]
    edge_targets: _Union[_tf.Tensor, _tf.TensorShape]
    node_features: _Union[_tf.Tensor, _tf.TensorShape]

    @classmethod
    def get_data_generator(
        cls, files: _List[_PathLike], node_feature_names: _List[str],
        edge_feature_names: _List[str], target: _Union[str, _List[str]]
    ):
        """Define a class generator to form a _tf.data.Dataset, and return generator and output
        types and shapes.
        """
        if isinstance(target, str):
            target = [target]
        num_edge_features = len(edge_feature_names)
        num_node_features = len(node_feature_names)
        output_size = len(target)
        output_types = (GNNInput(_tf.float32, _tf.int32, _tf.int32, _tf.float32), _tf.float32)
        output_shapes = (
            GNNInput(
                edge_features=_tf.TensorShape([None, num_edge_features]),
                edge_sources=_tf.TensorShape([None]), edge_targets=_tf.TensorShape([None]),
                node_features=_tf.TensorShape([None, num_node_features]),
            ), _tf.TensorShape([output_size])
        )

        def get_data():
            """Data generator for graph json files into GNNInput named tuples together with target
            tensor.
            """
            while True:
                for fn in files:
                    with open(fn, "r") as fp:
                        sample = _json.load(fp)
                    graph = _nx.readwrite.json_graph.node_link_graph(sample)
                    edges = graph.edges(data=True)
                    edge_features = _tf.squeeze(_tf.constant(_np.array([
                        [v for k, v in edge.items() if k in edge_feature_names]
                        for _, _, edge in edges
                    ])))
                    edge_sources = _tf.squeeze(_tf.constant(_np.array(
                        [src for src, _, _ in edges]
                    )))
                    edge_targets = _tf.squeeze(_tf.constant(_np.array(
                        [tgt for _, tgt, _ in edges]
                    )))
                    node_features = _tf.squeeze(_tf.constant(_np.array([
                        [v for k, v in node.items() if k in node_feature_names]
                        for node in dict(graph.nodes(data=True)).values()
                    ])))
                    data = GNNInput(
                        edge_features=edge_features,
                        edge_sources=edge_sources,
                        edge_targets=edge_targets,
                        node_features=node_features,
                    )
                    y = _tf.constant(_np.array([
                        graph.graph[_target] for _target in target
                    ]))
                    yield data, y
        return {
            "generator": get_data,
            "output_types": output_types,
            "output_shapes": output_shapes
        }


class MessagePassingInput(_NamedTuple):
    """Input named tuple for the MessagePassing layers."""
    edge_features: _Union[_tf.Tensor, _tf.TensorShape]
    edge_sources: _Union[_tf.Tensor, _tf.TensorShape]
    edge_targets: _Union[_tf.Tensor, _tf.TensorShape]
    hidden: _Union[_tf.Tensor, _tf.TensorShape]


class MessageFunctionInput(_NamedTuple):
    """Input named tuple for the Message function inside MessagePassing layers."""
    edges: _Union[_tf.Tensor, _tf.TensorShape]
    neighbours: _Union[_tf.Tensor, _tf.TensorShape]
    node: _Union[_tf.Tensor, _tf.TensorShape]


class UpdateInput(_NamedTuple):
    """Input named tuple for the Update layers."""
    hidden: _Union[_tf.Tensor, _tf.TensorShape]
    hidden_initial: _Union[_tf.Tensor, _tf.TensorShape]
    messages: _Union[_tf.Tensor, _tf.TensorShape]


class ReadoutInput(_NamedTuple):
    """Input named tuple for the Readout layers."""
    hidden: _Union[_tf.Tensor, _tf.TensorShape]
    hidden_initial: _Union[_tf.Tensor, _tf.TensorShape]
