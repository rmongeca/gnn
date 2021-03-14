"""Graph Input named tuple implementation"""
import json as _json
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
    def get_data_generator(cls, files: _List[_PathLike], target: _Union[str, _List[str]]):
        """Define a class generator to form a _tf.data.Dataset, and return generator and output
        types and shapes.
        """
        # Take first file to infer GNNInput properties for _tf.data.Dataset
        with open(files[0], "r") as _fp:
            _graph = _json.load(_fp)
        _graph.pop("edge_sources")
        _graph.pop("edge_targets")
        num_edge_features = len([key for key in _graph.keys() if "edge_" in key])
        num_node_features = len([key for key in _graph.keys() if "node_" in key])
        output_size = len(target) if isinstance(target, _List) else 1
        output_types = (GNNInput(_tf.float32, _tf.int32, _tf.int32, _tf.float32), _tf.float32)
        output_shapes = (
            GNNInput(
                edge_features=_tf.TensorShape([None, num_edge_features]),
                edge_sources=_tf.TensorShape([None]), edge_targets=_tf.TensorShape([None]),
                node_features=_tf.TensorShape([None, num_node_features]),
            ), _tf.TensorShape([output_size])
        )

        def get_data():
            """Data generator for graph _json files into GNNInput named tuples together with target
            tensor.
            """
            while True:
                for fn in files:
                    with open(fn, "r") as fp:
                        graph = _json.load(fp)
                    edge_sources = _tf.squeeze(_tf.constant(graph.pop("edge_sources")))
                    edge_targets = _tf.squeeze(_tf.constant(graph.pop("edge_targets")))
                    edge_features = _tf.squeeze(_tf.stack(
                        [_tf.constant(values, dtype=float) for key, values in graph.items() if
                         "edge_" in key],
                        axis=1
                    ))
                    node_features = _tf.squeeze(_tf.stack(
                        [
                            _tf.constant(values, dtype=float)
                            for key, values in graph.items() if "node_" in key
                        ],
                        axis=1
                    ))
                    data = GNNInput(
                        edge_features=edge_features,
                        edge_sources=edge_sources,
                        edge_targets=edge_targets,
                        node_features=node_features,
                    )
                    y = _tf.squeeze(_tf.stack(
                        [_tf.constant(graph[_target], dtype=float) for _target in target],
                        axis=0
                    )) if isinstance(target, _List) else _tf.constant([graph[target]])
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
