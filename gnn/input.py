"""Graph Input named tuple implementation"""
import json as _json
import networkx as _nx
import numpy as _np
import tensorflow as _tf
from pathlib import Path as _Path
from typing import List as _List, NamedTuple as _NamedTuple, Union as _Union


_TfTypesUnion = _Union[_tf.Tensor, _tf.TensorShape, _tf.TensorSpec]


class GNNInput(_NamedTuple):
    """Input named tuple for the GNN."""
    edge_features: _TfTypesUnion
    edge_sources: _TfTypesUnion
    edge_targets: _TfTypesUnion
    node_features: _TfTypesUnion
    additional_inputs: _TfTypesUnion


class MessagePassingInput(_NamedTuple):
    """Input named tuple for the MessagePassing layers."""
    edge_features: _TfTypesUnion
    edge_sources: _TfTypesUnion
    edge_targets: _TfTypesUnion
    hidden: _TfTypesUnion


class MessageFunctionInput(_NamedTuple):
    """Input named tuple for the Message function inside MessagePassing layers."""
    edges: _TfTypesUnion
    neighbours: _TfTypesUnion
    node: _TfTypesUnion


class UpdateInput(_NamedTuple):
    """Input named tuple for the Update layers."""
    hidden: _TfTypesUnion
    hidden_initial: _TfTypesUnion
    messages: _TfTypesUnion


class ReadoutInput(_NamedTuple):
    """Input named tuple for the Readout layers."""
    hidden: _TfTypesUnion
    hidden_initial: _TfTypesUnion


def get_dataset_from_files(
    files: _List[_Path], node_feature_names: _List[str], edge_feature_names: _List[str],
    target: _Union[str, _List[str]], additional_inputs_names: _List[str] = [], target_shapes=None,
    batch_size=1, local=False
):
    """Define a class generator to form a _tf.data.Dataset, and return generator and output
    types and shapes.
    """
    if isinstance(target, str):
        target = [target]
    num_edge_features = len(edge_feature_names)
    num_node_features = len(node_feature_names)
    num_additional_inputs = len(additional_inputs_names)
    output_size = len(target) if local else None
    if target_shapes is None:
        target_shapes = _tf.TensorShape([output_size])
    output_types = (
        GNNInput(_tf.float32, _tf.int32, _tf.int32, _tf.float32, _tf.float32), _tf.float32
    )
    output_shapes = (
        GNNInput(
            edge_features=_tf.TensorShape([None, num_edge_features]),
            edge_sources=_tf.TensorShape([None]), edge_targets=_tf.TensorShape([None]),
            node_features=_tf.TensorShape([None, num_node_features]),
            additional_inputs=_tf.TensorShape(
                [None, num_additional_inputs] if local else [num_additional_inputs]
            )
        ),
        target_shapes
    )

    def get_data():
        """Data generator for graph json files into GNNInput named tuples together with target
        tensor.
        """
        samples = []
        for fn in files:
            samples.extend(_json.load(open(fn, "r")))
        for sample in samples:
            graph = _nx.readwrite.json_graph.node_link_graph(sample)
            _edges = graph.edges(data=True)
            _nodes = dict(graph.nodes(data=True)).values()
            sources, targets, edges = zip(*[(src, tgt, edge) for src, tgt, edge in _edges])
            edge_features = _tf.constant(_np.array([
                [edge[k] for k in edge_feature_names if k in edge] for edge in edges
            ]))
            edge_sources = _tf.squeeze(_tf.constant(_np.array(sources)))
            edge_targets = _tf.squeeze(_tf.constant(_np.array(targets)))
            node_features = _tf.constant(_np.array([
                [node[k] for k in node_feature_names if k in node]
                for node in _nodes
            ]))
            additional_inputs = (
                _tf.constant(_np.array([
                    [node[k] for k in additional_inputs_names if k in node]
                    for node in _nodes
                ]))
                if local else
                _tf.constant(_np.array([
                    graph.graph[additional_input] for additional_input in additional_inputs_names
                    if additional_input in graph.graph
                ]))
            )
            data = GNNInput(
                edge_features=edge_features,
                edge_sources=edge_sources,
                edge_targets=edge_targets,
                node_features=node_features,
                additional_inputs=additional_inputs,
            )
            if local:
                y = _tf.squeeze(_tf.constant(_np.array([
                    [node[k] for k in target if k in node] for node in _nodes
                ])))
            else:
                y = _tf.constant(_np.array([
                    graph.graph[_target] for _target in target if _target in graph.graph
                ]))
            yield data, y

    return _tf.data.Dataset\
        .from_generator(generator=get_data, output_types=output_types, output_shapes=output_shapes)\
        .padded_batch(batch_size)\
        .prefetch(_tf.data.experimental.AUTOTUNE)\
        .repeat()
