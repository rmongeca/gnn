import json

from .os import empty_dirs as _empty_dirs


def join_graphs_into_dataset(files, output_dir, output_file_name="data.json", empty_dirs=False):
    if empty_dirs:
        _empty_dirs(output_dir)
    graphs = [json.load(open(file, "r")) for file in files]
    with open(output_dir / output_file_name, "w") as fp:
        json.dump(graphs, fp)
