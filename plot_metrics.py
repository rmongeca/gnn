#!/usr/bin/env python
# coding: utf-8
import json
import math
import os
from functools import partial
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from scipy.signal import savgol_filter
from statsmodels.distributions import ECDF

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import tensorflow as tf  # noqa: E402

_model_labels = [
    ("GNN TF2", "gnn"),
    ("iGNNition", "experiment"),
]
plt.rcParams.update({"font.size": 22})


def extract_event_simple_value(event_file, tag_dict):
    return [
        (tag_dict[value.tag], value.simple_value)
        for serialized in tf.data.TFRecordDataset(str(event_file))
        for value in tf.core.util.event_pb2.Event.FromString(
            serialized.numpy()
        ).summary.value
        if value.tag in tag_dict
    ]


def process_gnn_history(run, metrics):
    return [json.load(open(run / "history.json", "r"))[metric] for _, metric in metrics]


def process_ignnition_history(run, metrics):
    train_fn = next(next(run.glob("**/train")).glob("*tfevents*.v2"))
    valid_fn = next(next(run.glob("**/validation")).glob("*tfevents*.v2"))
    train_metrics = {
        f"epoch_{metric}": name for name, metric in metrics if "val" not in metric
    }
    train_events = extract_event_simple_value(
        event_file=train_fn, tag_dict=train_metrics
    )
    valid_metrics = {
        f"epoch_{metric.replace('val_','')}": name
        for name, metric in metrics
        if "val" in metric
    }
    valid_events = extract_event_simple_value(
        event_file=valid_fn, tag_dict=valid_metrics
    )
    return [
        [
            event_value
            for event_name, event_value in train_events + valid_events
            if event_name == name
        ]
        for name, _ in metrics
    ]


def process_history(run, model_label, metrics):
    stat = (
        process_gnn_history(run, metrics)
        if model_label == "gnn"
        else process_ignnition_history(run, metrics)
    )
    epoch_col = [list(range(1, len(stat[0]) + 1))]
    run_col = [[int(run.name.split("_")[-1].replace("-", ""))]*len(stat[0])]
    columns = [metric for _, metric in metrics] + ["epoch", "run"]
    return pd.DataFrame(
        np.transpose(stat + epoch_col + run_col), columns=columns,
    )


def get_stats(runs, metrics, model_labels):
    return {
        model_name: pd.concat(
            [
                process_history(run, model_label, metrics)
                for run in runs
                if model_label in run.name
            ]
        )
        for model_name, model_label in model_labels
    }


def smoooth_func(
    x,
    y,
    x_min=None,
    x_max=None,
    interp_kind="zero",
    interp_num=300,
    window_length=31,
    polyorder=1,
):
    if x_min is None:
        x_min = x[x != -np.inf].min()
    if x_max is None:
        x_max = x[x != np.inf].max()
    if interp_kind and interp_num:
        f = interp1d(x, y, kind=interp_kind)
        x = np.linspace(x_min, x_max, num=interp_num, endpoint=True)
        y = f(x)
    y = savgol_filter(y, window_length, polyorder)
    return x, y


def ecdf_plots(
    model,
    stats,
    metrics,
    output_dir,
    smooth=True,
    interp_kind="zero",
    interp_num=200,
    window_length=3,
    polyorder=1,
):
    smooth_partial = partial(
        smoooth_func,
        interp_kind=interp_kind,
        interp_num=interp_num,
        window_length=window_length,
        polyorder=polyorder,
    )
    stats_gnn = stats["GNN TF2"]
    stats_gnn = stats_gnn[stats_gnn["epoch"] == stats_gnn["epoch"].max()]
    stats_ign = stats["iGNNition"]
    stats_ign = stats_ign[stats_ign["epoch"] == stats_ign["epoch"].max()]
    figures = []
    for name, metric in metrics:
        ecdf_gnn = ECDF(stats_gnn[metric].values)
        ecdf_ignnition = ECDF(stats_ign[metric].values)
        x_min = max(
            ecdf_gnn.x[ecdf_gnn.x != -np.inf].min(),
            ecdf_ignnition.x[ecdf_ignnition.x != -np.inf].min(),
        )
        x_max = min(
            ecdf_gnn.x[ecdf_gnn.x != np.inf].max(),
            ecdf_ignnition.x[ecdf_ignnition.x != np.inf].max(),
        )
        x_gnn, y_gnn = (
            smooth_partial(ecdf_gnn.x, ecdf_gnn.y, x_min=x_min, x_max=x_max)
            if smooth
            else (ecdf_gnn.x, ecdf_gnn.y)
        )
        x_ignnition, y_ignnition = (
            smooth_partial(ecdf_ignnition.x, ecdf_ignnition.y, x_min=x_min, x_max=x_max)
            if smooth
            else (ecdf_ignnition.x, ecdf_ignnition.y)
        )
        fig = plt.figure(figsize=(10, 8), dpi=300, facecolor="white")
        plt.plot(x_gnn, y_gnn, "k-", label="GNN TF2")
        plt.plot(x_ignnition, y_ignnition, "k--", label="iGNNition")
        plt.legend()
        plt.xlabel(name)
        plt.ylabel("Probability")
        plt.tight_layout()
        plt.savefig(output_dir / f"{model}_{name}.png", facecolor=fig.get_facecolor())
        figures.append(fig)
    return figures


def epoch_plots(
    model,
    stats,
    metrics,
    output_dir,
    error_bars=False,
    smooth=True,
    interp_kind="zero",
    interp_num=200,
    window_length=101,
    polyorder=1,
):
    smooth_partial = partial(
        smoooth_func,
        interp_kind=interp_kind,
        interp_num=interp_num,
        window_length=window_length,
        polyorder=polyorder,
    )
    mean_gnn = stats["GNN TF2"].groupby("epoch").mean().reset_index()
    sem_gnn = stats["GNN TF2"].groupby("epoch").sem().reset_index()
    mean_ign = stats["iGNNition"].groupby("epoch").mean().reset_index()
    sem_ign = stats["iGNNition"].groupby("epoch").sem().reset_index()
    figures = []
    for name, metric in metrics:
        fig = plt.figure(figsize=(10, 8), dpi=300, facecolor="white")
        x_gnn, y_gnn = (
            smooth_partial(mean_gnn["epoch"], mean_gnn[metric])
            if smooth
            else (mean_gnn["epoch"], mean_gnn[metric])
        )
        x_ign, y_ign = (
            smooth_partial(mean_ign["epoch"], mean_ign[metric])
            if smooth
            else (mean_ign["epoch"], mean_ign[metric])
        )
        plt.plot(
            x_gnn,
            y_gnn,
            "k-",
            label="GNN TF2",
        )
        plt.plot(
            x_ign,
            y_ign,
            "k--",
            label="iGNNition",
        )
        if error_bars:
            size = sem_gnn[metric].shape[0]
            idx = np.linspace(0, x_gnn.shape[0] - 1, num=size, dtype=int)
            plt.errorbar(
                x_gnn[idx],
                y_gnn[idx],
                color="black",
                fmt="none",
                yerr=sem_gnn[metric],
            )
            plt.errorbar(
                x_ign[idx],
                y_ign[idx],
                color="black",
                fmt="none",
                yerr=sem_ign[metric],
            )
        plt.legend()
        plt.xlabel("Epoch")
        plt.ylabel(name)
        plt.tight_layout()
        plt.savefig(
            output_dir / f"{model}_{name}_epoch.png", facecolor=fig.get_facecolor()
        )
        figures.append(fig)
    return figures


def time_plots(model, log_dir, output_dir):
    _model_labels = {"ignnition": "iGNNition", "gnn": "GNN TF2"}
    times = json.load(open(next(log_dir.glob("run_model_times_*.json")), "r"))
    times_sample = [
        [1000 * time / 11000 for time in times[label]] for label in sorted(times)
    ]
    fig = plt.figure(figsize=(10, 8), dpi=300, facecolor="white")
    plt.boxplot(
        times_sample,
        sym="",
        labels=[_model_labels[label] for label in sorted(times)],
        widths=[0.5] * 2,
    )
    plt.ylim(
        (
            math.floor(np.array(times_sample).min()),
            math.ceil(np.array(times_sample).max()),
        )
    )
    plt.ylabel("Time / sample (ms)")
    plt.tight_layout()
    plt.savefig(output_dir / f"{model}_time.png", facecolor=fig.get_facecolor())
    return fig


def main(model, metrics, model_labels=None, output_dir=Path("plots")):
    if model_labels is None:
        model_labels = _model_labels
    print(f"Plotting {model} model metrics.")

    log_dir = Path(f"ignnition/{model}/logs")
    log_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    runs = [_dir for _dir in log_dir.glob("*") if _dir.is_dir()]
    print(f"Found {len(runs)} log directories, extracting stats...")

    stats = get_stats(runs, metrics, model_labels)
    print("Stats extracted, making plots...")

    ecdf_plots(model, stats, metrics, output_dir, smooth=True)
    epoch_plots(model, stats, metrics, output_dir, error_bars=True, smooth=True)
    time_plots(model, log_dir, output_dir)


if __name__ == "__main__":
    main(
        model="qm9",
        metrics=[
            ("Mean Square Error", "loss"),
            ("Mean Absolute Error", "mean_absolute_error"),
            ("Validation Mean Square Error", "val_loss"),
            ("Validation Mean Absolute Error", "val_mean_absolute_error"),
        ],
        model_labels=[
            ("GNN TF2", "gnn"),
            ("iGNNition", "experiment"),
        ],
        output_dir=Path("plots"),
    )
    main(
        model="radio-resource-management",
        metrics=[
            ("Sum Rate Loss", "loss"),
            ("WMMSE Sum Rate Ratio", "sum_rate_metric"),
            ("Validation Sum Rate Loss", "val_loss"),
            ("Validation WMMSE Sum Rate Ratio", "val_sum_rate_metric"),
        ],
        model_labels=[
            ("GNN TF2", "gnn"),
            ("iGNNition", "experiment"),
        ],
        output_dir=Path("plots"),
    )
