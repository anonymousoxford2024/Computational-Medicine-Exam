from datetime import datetime
from pathlib import Path
from typing import Literal, Union, List

import numpy as np
import torch
from matplotlib import pyplot as plt
from sklearn.metrics import roc_curve
from torch import nn
from torch.utils.data import DataLoader
from torch_geometric.datasets import MoleculeNet
from torch_geometric.loader import DataLoader as GeomDataLoader
from torchmetrics import AUROC

from models.predict import predict


def auroc_score(
    all_preds: torch.Tensor,
    all_labels: torch.Tensor,
    task: Literal["binary", "multiclass", "multilabel"] = "binary",
    average: Union[Literal["macro", "weighted", "none"], None] = "weighted",
):
    """
    Calculate the Area Under the Receiver Operating Characteristic (AUROC) score.
    """
    auroc = AUROC(task=task, average=average)
    score = auroc(all_preds, all_labels)

    return score


def bootstrap_auroc(
    model: nn.Module, data_loader: Union[DataLoader, GeomDataLoader], n_bootstrap=100
):
    """
    Computes AUROC scores using bootstrap sampling for model evaluation, returning the mean AUROC
    score and the standard deviation.
    """
    scores = []
    for _ in range(n_bootstrap):
        dataset: MoleculeNet = data_loader.dataset
        indices = torch.randint(0, len(dataset), (len(dataset),))

        if isinstance(data_loader, GeomDataLoader):
            subset = [dataset[i] for i in indices.tolist()]
            subset_loader = GeomDataLoader(subset, batch_size=data_loader.batch_size)
        elif isinstance(data_loader, DataLoader):
            subset_sampler = torch.utils.data.SubsetRandomSampler(indices)
            subset_loader = torch.utils.data.DataLoader(
                data_loader.dataset,
                batch_size=data_loader.batch_size,
                sampler=subset_sampler,
            )

        preds, labels = predict(model, subset_loader)
        score = auroc_score(preds, labels)
        scores.append(score)

    mean_score = np.mean(scores)
    std_score = np.std(scores)
    return mean_score, std_score


def plot_roc_curve(
    model_names: List[str],
    preds: List[torch.Tensor],
    labels: List[torch.Tensor],
    legend: List[str],
    title: str = "ROC Curve",
    save_fig: bool = True,
) -> None:
    """
    Plots the Receiver Operating Characteristic (ROC) curve for each model specified,
    comparing their performance in terms of True Positive Rate vs. False Positive Rate.
    """
    plt.figure(figsize=(8, 8))
    colors = [
        "#E69F00",
        "#56B4E9",
        "#009E73",
    ]

    # plot random classifier
    plt.plot(
        [0, 1], [0, 1], color="navy", lw=2, linestyle="--", label="Random Classifier"
    )

    # plot all the models
    for i, model_name in enumerate(model_names):
        fpr, tpr, thresholds = roc_curve(
            labels[i].cpu().numpy(), preds[i].cpu().numpy()
        )
        plt.plot(
            fpr,
            tpr,
            color=colors[i],
            lw=2,
            label=f"{model_name} (AUROC = {legend[i]})",
        )

    plt.xlim([0.0, 1.05])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate", fontsize=14)
    plt.ylabel("True Positive Rate", fontsize=14)
    plt.title(title, fontsize=16)
    plt.legend(loc="lower right", fontsize=13)

    plt.gca().set_aspect("equal", adjustable="box")
    plt.gca().set_facecolor("#f6f6f6")
    plt.grid(True)

    if save_fig:
        current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        plots_dir = Path(__file__).parent.parent.parent / "plots"
        filename = plots_dir / f"ROC_curve_{current_time}.png"

        plt.savefig(filename, bbox_inches="tight")

    plt.show()


def plot_training_progress(
    aurocs: dict,
    n_epochs: int,
    title: str,
    plt_train: bool = True,
    plt_val: bool = True,
    plt_test: bool = True,
):
    """
    Plots the auroc scores over epochs for train, validation, and test sets.
    """
    plt.figure(figsize=(10, 6))

    # Colorblind-friendly colors
    colors = ["#E69F00", "#56B4E9", "#009E73"]

    epochs = [i for i in range(n_epochs)]
    if plt_train:
        plt.plot(
            epochs,
            aurocs["train"],
            label="Train AUROC",
            color=colors[0],
            marker="o",
        )
    if plt_val:
        plt.plot(
            epochs,
            aurocs["val"],
            label="Validation AUROC",
            color=colors[1],
            marker="s",
        )
    if plt_test:
        plt.plot(
            epochs,
            aurocs["test"],
            label="Test AUROC",
            color=colors[2],
            marker="^",
        )

    plt.title(title)
    plt.xlabel("Epoch")
    plt.ylabel("AUROC Score")
    plt.legend()
    plt.grid(True)
    plt.show()
