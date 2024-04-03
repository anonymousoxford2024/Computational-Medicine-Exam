import json
import os
from collections import OrderedDict
from pathlib import Path

import torch
from torch import nn

from data.data_loading import load_tox21_data_as_graphs, load_tox21_data_as_embeddings
from evaluation.evaluate import auroc_score, plot_roc_curve, bootstrap_auroc
from models.baseline_model import BaselineClassifier
from models.gnn import JKAdvancedGNN
from models.predict import predict
from utils import set_random_seeds


def load_model(model_path: str, params: dict, state_dict: OrderedDict) -> nn.Module:
    """
    Load a Basline or JKGNN model from its state dictionary and parameters.

    This function dynamically loads a model based on its type indicated within the model_path,
    initializes it with the provided parameters, then loads the model weights from the state_dict.
    """
    model: nn.Module
    if "Baseline" in model_path:
        model = BaselineClassifier(
            params["emb_dim"],
            params["hidden_channels"],
            params["n_classes"],
            params["n_layers"],
            params["p_dropout"],
        )
        model.load_state_dict(state_dict)

    elif "GNN" in model_path:
        model = JKAdvancedGNN(
            params["input_dim"],
            params["hidden_channels"],
            params["n_classes"],
            params["n_layers"],
            params["p_dropout"],
            params["pool"],
        )
        model.load_state_dict(state_dict)
    else:
        raise ValueError(f"Model path is not valid: {model_path}")
    return model


if __name__ == "__main__":
    set_random_seeds(42)

    model_dir_path = Path(__file__).parent.parent.parent / "models"
    models = {}

    for file_name in os.listdir(model_dir_path):
        if file_name.endswith(".pth"):
            file_path = os.path.join(model_dir_path, file_name)
            state_dict = torch.load(file_path)
            with open(file_path.replace(".pth", ".json"), "r") as file:
                params = json.load(file)

            model_name = ""
            train_loader, val_loader, test_loader = None, None, None

            model: nn.Module
            if "Baseline" in file_name:
                model_name = "Baseline NN Model    "
                train_loader, val_loader, test_loader = load_tox21_data_as_embeddings(
                    params["binary_cls"], params["upsampling"], params["emb_dim"]
                )
            elif "GNN" in file_name:
                model_name = "JumpKnowledge GNN"
                train_loader, val_loader, test_loader = load_tox21_data_as_graphs(
                    params["binary_cls"], params["upsampling"]
                )

            model = load_model(model_name, params, state_dict)
            models[model_name] = {}
            models[model_name]["model"] = model

            train_preds, train_labels = predict(model, train_loader)
            models[model_name]["train_auroc"] = auroc_score(train_preds, train_labels)

            val_preds, val_labels = predict(model, val_loader)
            models[model_name]["val_auroc"] = auroc_score(val_preds, val_labels)

            test_mean_auroc, test_std_auroc = bootstrap_auroc(
                model, test_loader, n_bootstrap=100
            )
            models[model_name][
                "test_auroc"
            ] = f"{100*test_mean_auroc:.1f}% ± {100*test_std_auroc:.1f}%"

            test_preds, test_labels = predict(model, test_loader)
            models[model_name]["test_preds"] = test_preds
            models[model_name]["test_labels"] = test_labels

    model_names = []
    all_preds = []
    all_labels = []
    legend_labels = []
    for model_name, details in models.items():
        model_names.append(model_name)
        all_preds.append(details["test_preds"])
        all_labels.append(details["test_labels"])
        legend_labels.append(details["test_auroc"])

    plot_roc_curve(
        model_names,
        preds=all_preds,
        labels=all_labels,
        title=f"ROC-Curve",
        legend=legend_labels,
    )
