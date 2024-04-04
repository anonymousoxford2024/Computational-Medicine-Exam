import json
from pathlib import Path

import torch
from tqdm import tqdm

from data.data_loading import load_tox21_data_as_embeddings
from evaluation.evaluate import auroc_score, plot_training_progress
from models.baseline_model import BaselineClassifier
from models.predict import predict
from train_gnn import (
    train,
)
from utils import set_random_seeds

if __name__ == "__main__":
    set_random_seeds(42)  # For reproducibility
    binary_cls = True
    upsampling = True

    for p_dropout in [0.2, 0.4, 0.6]:
        for weight_decay in [0, 5e-4]:
            for n_layers in [2, 3, 4]:
                for lr in [1e-03, 1e-04, 1e-05]:
                    for emb_dim in [512, 1024, 2048]:
                        for hidden_channels in [64, 128, 512, 1024]:
                            highest_model_auroc = 0.0

                            train_loader, val_loader, test_loader = (
                                load_tox21_data_as_embeddings(
                                    binary_cls, upsampling, emb_dim=emb_dim
                                )
                            )
                            n_classes = 1 if binary_cls else 12
                            model = BaselineClassifier(
                                input_dim=emb_dim,
                                hidden_channels=hidden_channels,
                                n_classes=n_classes,
                                n_layers=n_layers,
                                p_dropout=p_dropout,
                            )
                            criterion = torch.nn.BCEWithLogitsLoss()
                            optimizer = torch.optim.Adam(
                                model.parameters(), lr=lr, weight_decay=weight_decay
                            )

                            aurocs = {"train": [], "val": [], "test": []}
                            n_epochs = 150
                            for epoch in tqdm(range(n_epochs)):
                                loss = train(model, train_loader, optimizer, criterion)
                                all_preds, all_labels = predict(model, train_loader)
                                train_auroc = auroc_score(all_preds, all_labels)

                                all_preds, all_labels = predict(model, val_loader)
                                val_auroc = auroc_score(all_preds, all_labels)

                                all_preds, all_labels = predict(model, test_loader)
                                test_auroc = auroc_score(all_preds, all_labels)

                                aurocs["train"].append(train_auroc)
                                aurocs["val"].append(val_auroc)
                                aurocs["test"].append(test_auroc)

                                if val_auroc > highest_model_auroc:
                                    highest_train_auroc = train_auroc.item()
                                    highest_val_auroc = val_auroc.item()
                                    highest_test_auroc = test_auroc.item()

                                    models_dir = Path(__file__).parent.parent / "models"
                                    file_name = f"{model.__class__.__name__}_model_pd={p_dropout}_wd={weight_decay}_nl={n_layers}_lr={lr}_emd={emb_dim}"
                                    model_path = models_dir / f"{file_name}.pth"
                                    param_path = models_dir / f"{file_name}.json"

                                    torch.save(model.state_dict(), model_path)

                                    with open(param_path, "w") as file:
                                        parameters = {
                                            "hidden_channels": hidden_channels,
                                            "n_classes": n_classes,
                                            "n_layers": n_layers,
                                            "binary_cls": binary_cls,
                                            "upsampling": upsampling,
                                            "p_dropout": p_dropout,
                                            "weight_decay": weight_decay,
                                            "lr": lr,
                                            "emb_dim": emb_dim,
                                            "performance": {
                                                "train_auroc": train_auroc.item(),
                                                "val_auroc": val_auroc.item(),
                                                "test_auroc": test_auroc.item(),
                                            },
                                        }
                                        json.dump(parameters, file, indent=4)

                            print(f"Highest val auroc = {highest_val_auroc}")
                            print(f"Corres. train auroc = {highest_train_auroc}")
                            print(f"Corres. test auroc = {highest_test_auroc}")

                            title = f"AUROC Scores Over Epochs (max_test={round(max(aurocs['test']).item(), 4)}, hidden_chan={hidden_channels}) \n(model={model.__class__.__name__}, lr={lr}, p_dropout={p_dropout}, weight_decay={weight_decay}, emb_dim={emb_dim}, n_layers={n_layers})"
                            plot_training_progress(
                                aurocs, n_epochs, title, plt_test=False
                            )
