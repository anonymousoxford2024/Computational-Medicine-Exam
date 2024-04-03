import json
from pathlib import Path

import torch
from tqdm import tqdm

from data.data_loading import load_tox21_data_as_graphs
from evaluation.evaluate import auroc_score, plot_training_progress
from models.gnn import JKAdvancedGNN
from models.predict import predict
from utils import set_random_seeds


def train(model, loader, opt, loss_fn):
    """
    Trains a given model using the provided data loader, optimizer, and loss function.
    This function iterates over the dataset, computes the loss for each batch,
    performs backpropagation, and updates the model's weights.
    Returns average loss.
    """
    model.train()
    total_loss = 0
    for i, data in enumerate(loader):
        opt.zero_grad()
        if hasattr(data, "y"):
            y = data.y
        else:
            y = data["y"]

        output = model(data)
        loss = loss_fn(output, y)
        loss.backward()
        opt.step()
        num_graphs = 1.0
        if hasattr(data, "num_graphs"):
            num_graphs = data.num_graphs
        total_loss += loss.item() * num_graphs

    return total_loss / len(loader)


if __name__ == "__main__":
    set_random_seeds(42)
    binary_cls = True
    upsampling = True
    train_loader, val_loader, test_loader = load_tox21_data_as_graphs(
        binary_cls, upsampling
    )

    for p_dropout in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]:
        for weight_decay in [5e-4]:
            for pool in ["mean"]:  # , "max", "add"]:
                for n_layers in [2, 3, 4]:
                    for lr in [0.0001]:  # , 0.00005, 0.00001]:
                        highest_model_auroc = 0.0

                        input_dim = 9
                        hidden_channels = 64
                        n_classes = 1 if binary_cls else 12

                        model = JKAdvancedGNN(
                            input_dim=input_dim,
                            hidden_channels=hidden_channels,
                            n_classes=n_classes,
                            n_layers=n_layers,
                            p_dropout=p_dropout,
                            pool=pool,
                            # jk_mode=jk_mode,
                        )
                        criterion = torch.nn.BCEWithLogitsLoss()
                        optimizer = torch.optim.Adam(
                            model.parameters(), lr=lr, weight_decay=weight_decay
                        )

                        aurocs = {"train": [], "val": [], "test": []}
                        n_epochs = 100
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
                                file_name = f"{model.__class__.__name__}_model_pd={p_dropout}_wd={weight_decay}_nl={n_layers}_lr={lr}_pool={pool}"
                                model_path = models_dir / f"{file_name}.pth"

                                torch.save(model.state_dict(), model_path)
                                param_path = models_dir / f"{file_name}.json"

                                with open(param_path, "w") as file:
                                    parameters = {
                                        "input_dim": input_dim,
                                        "hidden_channels": hidden_channels,
                                        "n_classes": n_classes,
                                        "pool": pool,
                                        # "jk_mode": jk_mode,
                                        "binary_cls": binary_cls,
                                        "upsampling": upsampling,
                                        "p_dropout": p_dropout,
                                        "weight_decay": weight_decay,
                                        "n_layers": n_layers,
                                        "lr": lr,
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

                        title = f"AUROC Scores Over Epochs (max_test={round(max(aurocs['test']).item(), 4)}, upsampling={upsampling}) \n(model={model.__class__.__name__}, lr={lr}, p_dropout={p_dropout}, weight_decay={weight_decay}, pool={pool}, n_layers={n_layers})"
                        plot_training_progress(aurocs, n_epochs, title, plt_test=False)
