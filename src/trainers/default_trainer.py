import torch
import torch.nn as nn
import numpy as np
import os.path as osp
import networkx as nx
import torch.nn.functional as func
from tqdm import tqdm
from torch import optim
from datetime import datetime
from torch_geometric.loader import DataLoader
from torch_geometric.utils import to_dense_batch

from utils.dataset import STDataset
from utils.metric import cal_metric, masked_mae_np
from utils.tools import mkdirs, load_best_model


def train(inputs, args):
    # Define the current year model save path
    path = osp.join(args.path, str(args.year))
    mkdirs(path)

    # Set loss function
    if args.loss == "mse":
        lossfunc = func.mse_loss
    elif args.loss == "huber":
        lossfunc = func.smooth_l1_loss

    # Data loaders
    train_loader = DataLoader(
        STDataset(inputs, "train"),
        batch_size=args.batch_size,
        shuffle=True,
        pin_memory=True,
        num_workers=32
    )
    val_loader = DataLoader(
        STDataset(inputs, "val"),
        batch_size=args.batch_size,
        shuffle=False,
        pin_memory=True,
        num_workers=32
    )
    test_loader = DataLoader(
        STDataset(inputs, "test"),
        batch_size=args.batch_size,
        shuffle=False,
        pin_memory=True,
        num_workers=32
    )

    # Use the adjacency matrix of the entire graph
    vars(args)["sub_adj"] = vars(args)["adj"]

    args.logger.info("[*] Year " + str(args.year) + " Dataset load!")

    # Model definition
    if args.init is True and args.year > args.begin_year:
        # Not the first year: load best model from previous year
        gnn_model, _ = load_best_model(args)
        model = gnn_model

        # Freeze backbone layers
        for name, param in model.named_parameters():
            if "gcn1" in name or "tcn1" in name or "gcn2" in name or "fc" in name:
                param.requires_grad = False

        model.expand_adaptive_params(args.graph_size)
    else:
        # First year: instantiate base model
        gnn_model = args.methods[args.method](args).to(args.device)
        model = gnn_model
        model.expand_adaptive_params(args.graph_size)
        model.count_parameters()

    # Optimizer: only optimize trainable parameters
    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr
    )

    args.logger.info("[*] Year " + str(args.year) + " Training start")
    lowest_validation_loss = 1e7
    counter = 0
    patience = 5
    model.train()
    use_time = []

    for epoch in range(args.epoch):
        start_time = datetime.now()

        # ------------------ Training Phase ------------------
        cn = 0
        training_loss = 0.0
        for batch_idx, data in enumerate(train_loader):
            if epoch == 0 and batch_idx == 0:
                args.logger.info("node number {}".format(data.x.shape))

            data = data.to(args.device, non_blocking=True)
            optimizer.zero_grad()
            pred = model(data, args.sub_adj)  # [B * N, T_out]

            loss = lossfunc(data.y, pred, reduction="mean")
            training_loss += float(loss)
            cn += 1

            loss.backward()
            optimizer.step()

        # Record time
        epoch_time = (datetime.now() - start_time).total_seconds()
        if epoch == 0:
            total_time = epoch_time
        else:
            total_time += epoch_time
        use_time.append(epoch_time)
        training_loss /= cn

        # ------------------ Validation Phase ------------------
        validation_loss = 0.0
        cn = 0
        with torch.no_grad():
            for batch_idx, data in enumerate(val_loader):
                data = data.to(args.device, non_blocking=True)
                pred = model(data, args.sub_adj)
                # Use masked MAE (numpy version) for validation
                loss = masked_mae_np(
                    data.y.cpu().data.numpy(),
                    pred.cpu().data.numpy(),
                    0
                )
                validation_loss += float(loss)
                cn += 1
        validation_loss /= cn

        args.logger.info(
            f"epoch:{epoch}, "
            f"training loss:{training_loss:.4f} "
            f"validation loss:{validation_loss:.4f}"
        )

        # ------------------ Early Stopping & Model Saving ------------------
        if validation_loss <= lowest_validation_loss:
            counter = 0
            lowest_validation_loss = round(validation_loss, 4)
            save_path = osp.join(path, f"{lowest_validation_loss:.4f}.pkl")
            torch.save({'model_state_dict': model.state_dict()}, save_path)
        else:
            counter += 1
            if counter > patience:
                break

    # Load best model for testing
    best_model_path = osp.join(path, f"{lowest_validation_loss:.4f}.pkl")
    best_model = model
    best_model.load_state_dict(
        torch.load(best_model_path, map_location=args.device)["model_state_dict"]
    )
    best_model = best_model.to(args.device)

    # Test the model
    test_model(best_model, args, test_loader, pin_memory=True)
    args.result[args.year] = {
        "total_time": total_time,
        "average_time": sum(use_time) / len(use_time),
        "epoch_num": epoch + 1
    }
    args.logger.info(
        "Finished optimization, total time:{:.2f} s, best model:{}".format(
            total_time, best_model_path
        )
    )


def test_model(model, args, testset, pin_memory):
    model.eval()
    pred_ = []
    truth_ = []
    loss = 0.0
    with torch.no_grad():
        cn = 0
        for data in testset:
            data = data.to(args.device, non_blocking=pin_memory)
            pred = model(data, args.adj)  # [B * N, T_out]

            loss += func.mse_loss(data.y, pred, reduction="mean")

            # Convert to dense batch for metric computation
            pred, _ = to_dense_batch(pred, batch=data.batch)      # [B, N, T_out]
            data.y, _ = to_dense_batch(data.y, batch=data.batch)  # [B, N, T_out]

            pred_.append(pred.cpu().data.numpy())
            truth_.append(data.y.cpu().data.numpy())
            cn += 1

        loss /= cn
        args.logger.info("[*] loss:{:.4f}".format(loss))

        pred_ = np.concatenate(pred_, axis=0)   # [B_total, N, T_out]
        truth_ = np.concatenate(truth_, axis=0) # [B_total, N, T_out]
        cal_metric(truth_, pred_, args)


def masked_mae(prediction: torch.Tensor, target: torch.Tensor, null_val: float = np.nan) -> torch.Tensor:
    """
    Compute masked Mean Absolute Error.
    """
    if np.isnan(null_val):
        mask = ~torch.isnan(target)
    else:
        eps = 5e-5
        mask = ~torch.isclose(
            target,
            torch.tensor(null_val).expand_as(target).to(target.device),
            atol=eps,
            rtol=0.0
        )

    mask = mask.float()
    mask /= torch.mean(mask)          # Normalize to avoid bias
    mask = torch.nan_to_num(mask)     # Handle potential NaNs

    loss = torch.abs(prediction - target)
    loss = loss * mask
    loss = torch.nan_to_num(loss)

    return torch.mean(loss)