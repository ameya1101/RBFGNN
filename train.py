import torch
import numpy as np
from torch_geometric.datasets import TUDataset
from torch_geometric.data import DataLoader
from torch_geometric import utils
from model.model import RBFGNN
import torch.nn.functional as F
import argparse
import os
from torch.utils.data.dataset import Subset
from time import time
from statistics import mean


def test(args, model, loader):
    model.eval()
    correct = 0
    loss = 0.0
    with torch.no_grad():
        for data in loader:
            data = data.to(args.device)
            out = model(data)
            pred = out.max(dim=1)[1]
            correct += pred.eq(data.y).sum().item()
            loss += F.nll_loss(out, data.y, reduction="sum").item()
    return correct / len(loader.dataset), loss / len(loader.dataset)


def train(args, model, train_loader, val_loader, optimizer, epoch):
    losses = []
    epoch_t0 = time()
    model.train()
    for batch_idx, data in enumerate(train_loader):
        data = data.to(args.device)
        optimizer.zero_grad()
        out = model(data)
        loss = F.nll_loss(out, data.y)
        loss.backward()
        optimizer.step()
        if batch_idx % 100 == 0:
            print(
                f"Train Epoch {epoch} [{batch_idx}/{len(train_loader.dataset)}]  ({100. * batch_idx / len(train_loader):.6f})]\tLoss: {loss.item():.6f}"
            )

        if args.dry_run:
            quit()
        losses.append(loss.item())

    print(f"...epoch time: {time() - epoch_t0}s")
    print(f"...epoch {epoch}: average train loss = {mean(losses)}")

    val_acc, val_loss = test(model, val_loader)
    return mean(losses), val_loss, val_acc


def main():
    parser = argparse.ArgumentParser(
        description="Radial Basis Function - GNN Implementation"
    )

    parser.add_argument("--seed", type=int, default=777, help="seed")
    parser.add_argument("--batch_size", type=int, default=128, help="batch size")
    parser.add_argument("--lr", type=float, default=0.0005, help="learning rate")
    parser.add_argument(
        "--weight_decay", type=float, default=0.0001, help="weight decay"
    )
    parser.add_argument(
        "--pooling_ratio", type=float, default=0.5, help="pooling ratio"
    )
    parser.add_argument(
        "--dropout_ratio", type=float, default=0.5, help="dropout ratio"
    )
    parser.add_argument(
        "--num_hidden", type=int, default=128, help="number of hidden channels"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="DS",
        help="dataset subdirectory under dir: data. e.g. DS",
    )
    parser.add_argument(
        "--epochs", type=int, default=1000, help="maximum number of epochs"
    )
    parser.add_argument(
        "--val_epochs", type=int, default=25, help="maximum number of validation epochs"
    )
    parser.add_argument(
        "--patience", type=int, default=50, help="patience for early stopping"
    )
    parser.add_argument(
        "--conv", type=str, default="GCNConv", help="type of convolution layer"
    )
    parser.add_argument(
        "--use_node_attr", type=bool, default=True, help="node features"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        default=False,
        help="quickly check a single pass",
    )

    args = parser.parse_args()
    args.device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.cuda.manual_seed(args.seed)

    dataset = TUDataset(
        os.path.join("data", args.dataset),
        name=args.dataset,
        use_node_attr=args.use_node_attr,
    )
    args.num_classes = dataset.num_classes
    args.num_features = dataset.num_features

    # This section enforces a particular train/test split based on a predetermined list
    n_lines = sum(1 for line in open("./data/DS/train_split"))
    n_trn_lines = round(0.9 * n_lines)
    train_ids = (
        np.loadtxt("./data/DS/train_split", max_rows=n_trn_lines, dtype=int) - 1
    ).tolist()
    val_ids = (
        np.loadtxt("./data/DS/train_split", skip_rows=n_trn_lines, dtype=int) - 1
    ).tolist()
    test_ids = (np.loadtxt("./data/DS/test_split", dtype=int) - 1).tolist()
    train_set = Subset(dataset, train_ids)
    val_set = Subset(dataset, val_ids)
    test_set = Subset(dataset, test_ids)

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_set, batch_size=1, shuffle=False)

    model = RBFGNN(args).to(args.device)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )

    output = {"train_loss": [], "test_loss": [], "test_acc": [], "val_loss": [], "val_acc": []}
    for epoch in range(1, args.epochs + 1):
        print(f"---- Epoch {epoch} ----")
        train_loss, val_loss, val_acc = train(args, model, train_loader, val_loader, optimizer)
        print(f"Validation loss: {val_loss}\t Validation Accuracy: {val_acc}")
        test_acc, test_loss = test(args, model, test_loader)
        print(f"Test loss: {test_loss}\t Test Accuracy: {test_acc}")

        output["train_loss"].append(train_loss)
        output["test_loss"].append(test_loss)
        output["test_acc"].append(test_acc)
        output["val_loss"].append(val_loss)
        output["val_acc"].append(val_acc)

    with open("final_score.txt", "w") as f:
        f.write("%d\t%f\t%f\n" % (args.num_hidden, test_loss, test_acc))

if __name__ == '__main__':
    main()