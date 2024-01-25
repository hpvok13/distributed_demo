import lightning as L
import torch
import torch.nn as nn
import torch.optim as optim
import os
import subprocess
import yaml
from utils import AttrDict, getImageNetDataLoaders
from model import ResNet


def train(config):
    fabric = L.Fabric(
        accelerator=config.accelerator,
        devices=config.devices,
        strategy=config.strategy,
        num_nodes=config.num_nodes,
    )
    fabric.launch()

    def printHelper(*args):
        if fabric.is_global_zero:
            print(*args)

    config.batch_size = config.batch_size // fabric.world_size

    # Initialize the model and optimizer
    printHelper("Initializing model, optimizer, and data loaders")
    model = ResNet(**config)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=config.lr)

    # Get the dataloaders
    train_loader, val_loader = getImageNetDataLoaders(
        config.data_dir, config.batch_size
    )

    model, optimizer = fabric.setup(model, optimizer)
    train_loader, val_loader = fabric.setup_dataloaders(
        train_loader, val_loader
    )
    if config.log_freq is None:
        config.log_freq = len(train_loader)

    printHelper("Starting training loop")
    # Training loop
    for epoch in range(config.epochs):
        running_loss = 0.0
        running_correct = 0
        running_total = 0
        for i, (inputs, labels) in enumerate(train_loader):
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            fabric.backward(loss)
            optimizer.step()

            # Update statistics
            running_loss += loss.item()
            predicted = outputs.argmax(-1)
            running_correct += (predicted == labels).sum().item()
            running_total += len(labels)
            if i % config.log_freq == config.log_freq - 1:
                running_loss /= config.log_freq
                acc = running_correct / running_total
                running_loss = fabric.all_reduce(torch.tensor(running_loss))
                acc = fabric.all_reduce(torch.tensor(acc))
                printHelper(
                    f"Epoch {epoch} | Batch {i+1} | Loss: {running_loss:.4f} | Acc: {acc:.2%}"
                )
                running_loss = 0.0
                running_correct = 0
                running_total = 0

        # Test loop
        model.eval()
        test_loss = 0.0
        test_correct = 0
        test_total = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                outputs = model(inputs)
                loss = criterion(outputs, labels)

                # Update statistics
                test_loss += loss.item()
                predicted = outputs.argmax(-1)
                test_correct += (predicted == labels).sum().item()
                test_total += len(labels)

        test_loss /= len(val_loader)
        acc = test_correct / test_total
        test_loss = fabric.all_reduce(torch.tensor(test_loss))
        acc = fabric.all_reduce(torch.tensor(acc))
        printHelper(
            f"Validation Set Results | Loss: {test_loss:.4f} | Acc: {acc:.2%}"
        )

        fabric.save(
            os.path.join(config.checkpoint_dir, f"checkpoint_{epoch}.ckpt"),
            {
                "model": model,
                "optimizer": optimizer,
                "train_loader": train_loader,
                "val_loader": val_loader,
                "epoch": epoch,
            },
        )


if __name__ == "__main__":
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)
    config = AttrDict(config)
    data_dir = os.getenv("IMAGENET_PATH")
    if not os.path.exists(data_dir):
        cmd = "/usr/local/bin/mount-squashfs /home/gridsan/groups/datasets/ImageNet/imagenet-b1M.fs"
        output = subprocess.check_output(cmd, shell=True)
        data_dir = output.decode("utf-8")[0 : len(output) - 1]
    assert os.path.exists(data_dir)
    config.data_dir = os.path.join(data_dir, config.normal_or_raw)
    train(config)
