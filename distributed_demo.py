import lightning as L
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import yaml
from tqdm import tqdm
import wandb


class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.__dict__ = self


class WandbLogger:
    def __init__(self, config):
        wandb.init(project="distributed-demo", config=config)
        self.config = config

    def log(self, fabric, log_dict):
        if fabric.is_global_zero:
            wandb.log(log_dict)


# Define the model
class SmallResNet(nn.Module):
    def __init__(self, image_shape, num_classes, **kwargs):
        super(SmallResNet, self).__init__()
        channels, height, width = image_shape
        self.conv1 = nn.Conv2d(
            channels, 16, kernel_size=3, stride=1, padding=1
        )
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1)
        self.fc = nn.Linear(16 * height * width, num_classes)

    def forward(self, x):
        out = self.conv1(x)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.relu(out)
        out = out.flatten(1)
        out = self.fc(out)
        return out


def get_dataloaders(data_dir, batch_size):
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )

    train_set = torchvision.datasets.CIFAR10(
        root=data_dir, train=True, download=True, transform=transform
    )
    test_set = torchvision.datasets.CIFAR10(
        root=data_dir, train=False, download=True, transform=transform
    )

    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=batch_size, shuffle=True, num_workers=0
    )
    test_loader = torch.utils.data.DataLoader(
        test_set, batch_size=batch_size, shuffle=False, num_workers=0
    )

    return train_loader, test_loader


def train(config):
    fabric = L.Fabric(
        accelerator=config.accelerator,
        devices=config.devices,
        strategy=config.strategy,
        num_nodes=config.num_nodes,
    )
    fabric.launch()

    torch.manual_seed(10)
    # Initialize the model and optimizer
    model = SmallResNet(**config)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=config.lr)

    # Get the dataloaders
    train_loader, test_loader = get_dataloaders(
        config.data_dir, config.batch_size
    )

    model, optimizer = fabric.setup(model, optimizer)
    train_loader, test_loader = fabric.setup_dataloaders(
        train_loader, test_loader
    )

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
            correct = (predicted == labels).sum().item()
            running_correct += correct
            running_total += len(labels)
            if i % config.log_freq == config.log_freq - 1:
                running_loss /= config.log_freq
                acc = running_correct / running_total
                fabric.print(
                    f"Epoch {epoch} | Batch {i} | Loss: {running_loss:.4f} | Acc: {acc:.2%}"
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
            for inputs, labels in test_loader:
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                predicted = outputs.argmax(-1)

                # Update statistics
                test_loss += loss.item()
                test_total += len(labels)
                test_correct += (predicted == labels).sum().item()

        acc = 100 * test_correct / test_total
        test_loss /= len(test_loader)
        fabric.print(
            f"Test Set Results | Loss: {test_loss:.4f} | Acc: {acc:.2%}"
        )


if __name__ == "__main__":
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)
    config = AttrDict(config)
    train(config)
