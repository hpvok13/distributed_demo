import lightning as L
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import yaml
from torch.utils.data import random_split
from tqdm import tqdm


# Define the model
class SmallResNet(nn.Module):
    def __init__(self, num_classes=10):
        super(SmallResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1)
        self.fc = nn.Linear(16 * 32 * 32, num_classes)

    def forward(self, x):
        out = self.conv1(x)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.relu(out)
        out = out.flatten(1)
        out = self.fc(out)
        return out


def get_dataloaders():
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    )

    train_set = torchvision.datasets.CIFAR10(
        root="./data", train=True, download=True, transform=transform
    )
    test_set = torchvision.datasets.CIFAR10(
        root="./data", train=False, download=True, transform=transform
    )

    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=128, shuffle=True, num_workers=0
    )
    test_loader = torch.utils.data.DataLoader(
        test_set, batch_size=128, shuffle=False, num_workers=0
    )

    return train_loader, test_loader


def train(config):
    torch.manual_seed(10)
    fabric = L.Fabric(accelerator="auto", devices="auto", num_nodes=2, strategy="auto")

    # Initialize the model and optimizer
    model = SmallResNet()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    # Get the dataloaders
    train_loader, test_loader = get_dataloaders()

    model, optimizer = fabric.setup(model, optimizer)
    train_loader, test_loader = fabric.setup_dataloaders(train_loader, test_loader)

    # Training loop
    for epoch in range(10):
        running_loss = 0.0
        running_correct = 0
        running_total = 0
        bar = tqdm(
            train_loader,
            desc=(f"Training | Epoch: {epoch} | Loss: {0:.4f} | Acc: {0:.2%}"),
            disable=fabric.local_rank != 0,
        )
        for i, (inputs, labels) in tqdm(enumerate(train_loader)):
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
                bar.set_description(
                    f"Training | Epoch: {epoch} | "
                    f"Loss: {running_loss:.4f} | "
                    f"Acc: {acc:.2%}"
                )
                running_loss = 0.0

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
        fabric.print(f"Test Set Results | Loss: {test_loss:.4f} | Acc: {acc:.2%}")


if __name__ == "__main__":
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)
    train(config)
