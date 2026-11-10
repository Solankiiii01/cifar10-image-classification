
import os
import random
import torch
import torch.nn as nn
import yaml

from src.dataset import get_dataloaders
from src.model import CIFAR10CNN


def train_model(config_path, checkpoint_dir):

    # 1. Load configuration
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)

    # 2. Set random seed
    seed = config["training"]["seed"]
    random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # 3. Select device
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("Training device:", device)

    # 4. Load dataset
    train_loader, validation_loader, _ = get_dataloaders(
        batch_size=config["training"]["batch_size"],
        augment=config["augmentation"]["enabled"],
        data_dir=config["dataset"]["data_dir"],
        seed=seed
    )

    # 5. Initialize CNN
    model = CIFAR10CNN().to(device)

    # 6. Define loss function
    criterion = nn.CrossEntropyLoss()

    # 7. Define optimizer
    learning_rate = config["training"]["learning_rate"]

    if config["training"]["optimizer"].lower() == "adam":
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=learning_rate
        )
    else:
        raise ValueError("Unsupported optimizer")

    # 8. Training settings
    epochs = config["training"]["epochs"]

    os.makedirs(checkpoint_dir, exist_ok=True)

    best_val_accuracy = 0.0
    history = []

    # 9. Start training
    for epoch in range(epochs):

        model.train()

        running_loss = 0.0
        correct = 0
        total = 0

        for images, labels in train_loader:

            images = images.to(device)
            labels = labels.to(device)

            # Clear previous gradients
            optimizer.zero_grad()

            # Forward propagation
            outputs = model(images)

            # Calculate loss
            loss = criterion(outputs, labels)

            # Backpropagation
            loss.backward()

            # Update model weights
            optimizer.step()

            running_loss += loss.item() * images.size(0)

            predictions = outputs.argmax(dim=1)

            total += labels.size(0)
            correct += (predictions == labels).sum().item()

        train_loss = running_loss / total
        train_accuracy = 100 * correct / total

        # 10. Validate the model
        model.eval()

        val_loss_total = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():

            for images, labels in validation_loader:

                images = images.to(device)
                labels = labels.to(device)

                outputs = model(images)

                loss = criterion(outputs, labels)

                val_loss_total += loss.item() * images.size(0)

                predictions = outputs.argmax(dim=1)

                val_total += labels.size(0)
                val_correct += (
                    predictions == labels
                ).sum().item()

        val_loss = val_loss_total / val_total
        val_accuracy = 100 * val_correct / val_total

        # 11. Store epoch results
        history.append({
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "train_accuracy": train_accuracy,
            "val_loss": val_loss,
            "val_accuracy": val_accuracy
        })

        print(
            f"Epoch [{epoch + 1}/{epochs}] "
            f"Train Loss: {train_loss:.4f} "
            f"Train Accuracy: {train_accuracy:.2f}% "
            f"Val Loss: {val_loss:.4f} "
            f"Val Accuracy: {val_accuracy:.2f}%"
        )

        # 12. Save best model
        if val_accuracy > best_val_accuracy:

            best_val_accuracy = val_accuracy

            torch.save(
                {
                    "epoch": epoch + 1,
                    "model_state_dict": model.state_dict(),
                    "validation_accuracy": val_accuracy
                },
                os.path.join(
                    checkpoint_dir,
                    "baseline_best.pth"
                )
            )

    # 13. Save final model
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "epoch": epochs
        },
        os.path.join(
            checkpoint_dir,
            "baseline_final.pth"
        )
    )

    print("\nTraining completed!")
    print("Best validation accuracy:", best_val_accuracy)

    return model, history