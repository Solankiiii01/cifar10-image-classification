
import os
import torch
import yaml
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

from src.dataset import get_dataloaders, CLASS_NAMES
from src.model import CIFAR10CNN


def evaluate_model(config_path, checkpoint_path):

    # 1. Load experiment configuration
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)

    # 2. Select device
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("Evaluation device:", device)

    # 3. Load test dataset
    _, _, test_loader = get_dataloaders(
        batch_size=config["training"]["batch_size"],
        augment=False,
        data_dir=config["dataset"]["data_dir"],
        seed=config["training"]["seed"]
    )

    # 4. Load the trained model
    model = CIFAR10CNN().to(device)

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device
    )

    model.load_state_dict(checkpoint["model_state_dict"])

    # 5. Set model to evaluation mode
    model.eval()

    all_predictions = []
    all_labels = []

    # 6. Make predictions on test images
    with torch.no_grad():

        for images, labels in test_loader:

            images = images.to(device)

            outputs = model(images)

            predictions = outputs.argmax(dim=1)

            all_predictions.extend(
                predictions.cpu().numpy()
            )

            all_labels.extend(labels.numpy())

    # 7. Calculate evaluation metrics
    accuracy = accuracy_score(
        all_labels, all_predictions
    )

    precision = precision_score(
        all_labels,
        all_predictions,
        average="macro",
        zero_division=0
    )

    recall = recall_score(
        all_labels,
        all_predictions,
        average="macro",
        zero_division=0
    )

    f1 = f1_score(
        all_labels,
        all_predictions,
        average="macro",
        zero_division=0
    )

    print("\n========== TEST RESULTS ==========")
    print(f"Accuracy:  {accuracy * 100:.2f}%")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1-score:  {f1:.4f}")

    # 8. Print classification report
    print("\n========== CLASSIFICATION REPORT ==========")

    print(
        classification_report(
            all_labels,
            all_predictions,
            target_names=CLASS_NAMES,
            digits=4,
            zero_division=0
        )
    )

    # 9. Generate confusion matrix
    cm = confusion_matrix(
        all_labels,
        all_predictions
    )

    plt.figure(figsize=(10, 8))

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=CLASS_NAMES,
        yticklabels=CLASS_NAMES
    )

    plt.xlabel("Predicted Label")
    plt.ylabel("Actual Label")
    plt.title("CIFAR-10 Baseline Confusion Matrix")
    plt.tight_layout()
    plt.show()

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "confusion_matrix": cm
    }
