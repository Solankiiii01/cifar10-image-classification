
# CIFAR-10 Image Classification Using CNN

A deep learning project that classifies images into 10 categories
using a Convolutional Neural Network (CNN) built with PyTorch.

This project compares a baseline CNN with a CNN trained using
data augmentation to understand how augmentation affects
model performance and generalization.

## Project Overview

CIFAR-10 contains 60,000 RGB images of size 32x32 pixels
belonging to 10 classes.

The dataset is divided into:
- Training: 45,000 images
- Validation: 5,000 images
- Testing: 10,000 images

### Classes

Airplane, Automobile, Bird, Cat, Deer, Dog, Frog, Horse, Ship,
and Truck.

## Experiments

Two experiments were conducted:

**1. Baseline CNN**
- Custom CNN architecture
- Adam optimizer
- Learning rate: 0.001
- Batch size: 64
- Epochs: 20
- Data augmentation: Disabled

**2. Augmented CNN**
- Same CNN architecture and training settings
- Random cropping with padding
- Random horizontal flipping
- Data augmentation: Enabled

## Model Architecture

The CNN consists of:

1. Input layer: 3 x 32 x 32
2. Convolutional layer: 32 filters
3. ReLU activation
4. Max pooling
5. Convolutional layer: 64 filters
6. ReLU activation
7. Max pooling
8. Flatten layer
9. Fully connected layer: 128 neurons
10. Output layer: 10 classes

Total trainable parameters: 545,098.

## Results

### Test Performance

| Metric | Baseline | Augmented |
|---|---:|---:|
| Accuracy | 72.04% | 75.11% |
| Macro Precision | 0.7218 | 0.7592 |
| Macro Recall | 0.7204 | 0.7511 |
| Macro F1-score | 0.7201 | 0.7485 |

The augmented model improved test accuracy by 3.07
percentage points.

### Training Performance

| Metric | Baseline | Augmented |
|---|---:|---:|
| Best Validation Accuracy | 72.76% | 75.60% |
| Final Training Accuracy | 97.39% | 75.56% |

The baseline model showed significant overfitting.
Data augmentation helped improve generalization.

## Error Analysis

The augmented model's most frequent classification errors
included:

| Actual Class | Predicted Class | Errors |
|---|---|---:|
| Cat | Dog | 237 |
| Cat | Frog | 131 |
| Bird | Frog | 130 |
| Dog | Cat | 121 |
| Bird | Dog | 101 |

Prediction confidence analysis:

- Cat predicted as Dog: 59.84% average confidence.
- Cat predicted as Frog: 67.40% average confidence.

Although overall performance improved, distinguishing
some visually similar classes remains challenging.

## Technologies Used

- Python
- PyTorch
- Torchvision
- NumPy
- Pandas
- Matplotlib
- Seaborn
- Scikit-learn
- PyYAML
- Google Colab

## Project Structure

```text
cifar10-image-classification/
|
|-- configs/
|   |-- baseline.yaml
|   |-- augmented.yaml
|
|-- src/
|   |-- dataset.py
|   |-- model.py
|   |-- train.py
|   |-- evaluate.py
|
|-- notebooks/
|   |-- error_analysis.ipynb
|
|-- reports/
|   |-- baseline_results.md
|   |-- experiment_comparison.md
|   |-- error_analysis.md
|   |-- experiment_report.pdf
|   |-- augmented_loss_curve.png
|   |-- augmented_accuracy_curve.png
|   |-- test_accuracy_comparison.png
|   |-- confusion_matrix_comparison.png
|   |-- misclassified_images.png
|
|-- requirements.txt
|-- README.md
```

## Installation

Clone the repository:

```bash
git clone https://github.com/Solankiiii01/cifar10-image-classification.git

cd cifar10-image-classification
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## Dataset Setup

The project uses the CIFAR-10 dataset provided by
torchvision.

Download the dataset using:

```python
from torchvision import datasets

DATA_DIR = "/content/drive/MyDrive/cifar10-data"

datasets.CIFAR10(
    root=DATA_DIR,
    train=True,
    download=True
)

datasets.CIFAR10(
    root=DATA_DIR,
    train=False,
    download=True
)
```

Update the data directory in the YAML configuration files
if your dataset is stored elsewhere.

## Training and Evaluation

The project provides reusable training and evaluation
functions in:

- `src/train.py`
- `src/evaluate.py`

The functions can be imported and called from a Python
notebook or script.

Example:

```python
from src.train import train_model

model, history = train_model(
    config_path="configs/baseline.yaml",
    checkpoint_dir="checkpoints"
)
```

Evaluate a saved checkpoint:

```python
from src.evaluate import evaluate_model

results = evaluate_model(
    config_path="configs/baseline.yaml",
    checkpoint_path="checkpoints/baseline_best.pth"
)
```

For the augmented experiment, use `configs/augmented.yaml`
and its corresponding checkpoint directory.

## Reports

The `reports/` directory contains:
- Baseline experiment results
- Baseline vs augmented comparison
- Training and validation graphs
- Confusion matrix comparison
- Misclassified image analysis
- Final PDF experiment report

## Future Improvements

- Experiment with deeper CNN architectures.
- Explore dropout and batch normalization.
- Try learning-rate scheduling.
- Experiment with additional augmentation techniques.
- Improve classification performance for difficult classes.

## Conclusion

The augmented CNN achieved 75.11% test accuracy compared
with 72.04% for the baseline CNN.

The experiment demonstrates how data augmentation can
improve the generalization performance of a CNN on
the CIFAR-10 image classification dataset.
