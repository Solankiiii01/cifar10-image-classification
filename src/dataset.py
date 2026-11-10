
import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms


# CIFAR-10 class names
CLASS_NAMES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
]


# Mean and standard deviation for normalization
CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2470, 0.2435, 0.2616)


def get_dataloaders(
    batch_size=64,
    augment=False,
    data_dir="/content/drive/MyDrive/cifar10-data",
    seed=42
):

    # Basic preprocessing
    basic_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD)
    ])

    # Preprocessing for training
    if augment:
        train_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD)
        ])
    else:
        train_transform = basic_transform

    # Load training dataset
    train_dataset = datasets.CIFAR10(
        root=data_dir,
        train=True,
        download=False,
        transform=train_transform
    )

    # Load validation dataset
    validation_dataset = datasets.CIFAR10(
        root=data_dir,
        train=True,
        download=False,
        transform=basic_transform
    )

    # Load testing dataset
    test_dataset = datasets.CIFAR10(
        root=data_dir,
        train=False,
        download=False,
        transform=basic_transform
    )

    # Divide training data into training and validation
    generator = torch.Generator().manual_seed(seed)

    indices = torch.randperm(
        len(train_dataset),
        generator=generator
    ).tolist()

    train_indices = indices[:45000]
    validation_indices = indices[45000:]

    train_subset = Subset(train_dataset, train_indices)
    validation_subset = Subset(validation_dataset, validation_indices)

    # Create DataLoaders
    train_loader = DataLoader(
        train_subset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0
    )

    validation_loader = DataLoader(
        validation_subset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0
    )

    return train_loader, validation_loader, test_loader
