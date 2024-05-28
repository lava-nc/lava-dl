import torch
import torchvision
import torchvision.transforms as transforms


def split_train_val(train, val_split):
    train_len = int(len(train) * (1.0-val_split))
    train, val = torch.utils.data.random_split(
        train,
        (train_len, len(train) - train_len),
        generator=torch.Generator().manual_seed(42),
    )
    return train, val


def get_datasets():
    # Get validation data
    transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            transforms.Lambda(lambda x: x.view(3, 1024).t())
        ])
    transform_train = transform_test = transform

    # Datasets
    trainset = torchvision.datasets.CIFAR10(
        root='./../data/cifar/', train=True, download=True, transform=transform_train)
    trainset, val = split_train_val(trainset, val_split=0.1)

    testset = torchvision.datasets.CIFAR10(
        root='./../data/cifar/', train=False, download=True, transform=transform_test)

    val_loader = torch.utils.data.DataLoader(dataset=val , batch_size=1, shuffle=True, num_workers=8)
    return trainset, testset, val_loader