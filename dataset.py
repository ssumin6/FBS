import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader

_CIFAR_MEAN, _CIFAR_STD = (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)

def get_loader(batch_size, num_workers):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(_CIFAR_MEAN, _CIFAR_STD)
    ])

    train_data = torchvision.datasets.CIFAR10('data', train=True, transform=transform, download=True)
    test_data = torchvision.datasets.CIFAR10('data', train=False, transform=transform, download=True)

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

    return train_loader, test_loader
