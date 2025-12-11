import torch
import torchvision
import torchvision.transforms as transforms

def get_cifar_loaders(dataset_name='cifar10', batch_size=64, num_workers=2):
    print(f"Preparing Data ({dataset_name.upper()})...")

    if dataset_name == 'cifar10':
        # Mean/Std for CIFAR-10
        stats = ((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        DatasetClass = torchvision.datasets.CIFAR10
        num_classes = 10
    else:
        # Mean/Std for CIFAR-100
        stats = ((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
        DatasetClass = torchvision.datasets.CIFAR100
        num_classes = 100

    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4), #32x32 -> 40x40 -> 32x32
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(*stats),  # *stats -> stats[0], stats[1]
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(*stats),
    ])

    trainset = DatasetClass(root='./data', train=True, download=True, transform=transform_train)
    testset = DatasetClass(root='./data', train=False, download=True, transform=transform_test)

    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

    testloader = torch.utils.data.DataLoader(
        testset, batch_size=100, shuffle=False, num_workers=num_workers)

    return trainloader, testloader, num_classes