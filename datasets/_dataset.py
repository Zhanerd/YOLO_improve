from torchvision import datasets, transforms
import torch

def get_dataloaders(data_dir, batch_size=32):
    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor()
    ])
    train_data = datasets.ImageFolder(root=f"{data_dir}/train", transform=transform)
    val_data = datasets.ImageFolder(root=f"{data_dir}/val", transform=transform)

    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_data, batch_size=batch_size)

    return train_loader, val_loader, train_data.classes
