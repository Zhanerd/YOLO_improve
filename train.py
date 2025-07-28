import torch
from models.cnn_ import SimpleCNN
from datasets._dataset import get_dataloaders
from utils.train_utils import train_one_epoch, evaluate
from config.cnn_config import config


def main():
    device = torch.device(config['device'] if torch.cuda.is_available() else 'cpu')

    train_loader, val_loader, class_names = get_dataloaders(r"C:\Users\84728\Desktop\converted_output",
                                                            config['batch_size'])

    model = SimpleCNN(num_classes=len(class_names)).to(device)
    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'])

    for epoch in range(config['epochs']):
        print(f"\nEpoch {epoch + 1}/{config['epochs']}")
        train_loss = train_one_epoch(model, train_loader, loss_fn, optimizer, device)
        val_loss, val_acc = evaluate(model, val_loader, loss_fn, device)
        print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2%}")


if __name__ == "__main__":
    main()
