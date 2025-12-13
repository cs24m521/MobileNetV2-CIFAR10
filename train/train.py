import torch, random, numpy as np, os
from torch import nn, optim
from data.load_cifar10 import get_cifar10
from models.mobilenetv2_cifar10 import get_mobilenet_v2
from tqdm import tqdm
import wandb

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss, total_correct, total_seen = 0, 0, 0

    for x, y in tqdm(loader, leave=False):
        x, y = x.to(device), y.to(device)

        optimizer.zero_grad()
        out = model(x)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * x.size(0)
        total_correct += (out.argmax(1) == y).sum().item()
        total_seen += x.size(0)

    return total_loss / total_seen, total_correct / total_seen

@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, total_correct, total_seen = 0, 0, 0

    for x, y in loader:
        x, y = x.to(device), y.to(device)
        out = model(x)
        loss = criterion(out, y)

        total_loss += loss.item() * x.size(0)
        total_correct += (out.argmax(1) == y).sum().item()
        total_seen += x.size(0)

    return total_loss / total_seen, total_correct / total_seen

def main(cfg):
    set_seed(cfg['seed'])
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    trainloader, testloader = get_cifar10(batch_size=cfg['batch_size'])
    model = get_mobilenet_v2(num_classes=10, width_mult=cfg['width_mult']).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=cfg['lr'], momentum=0.9, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg['epochs'])

    wandb.init(project=cfg.get("wandb_project", "mobilenetv2_cifar10_compression"), config=cfg)

    for epoch in range(cfg['epochs']):
        train_loss, train_acc = train_one_epoch(model, trainloader, criterion, optimizer, device)
        val_loss, val_acc = evaluate(model, testloader, criterion, device)
        scheduler.step()

        print(f"[Epoch {epoch}] Train Acc: {train_acc*100:.4f} | Val Acc: {val_acc*100:.4f}")

        wandb.log({
            "epoch": epoch,
            "train_loss": train_loss,
            "train_acc": train_acc*100,
            "val_loss": val_loss,
            "val_acc": val_acc*100,
            "lr": scheduler.get_last_lr()[0]
        })

    os.makedirs("checkpoints", exist_ok=True)
    torch.save(model.state_dict(), "checkpoints/mobilenetv2_baseline.pth")

    wandb.finish()

if __name__ == "__main__":
    config = {
        "seed": 42,
        "batch_size": 128,
        "lr": 0.1,
        "epochs": 120,   # change to 120 for full training
        "width_mult": 1.0,
        "wandb_project": "mobilenetv2_cifar10_compression"
    }
    main(config)

