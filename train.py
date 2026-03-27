"""Training script for MiniCLIP."""

import torch
import torch.nn as nn
from torch.amp import autocast, GradScaler
from tqdm import tqdm

from config import CFG, get_args
from dataset import get_dataloaders
from models import MiniCLIP


def train_one_epoch(model, train_loader, optimizer, scheduler, criterion, scaler, device, epoch, num_epochs):
    """Run one training epoch."""
    model.train()
    total_loss = 0.0
    total_samples = 0

    for batch_idx, (images, text, ids) in enumerate(train_loader):
        images = images.to(device)
        text = {k: v.to(device) for k, v in text.items()}

        optimizer.zero_grad()

        with autocast("cuda"):
            logits = model(images, text)
            batch_size = images.size(0)
            labels = torch.arange(batch_size).to(device)
            loss_i = criterion(logits, labels)
            loss_t = criterion(logits.T, labels)
            loss = (loss_i + loss_t) / 2

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        total_loss += loss.item() * batch_size
        total_samples += batch_size

        if batch_idx % 200 == 0:
            print(f"[Train] Epoch [{epoch+1}/{num_epochs}] Batch [{batch_idx}/{len(train_loader)}] "
                  f"Loss: {loss.item():.4f} LR: {scheduler.get_last_lr()[0]:.6f}")

    return total_loss / total_samples


@torch.no_grad()
def validate(model, val_loader, criterion, device, epoch, num_epochs):
    """Run validation."""
    model.eval()
    total_loss = 0.0
    total_samples = 0

    for images, text, ids in val_loader:
        images = images.to(device)
        text = {k: v.to(device) for k, v in text.items()}

        with autocast("cuda"):
            logits = model(images, text)
            batch_size = images.size(0)
            labels = torch.arange(batch_size).to(device)
            loss_i = criterion(logits, labels)
            loss_t = criterion(logits.T, labels)
            loss = (loss_i + loss_t) / 2

        total_loss += loss.item() * batch_size
        total_samples += batch_size

        print(f" [VAL] Epoch [{epoch+1/{num_epochs}] Loss: {total_loss/ total_samples:.4f}")

    return total_loss / total_samples


def main():
    get_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Data
    train_loader, val_loader, train_dataset, val_dataset = get_dataloaders()
    print(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")

    # Model
    model = MiniCLIP().to(device)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable params: {total_params:,}")

    # Optimizer, scheduler, loss
    optimizer = torch.optim.Adam(model.parameters(), lr=CFG.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=CFG.epochs * len(train_loader)
    )
    criterion = nn.CrossEntropyLoss()
    scaler = GradScaler("cuda")

    # Training loop
    for epoch in range(CFG.epochs):
        train_loss = train_one_epoch(
            model, train_loader, optimizer, scheduler, criterion, scaler, device, epoch, CFG.epochs
        )
        val_loss = validate(model, val_loader, criterion, device, epoch, CFG.epochs)

        print(
            f"Epoch [{epoch+1}/{CFG.epochs}]  "
            f"Train Loss: {train_loss:.4f}  "
            f"Val Loss: {val_loss:.4f}  "
            f"LR: {scheduler.get_last_lr()[0]:.6f}"
        )

    # Save model
    torch.save(model.state_dict(), CFG.save_path)
    print(f"Model saved to {CFG.save_path}")


if __name__ == "__main__":
    main()
