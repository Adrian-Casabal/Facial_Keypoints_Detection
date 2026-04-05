import random
from pathlib import Path

import pandas as pd
import torch
import torch.nn.functional as F
import torch.optim as optim
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from tqdm import tqdm

from config import BACKBONE_LR, BATCH_SIZE, EPOCHS, HEAD_LR, NUM_WORKERS, SEED, VAL_SIZE, WEIGHT_DECAY
from dataset import FacialKeypointsDataset
from model import KeypointCNN


def seed_everything(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def masked_smooth_l1_loss(pred, target, mask, beta=0.05):
    diff = F.smooth_l1_loss(pred, target, reduction="none", beta=beta)
    diff = diff * mask
    return diff.sum() / mask.sum().clamp_min(1.0)


def main():
    src_dir = Path(__file__).resolve().parent
    project_root = src_dir.parent
    data_dir = project_root / "data"
    train_csv = data_dir / "training.csv"
    model_path = project_root / "best_model.pth"

    if not train_csv.exists():
        raise FileNotFoundError(f"Training data not found at: {train_csv}")

    seed_everything(SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("\n===== DEVICE INFO =====")
    print(f"Using device: {device}")

    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"CUDA Version: {torch.version.cuda}")
        print(f"Number of GPUs: {torch.cuda.device_count()}")
    else:
        print("Running on CPU")
    print("=======================\n")

    train_df = pd.read_csv(train_csv)
    train_df, val_df = train_test_split(train_df, test_size=VAL_SIZE, random_state=SEED)

    train_ds = FacialKeypointsDataset(train_df, train=True, augment=True)
    val_ds = FacialKeypointsDataset(val_df, train=True, augment=False)

    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=device.type == "cuda",
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=device.type == "cuda",
    )

    model = KeypointCNN(pretrained=True).to(device)
    optimizer = optim.AdamW(
        [
            {"params": model.backbone_parameters(), "lr": BACKBONE_LR},
            {"params": model.head_parameters(), "lr": HEAD_LR},
        ],
        weight_decay=WEIGHT_DECAY,
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=4,
        min_lr=1e-5,
    )

    print(
        "Backbone: HRNet-W18 | "
        f"pretrained_weights_loaded={model.using_pretrained_weights} | "
        f"backbone_lr={BACKBONE_LR} | head_lr={HEAD_LR}"
    )

    best_val = float("inf")
    epochs_without_improvement = 0

    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0.0

        progress = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{EPOCHS}", leave=False)

        for imgs, targets, masks in progress:
            imgs = imgs.to(device)
            targets = targets.to(device)
            masks = masks.to(device)

            optimizer.zero_grad()
            preds = model(imgs)
            loss = masked_smooth_l1_loss(preds, targets, masks)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_loss += loss.item()
            progress.set_postfix(loss=f"{loss.item():.4f}")

        model.eval()
        val_loss = 0.0

        with torch.no_grad():
            for imgs, targets, masks in val_loader:
                imgs = imgs.to(device)
                targets = targets.to(device)
                masks = masks.to(device)

                preds = model(imgs)
                loss = masked_smooth_l1_loss(preds, targets, masks)
                val_loss += loss.item()

        train_loss /= max(len(train_loader), 1)
        val_loss /= max(len(val_loader), 1)
        scheduler.step(val_loss)

        backbone_lr = optimizer.param_groups[0]["lr"]
        head_lr = optimizer.param_groups[1]["lr"]
        print(
            f"Epoch {epoch + 1}: train={train_loss:.4f}, val={val_loss:.4f}, "
            f"backbone_lr={backbone_lr:.6f}, head_lr={head_lr:.6f}"
        )

        if val_loss < best_val:
            best_val = val_loss
            torch.save(model.state_dict(), model_path)
            epochs_without_improvement = 0
            print(f"Saved improved model to {model_path}")
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= 12:
            print("Early stopping triggered.")
            break


if __name__ == "__main__":
    main()
