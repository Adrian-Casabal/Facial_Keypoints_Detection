import pandas as pd
import torch
from torch.utils.data import DataLoader
import torch.optim as optim
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from pathlib import Path

from dataset import FacialKeypointsDataset
from model import KeypointCNN

def masked_mse_loss(pred, target, mask):
    diff = (pred - target) ** 2
    diff = diff * mask
    return diff.sum() / mask.sum()

def main():
    src_dir = Path(__file__).resolve().parent
    project_root = src_dir.parent
    data_dir = project_root / "data"
    train_csv = data_dir / "training.csv"
    model_path = project_root / "best_model.pth"

    if not train_csv.exists():
        raise FileNotFoundError(f"Training data not found at: {train_csv}")


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
    train_df, val_df = train_test_split(train_df, test_size=0.1, random_state=42)

    train_ds = FacialKeypointsDataset(train_df, train=True)
    val_ds = FacialKeypointsDataset(val_df, train=True)

    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=64, shuffle=False)

    model = KeypointCNN().to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    best_val = float("inf")

    for epoch in range(25):
        model.train()
        train_loss = 0

        for imgs, targets, masks in tqdm(train_loader):
            imgs, targets, masks = imgs.to(device), targets.to(device), masks.to(device)

            optimizer.zero_grad()
            preds = model(imgs)
            loss = masked_mse_loss(preds, targets, masks)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        model.eval()
        val_loss = 0

        with torch.no_grad():
            for imgs, targets, masks in val_loader:
                imgs, targets, masks = imgs.to(device), targets.to(device), masks.to(device)
                preds = model(imgs)
                loss = masked_mse_loss(preds, targets, masks)
                val_loss += loss.item()

        print(f"Epoch {epoch+1}: train={train_loss:.4f}, val={val_loss:.4f}")

        if val_loss < best_val:
            best_val = val_loss
            torch.save(model.state_dict(), model_path)

if __name__ == "__main__":
    main()
