import pandas as pd
import torch
from torch.utils.data import DataLoader
import numpy as np
from pathlib import Path

from dataset import FacialKeypointsDataset
from model import KeypointCNN

def main():
    src_dir = Path(__file__).resolve().parent
    project_root = src_dir.parent
    data_dir = project_root / "data"
    test_csv = data_dir / "test.csv"
    model_path = project_root / "best_model.pth"
    predictions_path = project_root / "predictions.npy"

    if not test_csv.exists():
        raise FileNotFoundError(f"Test data not found at: {test_csv}")

    if not model_path.exists():
        raise FileNotFoundError(f"Model weights not found at: {model_path}")

    device = "cuda" if torch.cuda.is_available() else "cpu"

    test_df = pd.read_csv(test_csv)

    test_ds = FacialKeypointsDataset(test_df, train=False)
    test_loader = DataLoader(test_ds, batch_size=64, shuffle=False)

    model = KeypointCNN().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    predictions = []

    with torch.no_grad():
        for imgs in test_loader:
            imgs = imgs.to(device)
            preds = model(imgs).cpu().numpy()
            predictions.append(preds)

    predictions = np.vstack(predictions)
    predictions = predictions * 48.0 + 48.0

    np.save(predictions_path, predictions)

if __name__ == "__main__":
    main()
