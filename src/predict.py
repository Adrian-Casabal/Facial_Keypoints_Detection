from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from config import BATCH_SIZE, KEYPOINT_COLUMNS, LABEL_MEAN, LABEL_SCALE, NUM_WORKERS
from dataset import FacialKeypointsDataset, flip_normalized_keypoints, get_flip_mappings
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

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    test_df = pd.read_csv(test_csv)
    test_ds = FacialKeypointsDataset(test_df, train=False)
    test_loader = DataLoader(
        test_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=device.type == "cuda",
    )

    model = KeypointCNN(pretrained=False).to(device)
    try:
        state_dict = torch.load(model_path, map_location=device)
        model.load_state_dict(state_dict)
    except RuntimeError as exc:
        raise RuntimeError(
            "The saved weights do not match the current model architecture. "
            "Run train.py again to produce a new best_model.pth before predicting."
        ) from exc

    model.eval()

    flip_indices, x_indices = get_flip_mappings(KEYPOINT_COLUMNS)
    predictions = []

    with torch.no_grad():
        for imgs in test_loader:
            imgs = imgs.to(device)

            preds = model(imgs)

            flipped_imgs = torch.flip(imgs, dims=[3])
            flipped_preds = model(flipped_imgs)
            flipped_preds = flip_normalized_keypoints(flipped_preds, flip_indices, x_indices)

            preds = (preds + flipped_preds) / 2.0
            preds = preds.clamp(-1.0, 1.0)

            predictions.append(preds.cpu().numpy())

    predictions = np.vstack(predictions)
    predictions = predictions * LABEL_SCALE + LABEL_MEAN

    np.save(predictions_path, predictions)
    print(f"Saved predictions to {predictions_path}")


if __name__ == "__main__":
    main()
