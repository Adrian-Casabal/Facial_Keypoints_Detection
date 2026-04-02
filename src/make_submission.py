import pandas as pd
import numpy as np
from pathlib import Path

def main():
    src_dir = Path(__file__).resolve().parent
    project_root = src_dir.parent
    data_dir = project_root / "data"
    predictions_path = project_root / "predictions.npy"
    train_csv = data_dir / "training.csv"
    lookup_csv = data_dir / "IdLookupTable.csv"
    submissions_dir = project_root / "submissions"
    submission_path = submissions_dir / "submission.csv"

    if not predictions_path.exists():
        raise FileNotFoundError(f"Predictions file not found at: {predictions_path}")

    if not train_csv.exists():
        raise FileNotFoundError(f"Training data not found at: {train_csv}")

    if not lookup_csv.exists():
        raise FileNotFoundError(f"Lookup table not found at: {lookup_csv}")

    submissions_dir.mkdir(exist_ok=True)

    predictions = np.load(predictions_path)
    train = pd.read_csv(train_csv)
    lookup = pd.read_csv(lookup_csv)

    feature_cols = [c for c in train.columns if c != "Image"]
    feature_to_index = {name: idx for idx, name in enumerate(feature_cols)}

    locations = []

    for _, row in lookup.iterrows():
        image_id = row["ImageId"] - 1
        feature_name = row["FeatureName"]
        col_idx = feature_to_index[feature_name]

        locations.append(predictions[image_id, col_idx])

    submission = pd.DataFrame({
        "RowId": lookup["RowId"],
        "Location": locations
    })

    submission.to_csv(submission_path, index=False)

if __name__ == "__main__":
    main()
