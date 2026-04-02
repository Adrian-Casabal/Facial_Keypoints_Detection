import torch
from torch.utils.data import Dataset
import numpy as np
import pandas as pd

class FacialKeypointsDataset(Dataset):
    def __init__(self, df, train=True):
        self.df = df.reset_index(drop=True)
        self.train = train
        self.keypoint_cols = [c for c in df.columns if c != "Image"]

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # image
        img = np.fromstring(row["Image"], sep=' ', dtype=np.float32).reshape(96, 96)
        img /= 255.0
        img = np.expand_dims(img, axis=0)
        img = torch.tensor(img, dtype=torch.float32)

        if self.train:
            y = row[self.keypoint_cols].values.astype(np.float32)
            mask = ~np.isnan(y)

            y = np.nan_to_num(y, nan=48.0)
            y = (y - 48.0) / 48.0

            return (
                img,
                torch.tensor(y, dtype=torch.float32),
                torch.tensor(mask, dtype=torch.float32)
            )
        else:
            return img