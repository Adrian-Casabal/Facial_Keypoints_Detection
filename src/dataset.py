import torch
from torch.utils.data import Dataset
import numpy as np

from config import FLIP_PAIRS, LABEL_MEAN, LABEL_SCALE


def get_flip_mappings(keypoint_cols):
    index_by_name = {name: idx for idx, name in enumerate(keypoint_cols)}
    flip_indices = list(range(len(keypoint_cols)))

    for left_name, right_name in FLIP_PAIRS:
        for suffix in ("_x", "_y"):
            left_key = f"{left_name}{suffix}"
            right_key = f"{right_name}{suffix}"

            if left_key in index_by_name and right_key in index_by_name:
                left_idx = index_by_name[left_key]
                right_idx = index_by_name[right_key]
                flip_indices[left_idx] = right_idx
                flip_indices[right_idx] = left_idx

    x_indices = [idx for idx, name in enumerate(keypoint_cols) if name.endswith("_x")]
    return flip_indices, x_indices


def flip_normalized_keypoints(keypoints, flip_indices, x_indices):
    if torch.is_tensor(keypoints):
        flipped = keypoints.clone()
    else:
        flipped = np.copy(keypoints)

    flipped[..., x_indices] *= -1
    flipped = flipped[..., flip_indices]
    return flipped

class FacialKeypointsDataset(Dataset):
    def __init__(self, df, train=True, augment=False):
        self.df = df.reset_index(drop=True)
        self.train = train
        self.augment = augment and train
        self.keypoint_cols = [c for c in df.columns if c != "Image"]
        self.flip_indices, self.x_indices = get_flip_mappings(self.keypoint_cols)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        img = np.fromstring(row["Image"], sep=" ", dtype=np.float32).reshape(96, 96)
        img = img / 255.0
        img = (img - 0.5) / 0.5

        if self.train:
            y = row[self.keypoint_cols].values.astype(np.float32)
            mask = ~np.isnan(y)

            y = np.nan_to_num(y, nan=LABEL_MEAN)
            y = (y - LABEL_MEAN) / LABEL_SCALE

            if self.augment and np.random.rand() < 0.5:
                img = np.fliplr(img).copy()
                y = flip_normalized_keypoints(y, self.flip_indices, self.x_indices)
                mask = mask[self.flip_indices]

            if self.augment:
                contrast = np.random.uniform(0.9, 1.1)
                brightness = np.random.uniform(-0.08, 0.08)
                img = np.clip(img * contrast + brightness, -1.0, 1.0)

            img = np.expand_dims(img, axis=0)
            img = torch.tensor(img, dtype=torch.float32)

            return (
                img,
                torch.tensor(y, dtype=torch.float32),
                torch.tensor(mask, dtype=torch.float32)
            )
        else:
            img = np.expand_dims(img, axis=0)
            img = torch.tensor(img, dtype=torch.float32)
            return img
