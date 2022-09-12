import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from torch.utils.data import Dataset
from typing import Tuple


def get_data() -> Tuple[np.array, np.array, float, float]:
    train_data = pd.read_csv("data/train.csv")
    test_data = pd.read_csv("data/test.csv")

    X_nums = np.vstack([train_data.iloc[:, 11:-1].to_numpy(), test_data.iloc[:, 11:].to_numpy()])
    X_nums = (X_nums - X_nums.mean(0)) / X_nums.std(0)

    X_cat = np.vstack([train_data.iloc[:, 1:11].to_numpy(), test_data.iloc[:, 1:11].to_numpy()])
    encoder = OneHotEncoder(sparse=False)
    X_cat = encoder.fit_transform(X_cat)

    X = np.hstack([X_cat, X_nums])
    y = train_data["target"].to_numpy().reshape(-1, 1)
    return X, y, X_cat.shape[1], X_nums.shape[1]


class SingleDataset(Dataset):
    def __init__(self, x: np.array, is_sparse: bool = False):
        self.x = x.astype("float32")
        self.is_sparse = is_sparse

    def __len__(self) -> float:
        return self.x.shape[0]

    def __getitem__(self, index: int) -> np.array:
        x = self.x[index]
        if self.is_sparse:
            x = x.toarray().squeeze()
        return x
