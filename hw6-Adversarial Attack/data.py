from torch.utils.data import Dataset, DataLoader
import torch


class CustomImageDataSet(Dataset):
    def __init__(self, x, y=None, transform=None, target_transform=None):
        self.x = x
        self.y = y
        if y is not None:
            self.y = torch.LongTensor(y)
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        X = self.x[idx]
        if self.transform is not None:
            X = self.transform(X)
        if self.y is not None:
            Y = self.y[idx]
            if self.target_transform is not None:
                Y = self.target_transform(Y)
            return X, Y
        else:
            return X
