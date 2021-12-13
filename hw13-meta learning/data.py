import numpy as np
import glob
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
class Omniglot(Dataset):
    def __init__(self, data_dir, k_way, q_query) -> None:
        self.file_list = [f for f in glob.glob(data_dir + "**/character*", recursive=True)]
        self.transform = transforms.Compose([transforms.ToTensor()])
        self.n = k_way + q_query
    def __getitem__(self, index):
        sample = np.arange(20)
        np.random.shuffle(sample)
        img_path = self.file_list[index]
        img_list = [f for f in glob.glob(img_path + "**/*.png", recursive=True)]
        img_list.sort()
        imgs = [self.transform(Image.open(img_file)) for img_file in img_list]
        imgs = torch.stack(imgs)[sample[:self.n]]
        return imgs
    def __len__(self):
        return len(self.file_list)