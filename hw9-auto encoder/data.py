import numpy as np
from torch.utils.data import Dataset


def preprocess(image_list):
    '''
    Normalize Image and Permute (N,H,W,C) to (N,C,W,H)
    :param image_list: list of images (9000, 32, 32, 3)
    :return: image_list: list of images (9000, 3, 32, 32)
    '''
    image_list = np.array(image_list)
    image_list = np.transpose(image_list, (0, 3, 1, 2))
    image_list = (image_list / 255.0) * 2 - 1
    image_list = image_list.astype(np.float32)
    return image_list


class ImgDataSet(Dataset):
    def __init__(self, data):
        self.data = data

    def __getitem__(self, item):
        return self.data[item]

    def __len__(self):
        return len(self.data)