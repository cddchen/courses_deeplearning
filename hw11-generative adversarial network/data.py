import glob
from torchvision import transforms
from torch.utils.data import Dataset
import cv2
import os


class FaceDataset(Dataset):
    def __init__(self, fnames, transform):
        super(FaceDataset, self).__init__()
        self.transform = transform
        self.fnames = fnames
        self.num_samples = len(self.fnames)

    def __getitem__(self, item):
        fname = self.fnames[item]
        img = cv2.imread(fname)
        img = self.BGR2RGB(img)
        img = self.transform(img)
        return img

    def __len__(self):
        return self.num_samples

    def BGR2RGB(self, img):
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def get_dataset(root):
    fnames = glob.glob(os.path.join(root, '*'))
    # resize the img to (64, 64)
    # linearly map [0,1] to [-1,1]
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5] * 3, std=[0.5] * 3)
    ])
    dataset = FaceDataset(fnames, transform)
    return dataset