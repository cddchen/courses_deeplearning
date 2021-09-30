import pandas as pd
import os
from PIL import Image


def readfile(image_dir_path, label_path, category_path):
    image_dir = sorted(os.listdir(image_dir_path))
    x = []
    for file in image_dir:
        if file.endswith('.png'):
            x.append(Image.open(os.path.join(image_dir_path, file)))

    label_mat = pd.read_csv(label_path)
    y = label_mat['TrueLabel']
    category_mat = pd.read_csv(category_path)
    z = category_mat['CategoryName']
    return x, y, z