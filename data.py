import math
import numpy as np
import albumentations as A
import tensorflow as tf
from tensorflow.keras.utils import Sequence, to_categorical
from tensorflow.keras.datasets.cifar10 import load_data

def get_cifar10():
    return load_data()

def get_train_transforms():
    return A.Compose(
        [
            A.RandomBrightnessContrast(p = .8),
            A.ToGray(p = 1e-2),
            A.ShiftScaleRotate(shift_limit_x = .2, shift_limit_y = .2, rotate_limit = 20),
            A.HorizontalFlip(p = .5),
            A.VerticalFlip(p = .5),
            A.Cutout(num_holes = 8, max_h_size = 4, max_w_size = 4, fill_value = 0, p = .5)
        ]
    )

class DataGenerator(Sequence):
    def __init__(self, train_data, train_labels, transforms = None, batch_size = 32, shuffle = True, train = True):
        self.train_data = train_data
        self.train_labels = train_labels
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.transforms = transforms
        self.indices = np.arange(len(train_data))
        self.train = train
        self.on_epoch_end()
    def __len__(self):
        if self.train: return len(self.train_data) // self.batch_size
        return math.ceil(self.train_data.shape[0] / self.batch_size)
    def __getitem__(self, index):
        indices = self.indices[index * self.batch_size : (index + 1) * self.batch_size]
        images = self.train_data[indices]
        labels = self.train_labels[indices]
        images = self.preprocess_images(images)
        return images.astype(np.float32) / 255., labels
    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indices)
    def preprocess_images(self, images):
        if self.transforms:
            return np.array(list(map(lambda image: self.transforms(image = image)['image'], images)))
        return images

