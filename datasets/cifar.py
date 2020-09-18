from __future__ import print_function
import numpy as np
import pathlib
import torch.utils.data as data

DATASET_PATH = pathlib.Path('path_to_dataset')


class Cifar(data.Dataset):
    # class 100 (64 train, 20 test, 16 val), images per class 600, images size 32x32, channels 3
    def __init__(self, train=True, all=False):
        if all:
            data = np.load(pathlib.Path.joinpath(DATASET_PATH, "train.npy"))
        else:
            data = np.load(pathlib.Path.joinpath(DATASET_PATH, "test.npy"))

        classes = data.shape[0]
        sample_elements = 20
        data = np.concatenate(data[:, :sample_elements])
        data = data.transpose(0, 3, 1, 2).astype(np.float32)
        data = np.pad(data, ((0, 0), (0, 0), (26, 26), (26, 26)))
        labels = np.repeat(np.arange(classes), sample_elements)

        self.data = list(data)
        self.targets = list(labels)

        self.data2 = []
        self.targets2 = []
        for a in range(int(len(self.targets) / 20)):
            start = a * 20
            if train:
                for b in range(start, start + 15):
                    self.data2.append(self.data[b])
                    self.targets2.append(self.targets[b])
            else:
                for b in range(start + 15, start + 20):
                    self.data2.append(self.data[b])
                    self.targets2.append(self.targets[b])

        self.targets = self.targets2
        self.data = self.data2

        print("Total classes = ", len(np.unique(self.targets)))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is index of the target character class.
        """
        image = self.data[index]
        target = self.targets[index]
        return image, target


if __name__ == '__main__':
    Cifar(train=True)