from torchvision.datasets import VisionDataset

from PIL import Image

import os
import os.path
import sys


def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')


class PACS(VisionDataset):
    def __init__(self, root, split='train', domain=None, transform=None, target_transform=None):
        super(PACS, self).__init__(root, transform=transform, target_transform=target_transform)

        self.split = split # define the split, default is train -> per ora inutilizzato
        self.images = list()
        self.labels = list()
        labels_path = [f"{self.root}txt_lists/{x}" for x in os.listdir(f"{self.root}txt_lists")]

        for domain in labels_path:
          with open(domain) as file:
            for line in file:
              [path, label] = line.rstrip().split()
              self.images.append(pil_loader(f"{self.root}PACS/{path}"))
              self.labels.append(label)

    def __getitem__(self, index):
        '''
        __getitem__ should access an element through its index
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        '''

        image, label = self.images[index], self.labels[index]
        # Applies preprocessing when accessing the image
        if self.transform is not None:
            image = self.transform(image)

        return image, label

    def __len__(self):
        '''
        The __len__ method returns the length of the dataset
        It is mandatory, as this is used by several other components
        '''
        length = len(self.images) # Provide a way to get the length (number of elements) of the dataset
        return length
