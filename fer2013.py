from torch.utils.data import Dataset
import numpy as np
from PIL import Image

class FER2013(Dataset):
    """FER2013 dataset."""
    def __init__(self, data, transform=None):
        """
        Args:
            data (tuple): A tuple of (images, labels).
            transform (callable, optional): Optional transform to be applied
                on an image.
        """
        self.images = data[0]
        self.labels = data[1]
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        image, label = self.images[index], self.labels[index]
        image = image[:, :, np.newaxis]
        image = np.concatenate((image, image, image), axis=2)
        image = Image.fromarray(image.astype('uint8'))
        if self.transform is not None:
            image = self.transform(image)
        return image, label
