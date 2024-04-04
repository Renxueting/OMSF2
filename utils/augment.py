import random
import torchvision.transforms as transforms
import torch
import numpy as np


class PairRandomCrop(object):

    def __init__(self, size):
        self.size = (int(size), int(size))

    def __call__(self, corr, intn, label):
        h, w, _ = intn.shape
        th, tw = self.size
        if w == tw and h == th:
            return corr, intn, label

        x1 = random.randint(0, w - tw)
        y1 = random.randint(0, h - th)
        corr = corr[y1: y1 + th,x1: x1 + tw]
        intn = intn[y1: y1 + th,x1: x1 + tw]
        label = label[y1: y1 + th,x1: x1 + tw]
        return corr, intn, label

class PairCenterCrop(object):

    def __init__(self, size):
        self.size = (int(size), int(size))

    def __call__(self, corr, intn, label):
        h, w, _ = intn.shape
        th, tw = self.size
        if w == tw and h == th:
            return corr, intn, label

        x1 = int(round((w-tw)/2.))
        y1 = int(round((h-th)/2.))

        corr = corr[y1: y1 + th,x1: x1 + tw]
        intn = intn[y1: y1 + th,x1: x1 + tw]
        label = label[y1: y1 + th,x1: x1 + tw]
        return corr, intn, label


class PairCompose(transforms.Compose):
    def __call__(self, corr, intn, label):
        for t in self.transforms:
            corr, intn, label = t(corr, intn, label)
        return corr, intn, label


class PairRandomHorizontalFilp(object):
    """Randomly horizontally flips the given Numpy(H,W,C) with a probability of 0.5
    """
    def __call__(self, corr, intn, label):

        if random.random() < 0.5:
            corr = np.copy(np.flipud(corr))
            intn = np.copy(np.flipud(intn))
            label = np.copy(np.flipud(label))
        return corr, intn, label


class PairToTensor(object):
    def __call__(self, corr, intn, label):
        """
        Args:
            pic (PIL Image or numpy.ndarray): Image to be converted to tensor.
        Returns:
            Tensor: Converted image.
        """
        corr = np.transpose(corr, (2, 0, 1))
        intn = np.transpose(intn, (2, 0, 1))
        label = np.transpose(label, (2, 0, 1))
        return torch.from_numpy(corr).float(), torch.from_numpy(intn).float(), torch.from_numpy(label).float()



if __name__ == "__main__":
    import numpy as np
    a = np.random.rand(32).reshape(4,4,2)
    print(a[:,:,0])
    aa = a[0: 2,0:2]
    print(aa[:,:,0])