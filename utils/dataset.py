import os
import cv2
import glob
import random
import numpy as np

import torch
import torch.utils.data as data

from augment import PairCompose, PairRandomCrop, PairRandomHorizontalFilp, PairToTensor, PairCenterCrop


def gen_data_list(dir):
    data_list = []
    for flow_map in sorted(glob.glob(os.path.join(dir, '*_img.npy'))):
        basename = os.path.basename(flow_map)[:-9]
        input_path = os.path.join(dir, basename+'_img.png')
        targe_path = os.path.join(dir, basename+'_gt.png')
        data_list.append([input_path, targe_path])
    return data_list


def creat_dataset(root):

    train_data_root = os.path.join(root, "train")
    val_data_root = os.path.join(root, "val")

    train_list = gen_data_list(train_data_root)
    val_list = gen_data_list(val_data_root)

    tra_dataset = MyData(train_list, transform=True)
    val_dataset = MyData(val_list, transform=False)

    print('{} train samples and {} val samples'.format(len(tra_dataset), len(val_dataset)))

    return tra_dataset, val_dataset


def creat_dataloader(train_set, val_set, batch_size, drop_last=True):
    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=batch_size, pin_memory=True, shuffle=True, drop_last=drop_last)
    val_loader = torch.utils.data.DataLoader(
        val_set, batch_size=batch_size, pin_memory=True, shuffle=False, drop_last=drop_last)

    return train_loader, val_loader


class MyData(data.Dataset):
    def __init__(self, path_list, transform=False):

        self.path_list = path_list
        if transform:
            self.tf = PairCompose(
                [
                    PairRandomCrop(256),
                    PairRandomHorizontalFilp(),
                    PairToTensor()
                ]
            )
        else:
            self.tf = PairCompose(
                [
                    PairCenterCrop(224),
                    PairToTensor()
                ]
            )

    def __getitem__(self, index):
        img_file, gt_file = self.path_list[index]
        img, gt = self.load_data(img_file, gt_file)
        img, gt = self.tf(img, gt)
        return img, gt

    def load_data(self, img_file, gt_file):
        img = cv2.imread(img_file)
        gt = cv2.imread(gt_file)
        return img/255., gt/255.

    def __len__(self):
        return len(self.path_list)