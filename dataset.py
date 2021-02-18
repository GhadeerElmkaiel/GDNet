"""
 @Time    : 2020/3/15 18:56
 @Author  : TaylorMei
 @E-mail  : mhy666@mail.dlut.edu.cn
 
 @Project : CVPR2020_GDNet
 @File    : dataset.py
 @Function:
 
"""
import os
import os.path
from torch import uint8

import torch.utils.data as data
from PIL import Image
import numpy as np
from torchvision import transforms
import torchvision.transforms.functional as TF

from skimage import feature
import matplotlib.pyplot as plt

import random as r


def make_dataset(root):
    if isinstance(root, str):
        img_list = [os.path.splitext(f)[0] for f in os.listdir(os.path.join(root, 'image')) if f.endswith('.jpg')]
        return [
            (os.path.join(root, 'image', img_name + '.jpg'), os.path.join(root, 'mask', img_name + '.png'))
            for img_name in img_list]
    else:
        if isinstance(root, list):
            img_list = []
            res = []
            for root_ in root:
                img_list = [os.path.splitext(f)[0] for f in os.listdir(os.path.join(root_, 'image')) if f.endswith('.jpg')]
                res += [
                (os.path.join(root_, 'image', img_name + '.jpg'), os.path.join(root_, 'mask', img_name + '.png'))
                for img_name in img_list]
            return res


class ImageFolder(data.Dataset):
    def __init__(self, root, joint_transform=None, img_transform=None, target_transform=None, add_real_imgs=False, random_percent = 0.5, image_crop_percentage= 0.8):
        self.root = root
        self.imgs = make_dataset(root)
        self.joint_transform = joint_transform
        self.img_transform = img_transform
        self.target_transform = target_transform
        self.len = len(self.imgs)
        self.add_real_imgs = add_real_imgs
        self.random_percent = random_percent
        self.image_crop_percentage = image_crop_percentage
        

    def __getitem__(self, index):
        img_path, gt_path = self.imgs[index]
        img = Image.open(img_path).convert('RGB')
        target = Image.open(gt_path).convert('L')
        # Creating the boundaries for the mask
        target_np = np.array(target)
        edges = feature.canny(target_np)*255
        edges = edges.astype(np.uint8)

        # plt.imshow(edges)
        # print("Max in mask image:", np.amax(target_np))
        # print("Max in boundaries image:", np.amax(edges))
        edges = Image.fromarray(edges)
        # edges.show()
        
        width, height = img.size
        output_size=(int(height*self.image_crop_percentage), int(width*self.image_crop_percentage))
        if r.random() < self.random_percent:
            i, j, h, w = transforms.RandomCrop.get_params(img, output_size= output_size)
            img = TF.crop(img, i, j, h, w)
            target = TF.crop(target, i, j, h, w)
            edges = TF.crop(edges, i, j, h, w)

        if r.random() < self.random_percent:
            img = TF.hflip(img)
            target = TF.hflip(target)
            edges = TF.hflip(edges)

        if self.joint_transform is not None:
            img, target = self.joint_transform(img, target)
        if self.img_transform is not None:
            img = self.img_transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)
            edges = self.target_transform(edges)
        # print("Image:  ",img.size())
        # print("Target: ",target.size())
        return img, target, edges

    def __len__(self):
        return self.len

    def sample(self, batch_size):
        """
        function for getting batch of items of the dataset
        """
        batch = {"img":[], "mask":[], "size":[], "r_img":[], "r_mask":[]}
        indices = np.random.choice(self.len, batch_size, replace=False)
        masks = []
        for i in indices:
            (img, mask, edges) = self.__getitem__(i)
            batch["img"].append(np.asarray(img))
            masks.append(np.asarray(mask))

            img_path, gt_path = self.imgs[i]
            img = Image.open(img_path).convert('RGB')
            mask = Image.open(gt_path).convert('L')
            
            # Adding the real images to the batch for debugging if needed
            #TODO
            # Remove the add_real_imgs tage
            # if self.add_real_imgs:
            #     batch["r_img"].append(img)
            #     batch["r_mask"].append(mask)
            batch["r_img"].append(img)
            batch["r_mask"].append(mask)
            
            # Adding the real image size to the batch
            w, h = img.size
            batch["size"].append((w, h))
        batch["img"] = np.array(batch["img"])
        # print(masks)
        batch["mask"] = np.asarray(masks)/255.0
        return batch


class TestImageFolder(data.Dataset):
    def __init__(self, root, joint_transform=None, img_transform=None, target_transform=None, add_real_imgs=False):
        self.root = root
        self.imgs = make_dataset(root)
        self.joint_transform = joint_transform
        self.img_transform = img_transform
        self.target_transform = target_transform
        self.len = len(self.imgs)
        self.add_real_imgs = add_real_imgs

    def __getitem__(self, index):
        img_path, gt_path = self.imgs[index]
        real_img = Image.open(img_path).convert('RGB')
        w, h = real_img.size
        size = np.array([w, h])
        img = Image.open(img_path).convert('RGB')

        real_target = Image.open(gt_path)
        target = Image.open(gt_path)
        if self.joint_transform is not None:
            img, target = self.joint_transform(img, target)
        if self.img_transform is not None:
            img = self.img_transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target, np.array(real_img), np.array(real_target), size

    def __len__(self):
        return self.len
