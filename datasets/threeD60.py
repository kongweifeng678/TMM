#该数据集处理需要与SVS的数据集处理结合起来

from __future__ import print_function
import os
import cv2
import numpy as np
import random

import torch
from torch.utils import data
from torchvision import transforms
import PIL.Image as Image
from .util import Equirec2Cube


def read_list(list_file):
    rgb_depth_list = []
    with open(list_file) as f:
        lines = f.readlines()
        for line in lines:
            rgb_depth_list.append(line.strip().split(" "))
    return rgb_depth_list


def recover_filename(file_name):

    splits = file_name.split('.')
    rot_ang = splits[0].split('_')[-1]
    file_name = splits[0][:-len(rot_ang)] + "0." + splits[-2] + "." + splits[-1]

    return file_name, int(rot_ang)


class ThreeD60(data.Dataset):  #先去掉数据增强部分，
    """The 3D60 Dataset"""

    def __init__(self, root_dir, list_file, height=256, width=512, disable_color_augmentation=False,
                 disable_LR_filp_augmentation=False, disable_yaw_rotation_augmentation=False, is_training=False):
        """
        Args:
            root_dir (string): Directory of the 3D60 Dataset.
            list_file (string): Path to the txt file contain the list of image and depth files.
            height, width: input size.
            disable_color_augmentation, disable_LR_filp_augmentation,
            disable_yaw_rotation_augmentation: augmentation options.
            is_training (bool): True if the dataset is the training set.
        """
        self.root_dir = root_dir
        self.rgb_depth_list = read_list(list_file)
        self.w = width
        self.h = height
        self.is_training = is_training

        self.e2c = Equirec2Cube(self.h, self.w, self.h // 2)

        #self.color_augmentation = not disable_color_augmentation
        #self.LR_filp_augmentation = not disable_LR_filp_augmentation
        #self.yaw_rotation_augmentation = not disable_yaw_rotation_augmentation

        self.max_depth_meters = 10.0

        # try:
        #     self.brightness = (0.8, 1.2)
        #     self.contrast = (0.8, 1.2)
        #     self.saturation = (0.8, 1.2)
        #     self.hue = (-0.1, 0.1)
        #     self.color_aug= transforms.ColorJitter.get_params(
        #         self.brightness, self.contrast, self.saturation, self.hue)
        # except TypeError:
        #     self.brightness = 0.2
        #     self.contrast = 0.2
        #     self.saturation = 0.2
        #     self.hue = 0.1
        #     self.color_aug = transforms.ColorJitter.get_params(
        #         self.brightness, self.contrast, self.saturation, self.hue)

        self.to_tensor = transforms.ToTensor()
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    def __len__(self):
        return len(self.rgb_depth_list)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        inputs = {}
        #Unifuse是有对图片进行旋转
        # rgb_name, rot_ang = recover_filename(os.path.join(self.root_dir, self.rgb_depth_list[idx][0]))
        item = {}
        if (idx >= self.length):
            print("Index [{}] out of range. Dataset length: {}".format(idx, self.length))
        else:
            leftRGB = Image.open(self.sample["leftRGB"][idx])
            upRGB = Image.open(self.sample["upRGB"][idx])

            dtmp = np.array(cv2.imread(self.sample["leftDepth"][idx], cv2.IMREAD_ANYDEPTH))
            depth = torch.from_numpy(dtmp)
            depth.unsqueeze_(0)


            dtmp = np.array(cv2.imread(self.sample["upDepth"][idx], cv2.IMREAD_ANYDEPTH))
            up_depth = torch.from_numpy(dtmp)
            up_depth.unsqueeze_(0)


            item =  {
                "leftRGB": self.pilToTensor(leftRGB),
                ""
                "upRGB": self.pilToTensor(upRGB),
                "leftDepth": depth,
                "upDepth": up_depth,
                'leftDepth_filename': os.path.basename(self.sample['leftDepth'][idx][:-4])
            }
        return item





        rgb = cv2.imread(rgb_name)
        rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
        rgb = cv2.resize(rgb, dsize=(self.w, self.h))

        depth_name, _ = recover_filename(os.path.join(self.root_dir, self.rgb_depth_list[idx][1]))
        gt_depth = cv2.imread(depth_name, cv2.IMREAD_ANYDEPTH)
        gt_depth = cv2.resize(gt_depth, dsize=(self.w, self.h), interpolation=cv2.INTER_NEAREST)
        gt_depth[gt_depth>self.max_depth_meters] = self.max_depth_meters + 1

        # if self.is_training and self.yaw_rotation_augmentation:
        #     # random rotation
        #     roll_idx = random.randint(0, self.w//4) + (self.w*rot_ang)//360
        # else:
        #     roll_idx = (self.w * rot_ang) // 360

        #rgb = np.roll(rgb, roll_idx, 1)
        #gt_depth = np.roll(gt_depth, roll_idx, 1)

        # if self.is_training and self.LR_filp_augmentation and random.random() > 0.5:
        #     rgb = cv2.flip(rgb, 1)
        #     gt_depth = cv2.flip(gt_depth, 1)


        # if self.is_training and self.color_augmentation and random.random() > 0.5:
        #     aug_rgb = np.asarray(self.color_aug(transforms.ToPILImage()(rgb)))
        # else:
        #     aug_rgb = rgb

        #cube_rgb, cube_gt_depth = self.e2c.run(rgb, gt_depth[..., np.newaxis])
        cube_rgb = self.e2c.run(rgb)
       
        
        
        
        
        #cube_aug_rgb = self.e2c.run(aug_rgb)

        rgb = self.to_tensor(rgb.copy())
        cube_rgb = self.to_tensor(cube_rgb.copy())
        #aug_rgb = self.to_tensor(aug_rgb.copy())
        #cube_aug_rgb = self.to_tensor(cube_aug_rgb.copy())

        inputs["rgb"] = rgb
        #inputs["normalized_rgb"] = self.normalize(aug_rgb)

        inputs["cube_rgb"] = cube_rgb
        #inputs["normalized_cube_rgb"] = self.normalize(cube_aug_rgb)

        inputs["gt_depth"] = torch.from_numpy(np.expand_dims(gt_depth, axis=0))
        inputs["val_mask"] = ((inputs["gt_depth"] > 0) & (inputs["gt_depth"] <= self.max_depth_meters)
                                & ~torch.isnan(inputs["gt_depth"]))

        """
        cube_gt_depth = torch.from_numpy(np.expand_dims(cube_gt_depth[..., 0], axis=0))
        inputs["cube_gt_depth"] = cube_gt_depth
        inputs["cube_val_mask"] = ((cube_gt_depth > 0) & (cube_gt_depth <= self.max_depth_meters)
                                   & ~torch.isnan(cube_gt_depth))
        """
        return inputs



