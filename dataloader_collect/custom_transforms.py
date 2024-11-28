import os
import sys
import torch
from PIL import Image
from typing import Optional
import numpy as np
filepath = os.path.split(__file__)[0]
repopath = os.path.split(filepath)[0]
sys.path.append(repopath)


class static_resize:
    # Resize for training
    # size: h x w
    def __init__(self, size=[384, 384], base_size=None):
        self.size = size[::-1]
        self.base_size = base_size[::-1] if base_size is not None else None

    def __call__(self, sample):
        # Resize image
        sample['image'] = sample['image'].resize(self.size, Image.BILINEAR)

        # If 'gt' exists, resize it as well
        if 'gt' in sample and sample['gt'] is not None:
            sample['gt'] = sample['gt'].resize(self.size, Image.NEAREST)

        # If 'edge' exists, resize it as well
        if 'edge' in sample and sample['edge'] is not None:
            sample['edge'] = sample['edge'].resize(self.size, Image.NEAREST)

        if self.base_size is not None:
            sample['image_resized'] = sample['image'].resize(self.size, Image.BILINEAR)
            if 'gt' in sample and sample['gt'] is not None:
                sample['gt_resized'] = sample['gt'].resize(self.size, Image.NEAREST)
            if 'edge' in sample and sample['edge'] is not None:
                sample['edge_resized'] = sample['edge'].resize(self.size, Image.NEAREST)

        return sample
class tonumpy:
    def __init__(self):
        pass

    def __call__(self, sample):
        for key in sample.keys():
            if key in ['image', 'image_resized', 'gt', 'gt_resized', 'edge', 'edge_resized']:
                sample[key] = np.array(sample[key], dtype=np.float32)

        return sample
class normalize:
    def __init__(self, mean: Optional[list] = None, std: Optional[list] = None, div=255):
        self.mean = mean if mean is not None else 0.0
        self.std = std if std is not None else 1.0
        self.div = div

    def __call__(self, sample):
        if 'image' in sample.keys():
            sample['image'] /= self.div
            sample['image'] -= self.mean
            sample['image'] /= self.std

        if 'image_resized' in sample.keys():
            sample['image_resized'] /= self.div
            sample['image_resized'] -= self.mean
            sample['image_resized'] /= self.std

        if 'gt' in sample.keys():
            sample['gt'] /= self.div

        if 'gt_resized' in sample.keys():
            sample['gt_resized'] /= self.div

        if 'edge' in sample.keys():
            sample['edge'] /= self.div

        if 'edge_resized' in sample.keys():
            sample['edge_resized'] /= self.div
        return sample


class totensor:
    def __init__(self):
        pass

    def __call__(self, sample):
        if 'image' in sample.keys():
            sample['image'] = sample['image'].transpose((2, 0, 1))
            sample['image'] = torch.from_numpy(sample['image']).float()

        if 'image_resized' in sample.keys():
            sample['image_resized'] = sample['image_resized'].transpose((2, 0, 1))
            sample['image_resized'] = torch.from_numpy(sample['image_resized']).float()

        if 'gt' in sample.keys():
            sample['gt'] = torch.from_numpy(sample['gt'])
            sample['gt'] = sample['gt'].unsqueeze(dim=0)

        if 'gt_resized' in sample.keys():
            sample['gt_resized'] = torch.from_numpy(sample['gt_resized'])
            sample['gt_resized'] = sample['gt_resized'].unsqueeze(dim=0)

        if 'edge' in sample.keys():
            sample['edge'] = torch.from_numpy(sample['edge'])
            sample['edge'] = sample['edge'].unsqueeze(dim=0)

        if 'edge_resized' in sample.keys():
            sample['edge_resized'] = torch.from_numpy(sample['edge_resized'])
            sample['edge_resized'] = sample['edge_resized'].unsqueeze(dim=0)
        return sample
