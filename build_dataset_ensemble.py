'''
 * Copyright (c) 2022, salesforce.com, inc.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 * For full license text, see LICENSE.txt file in the repo root or https://opensource.org/licenses/BSD-3-Clause
 * By Junnan Li
 * References: timm, simmim, and slip
 * https://github.com/rwightman/pytorch-image-models/tree/master/timm
 * https://github.com/microsoft/SimMIM/
 * https://github.com/facebookresearch/SLIP
'''

import os
import os.path
import torch
import json
from transformers import CLIPTokenizer
from transformers import AutoTokenizer

from torchvision import datasets, transforms
from torchvision.transforms.functional import InterpolationMode 
from torch.utils.data import Dataset
from unicl.constants import IMAGENET_CLASSES
from unicl.constants import prompt_engineering
import os
import os.path
from typing import Any, Callable, cast, Dict, List, Optional, Tuple, Union

from PIL import Image
from timm.data import create_transform

from PIL import Image
import numpy as np
from unicl.text_encoder import build_tokenizer 
import random

def pil_loader(path: str) -> Image.Image:
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')
    
import random
import math
import numpy as np


IMG_EXTENSIONS = (".jpg", ".jpeg", ".png", ".ppm", ".bmp", ".pgm", ".tif", ".tiff", ".webp")


def pil_loader(path: str) -> Image.Image:
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, "rb") as f:
        img = Image.open(f)
        return img.convert("RGB")


# TODO: specify the return type
def accimage_loader(path: str) -> Any:
    import accimage

    try:
        return accimage.Image(path)
    except OSError:
        # Potentially a decoding problem, fall back to PIL.Image
        return pil_loader(path)


def default_loader(path: str) -> Any:
    from torchvision import get_image_backend

    if get_image_backend() == "accimage":
        return accimage_loader(path)
    else:
        return pil_loader(path)


class TextImageFolder(torchvision.datasets.ImageFolder):
    def __init__(
        self,
        root: str,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        loader: Callable[[str], Any],
        unicl_config = None ):
        super().__init__(
            root,
            loader=loader,
            transform=transform,
            target_transform=target_transform,,
        )
        self.imgs = self.samples
        self.config = unicl_config
        self.tokenizer = build_tokenizer(self.config)
    def __getitem__(self,index):
        path, target = self.samples[index]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)
        text = self._decode_text_from_label(target)
        tokens = self.tokenizer(
            text, padding='max_length', 
            truncation=True, 
            max_length=77,
            return_tensors='pt'
        ) if self.tokenizer else text
        tokens['input_ids'].squeeze_()
        tokens['attention_mask'].squeeze_()
        return sample, tokens, target
    def __len__(self):
        return len(self.samples)
    def _decode_text_from_label(self, label):
        concept = IMAGENET_CLASSES[int(label)]
        text = prompt_engineering(concept)
        return text

class MaskGenerator:
    def __init__(self, input_size, mask_patch_size, model_patch_size, mask_ratio):
        self.input_size = input_size
        self.mask_patch_size = mask_patch_size
        self.model_patch_size = model_patch_size
        self.mask_ratio = mask_ratio
        
        assert self.input_size % self.mask_patch_size == 0
        assert self.mask_patch_size % self.model_patch_size == 0
        
        self.rand_size = self.input_size // self.mask_patch_size
        self.scale = self.mask_patch_size // self.model_patch_size
        
        self.token_count = self.rand_size ** 2
        
        self.mask_count = int(np.ceil(self.token_count * self.mask_ratio))
        
    def __call__(self):

        mask_idx = np.random.permutation(self.token_count)[:self.mask_count]
        mask = np.zeros(self.token_count, dtype=int)
        mask[mask_idx] = 1
        
        mask = mask.reshape((self.rand_size, self.rand_size))
        mask = mask.repeat(self.scale, axis=0).repeat(self.scale, axis=1)
        
        return mask
    
class DataAugmentation:
    def __init__(self, weak_transform, strong_transform, args, train_config):
        self.transforms = [weak_transform, strong_transform]
        
        self.mask_generator = MaskGenerator(
            input_size=args.input_size,       
            mask_patch_size=train_config['mask_patch_size'],
            model_patch_size=train_config['model_patch_size'],
            mask_ratio=train_config['mask_ratio'],
        )      

    def __call__(self, x):
        images_weak = self.transforms[0](x)
        images_strong = self.transforms[1](x)
        
        return images_weak, images_strong, self.mask_generator()
    
    
class FileListDatasetText(Dataset):
    def __init__(self, image_file, label_file,  unicl_config = None,transform=None, target_transform=None):
        self.transform = transform
        self.target_transform = target_transform
        self.images = np.load(image_file)
        self.labels = np.load(label_file)
        self.config = unicl_config
        self.tokenizer = build_tokenizer(self.config)

    def __getitem__(self, index):
        path, target = self.samples[index]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)
        text = self._decode_text_from_label(target)
        tokens = self.tokenizer(
            text, padding='max_length', 
            truncation=True, 
            max_length=77,
            return_tensors='pt'
        ) if self.tokenizer else text
        tokens['input_ids'].squeeze_()
        tokens['attention_mask'].squeeze_()
        return sample, tokens, target

        return sample, target

    def __len__(self):
        return len(self.images)
    
    
def build_dataset(is_train, args, train_config=None):
    transform = build_transform(is_train, args, train_config)

    print("Transform = ")
    if isinstance(transform, tuple):
        for trans in transform:
            print(" - - - - - - - - - - ")
            for t in trans.transforms:
                print(t)
    elif isinstance(transform, DataAugmentation):     
        for T in transform.transforms:
            print(" - - - - - - - - - - ")
            for t in T.transforms:
                print(t)   
    else:
        for t in transform.transforms:
            print(t)                
    print("---------------------------")

    catalog = json.load(open('dataset_catalog.json','r'))
    assert args.dataset in catalog.keys(), "Dataset %s is not implemented"%args.dataset
    
    entry = catalog[args.dataset]
    if entry['type'] == 'special':
        if args.dataset == 'cifar10':
            dataset = datasets.CIFAR10(entry['path'], train=is_train, transform=transform, download=True)
        elif args.dataset == 'cifar100':
            dataset = datasets.CIFAR100(entry['path'], train=is_train, transform=transform, download=True)              
    elif entry['type']=='imagefolder':
        dataset = datasets.ImageFolder(os.path.join(entry['path'], entry['train'] if is_train else entry['test']), 
                                       transform=transform)
    else:     
        path = entry['train'] if is_train else entry['test']
        image_file = os.path.join(entry['path'], path + '_images.npy')
        label_file = os.path.join(entry['path'], path + '_labels.npy')
        target_transform = None
        dataset = FileListDataset(image_file, label_file, transform, target_transform)    

    return dataset


def build_transform(is_train, args, train_config=None):

    if is_train:
        weak_transform = transforms.Compose([
            transforms.Resize(args.input_size, interpolation=InterpolationMode.BICUBIC),       
            transforms.RandomCrop(args.input_size),  
            transforms.ToTensor(),
            transforms.Normalize(
                mean=torch.tensor(args.image_mean),
                std=torch.tensor(args.image_std))
        ])
         
        strong_transform = create_transform(
            input_size=args.input_size,
            scale=(args.train_crop_min,1),
            is_training=True,
            color_jitter=args.color_jitter,
            auto_augment=args.aa,
            interpolation=args.train_interpolation,
            mean=args.image_mean,
            std=args.image_std
        )          
        transform = DataAugmentation(weak_transform, strong_transform, args, train_config)

        return transform
    
    else:
        transform = transforms.Compose([
            transforms.Resize(args.input_size, interpolation=InterpolationMode.BICUBIC),
            transforms.CenterCrop(args.input_size),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=torch.tensor(args.image_mean),
                std=torch.tensor(args.image_std))
        ])
        return transform