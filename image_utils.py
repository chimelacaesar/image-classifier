#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# */AIPND-project/pyproject-classify-flower-images/image_utils.py
                                                                            
# PROGRAMMER: Chimela Caesar
# DATE CREATED: 09/04/2023                                 
# REVISED DATE: 
# PURPOSE: This set of functions are used to load data and preprocess images.
#
##

import os
import json
from PIL import Image
import numpy as np

import torch
from torchvision import datasets, transforms

def load_data(image_dir):
    """
    Loads image data from the image folder provided in the parameter and
    creates a dictionary of various data containers. Image datasets, data loaders, 
    dataset sizes, and class names for the three categories of training, validation,
    and testing, are created.
    Parameters:
     image_dir - The (full) path to the folder of images that are to be
                 used for training, validation, and testing by the neural network (string)
    Returns:
      data_dic - Dictionary with keys 'image_datasets', 'dataloaders', 'dataset_sizes', and
                 'class_names', and values as objects with corresponding names.
    """
    
    # The dataset is split into three parts, training, validation, and testing
    data_dir = image_dir
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'
    
    # Define transforms for the training, validation, and testing sets
    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomRotation(30),
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'valid': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'test': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }
    
    # Custom ImageFolder class to handle malformed image files
    class ImageFolderExt(datasets.ImageFolder):
        def __getitem__(self, index):
            try:
                return super().__getitem__(index)
            except Exception as e:
                print(f'An exception occurred: {e}')
                return None

    # Collation function to filter batch data
    def collate_fn(batch):
        batch = list(filter(lambda x: x is not None, batch))
        return torch.utils.data.dataloader.default_collate(batch)
    
    # Load the datasets with ImageFolder
    image_datasets = {x: ImageFolderExt(os.path.join(data_dir, x),
                                              data_transforms[x])
                      for x in ['train', 'valid', 'test']}

    # Using the image datasets and the trainforms, define the dataloaders
    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], collate_fn=collate_fn, batch_size=16,
                                                 shuffle=True)
                  for x in ['train', 'valid', 'test']}

    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'valid', 'test']}

    class_names = image_datasets['train'].classes
    
    data_dic = {'image_datasets': image_datasets,
                'dataloaders': dataloaders,
                'dataset_sizes': dataset_sizes,
                'class_names': class_names}
    
    return data_dic


def get_label_map(filepath):
    """
    Loads in a mapping from category label to category name. You can find this in the file 
    cat_to_name.json. It's a JSON object which you can read in with the json module. This 
    will give you a dictionary mapping the integer encoded categories to the actual names 
    of the flowers.
    Parameters:
     filepath - the path to the mapping file.
    Returns:
      cat_to_name - Dictionary with keys as category labels, and values as category names.
    """
    
    with open(filepath, 'r') as f:
        cat_to_name = json.load(f)
        
    return cat_to_name    


# this code was taken from this page:
# https://medium.com/@josh_2774/deep-learning-with-pytorch-9574e74d17ad
# Author: Josh Bernhard
def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns a Numpy array
    '''
    
    # Open image
    img = Image.open(image)
    
    # Resize
    if img.size[0] > img.size[1]: # if the width > height
        img.thumbnail((1000000, 256)) # constrain the height to be 256
    else:
        img.thumbnail((256, 200000)) # otherwise constrain the width
    
    # Crop image                  
    left_margin = (img.width-224)/2
    bottom_margin = (img.height-224)/2
    right_margin = left_margin + 224
    top_margin = bottom_margin + 224
                      
    img = img.crop((left_margin, bottom_margin, right_margin,    
                   top_margin))
                      
    # Normalize image
    img = np.array(img)/255                
    mean = np.array([0.485, 0.456, 0.406]) 
    std = np.array([0.229, 0.224, 0.225])                 
    img = (img - mean) / std  
                      
    # Pytorch expects color channel as first dimension
    img = img.transpose((2, 0, 1)) 
                      
    return img  