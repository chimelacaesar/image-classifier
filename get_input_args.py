#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# */AIPND-project/pyproject-classify-flower-images/train.py
                                                                            
# PROGRAMMER: Chimela Caesar
# DATE CREATED: 09/04/2023                                 
# REVISED DATE:  
# PURPOSE: Creates a function that retrieves the following 6 command line inputs 
#          from the user using the Argparse Python module. If the user fails to 
#          provide some of the 6 inputs, then the default values are
#          used for the missing inputs. Command Line Arguments:
#     1. Image Folder as data directory
#     2. Checkpoint Folder as --save_dir with default value 'saved_models'
#     3. CNN Model Architecture as --arch with default value 'vgg19'
#     4. Model Learning Rate Hyperparameter as --learning_rate with default value 0.001
#     5. Model Hidden Units Hyperparameter as --hidden_units with default value 4096
#     6. Model Epochs Hyperparameter as --epochs with default value 2
#
##
# Imports python modules
import argparse

def get_input_args():
    """
    Retrieves and parses the 7 command line arguments provided by the user when
    they run the train program from a terminal window. This function uses Python's 
    argparse module to create and define these 7 command line arguments. If 
    the user fails to provide some of the 7 arguments, then the default 
    values are used for the missing arguments. 
    Command Line Arguments:
      1. Image Folder as data directory
      2. Checkpoint Folder as --save_dir with default value 'saved_models'
      3. CNN Model Architecture as --arch with default value 'vgg19'
      4. Model Learning Rate Hyperparameter as --learning_rate with default value 0.001
      5. Model Hidden Units Hyperparameter as --hidden_units with default value 4096
      6. Model Epochs Hyperparameter as --epochs with default value 2
      7. Use GPU for training as --gpu with default value 'cpu'
    This function returns these arguments as an ArgumentParser object.
    Parameters:
     None - simply using argparse module to create & store command line arguments
    Returns:
     parse_args() -data structure that stores the command line arguments object  
    """
    # Create Parse using ArgumentParser
    parser = argparse.ArgumentParser()
    
    # Create 7 command line arguments as mentioned above using add_argument() from ArguementParser method
    # Argument 1: that's a path to a folder
    parser.add_argument('data_dir', type = str,
                       help = 'path to the folder of flower images')
    
    # Argument 2: that's a path to a folder that stores a checkpoint 
    parser.add_argument('--save_dir', type = str, default = 'saved_models',
                       help = 'path to the folder to save trained model')
    
    # Argument 3: that's the model architecture 
    parser.add_argument('--arch', type = str, default = 'vgg19', 
                       choices = ['densenet201', 'vgg19'], 
                       help = 'CNN model architecture to use. VGG19 is the default architecture with input units of 25088 and default hidden units of 4096. DenseNet201 is the available alternative architecture with input units of 1920 and recommended hidden units of 512.')
    
    # Argument 4: that's the model learning rate hyperparameter
    parser.add_argument('--learning_rate', type = float, default = 0.001,
                       help = 'model learning rate')
    
    # Argument 5: that's the model hidden units hyperparameter
    parser.add_argument('--hidden_units', type = int, default = 4096,
                       help = 'number of model hidden units')
    
    # Argument 6: that's the model epochs hyperparameter
    parser.add_argument('--epochs', type = int, default = 2,
                       help = 'number of training epochs')
    
    # Argument 7: use GPU for training 
    parser.add_argument('--gpu', action = 'store_true',
                       help = 'use GPU for training')
    
    # Return parser.parse_args() parsed argument collection created 
    return parser.parse_args()

def get_input_args_pred():
    """
    Retrieves and parses the 5 command line arguments provided by the user when
    they run the predict program from a terminal window. This function uses Python's 
    argparse module to create and define these 5 command line arguments. If 
    the user fails to provide some of the 5 arguments, then the default 
    values are used for the missing arguments. 
    Command Line Arguments:
      1. Path to image to predict
      2. Model checkpoint
      3. Top K most likely classes as --top_k with default value 5
      4. Use a mapping of categories to real names as --category_names with default 
         value 'cat_to_name.json'
      5. Use GPU for inference as --gpu with default value 'cpu'
    This function returns these arguments as an ArgumentParser object.
    Parameters:
     None - simply using argparse module to create & store command line arguments
    Returns:
     parse_args() -data structure that stores the command line arguments object  
    """
    # Create Parse using ArgumentParser
    parser = argparse.ArgumentParser()
    
    # Create 5 command line arguments as mentioned above using add_argument() from ArguementParser method
    # Argument 1: that's a path to an image
    parser.add_argument('input', type = str,
                       help = 'path to the flower image')
    
    # Argument 2: that's a path to the model checkpoint 
    parser.add_argument('checkpoint', type = str, 
                       help = 'path to the model checkpoint')
    
    # Argument 3: that's the top K most likely classes 
    parser.add_argument('--top_k', type = int, default = 5,  
                       help = 'top K most likely classes')
    
    # Argument 4: use a mapping of categories to real names
    parser.add_argument('--category_names', type = str, default = 'cat_to_name.json',
                       help = 'use a mapping of categories to real names')
    
    # Argument 5: use GPU for training 
    parser.add_argument('--gpu', action = 'store_true',
                       help = 'use GPU for training')
    
    # Return parser.parse_args() parsed argument collection created 
    return parser.parse_args()