#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# */AIPND-project/pyproject-classify-flower-images/train.py
                                                                            
# PROGRAMMER: Chimela Caesar
# DATE CREATED: 09/04/2023                                 
# REVISED DATE: 
# PURPOSE: Trains a new network on a data set of flower images, using a pretrained
#          CNN model for feature detection, and saves the model as a checkpoint. 
#          Prints out training loss, validation loss, and validation accuracy as 
#          the network trains. 
#
# Uses argparse Expected Call with <> indicating expected user input:
#      python train.py <directory with images>
#             --save_dir <directory to save checkpoint>
#             --arch <model>
#             --learning_rate <hyperparameter for learning rate>
#             --hidden_units <hyperparameter for number of hidden units>
#             --epochs <hyperparameter for number of epochs>
#             --gpu <use GPU for training>
#   Example call:
#    python train.py flowers --save_dir saved_models --arch vgg19 
#           --learning_rate 0.001  --hidden_units 4096  --epochs 20
#           --gpu
##

# Imports functions created for this program
from get_input_args import get_input_args
from image_utils import load_data
from classifier import build
from classifier import train
from classifier import save

# Main program function defined below
def main():
    
    # Collection of the command line arguments from the function call
    in_arg = get_input_args()

    # This function creates the data dictionary that contains the image datasets,
    # dataloaders, dataset sizes, and class names
    data = load_data(in_arg.data_dir)

    # Builds the classifier and returns associated objects for model training    
    train_params = build(in_arg.arch, in_arg.learning_rate, in_arg.hidden_units, in_arg.gpu)
    
    # Trains the classifier and returns a trained model    
    model = train(data, train_params, in_arg.epochs)
    
    # Dictionary with variable number of entries that provides extra 
    # parameters for the checkpoint.
    params = {'image_datasets': data['image_datasets'],
              'optimizer': train_params['optimizer'],
              'arch': in_arg.arch,
              'hidden_units': in_arg.hidden_units,
              'learning_rate': in_arg.learning_rate,
              'epochs': in_arg.epochs}
    
    # Saves the trained model
    save(model, in_arg.save_dir, params)

# Call to main function to run the program
if __name__ == "__main__":
    main()
