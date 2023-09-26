#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# */AIPND-project/pyproject-classify-flower-images/predict.py
                                                                            
# PROGRAMMER: Chimela Caesar
# DATE CREATED: 09/08/2023                                 
# REVISED DATE: 
# PURPOSE: Functions to use a trained network for inference. That is, an image is
# passed into the network and it predicts the class of the flower in the image. 
# A function called predict takes an image and a model, then returns the top 𝐾
# most likely classes along with the probabilities. 
#
# Uses argparse Expected Call with <> indicating expected user input:
#      python predict.py <path to image> <checkpoint>
#             --top_k <return top k most likely classes>
#             --category_names <use a mapping of categories to real names>
#             --gpu <use GPU for inference>
#   Example call:
#    python predict.py input checkpoint --top_k 3 --category_names cat_to_name.json 
#           --gpu
##

# Imports functions created for this program
from get_input_args import get_input_args_pred
from image_utils import process_image
from image_utils import get_label_map
from classifier import load
from classifier import predict

# Main program function defined below
def main():
    
    # Collection of the command line arguments from the function call
    in_arg = get_input_args_pred()
    
    # Processes a PIL image for use in a PyTorch model and returns an Numpy array
    img = process_image(in_arg.input)
    
    # Loads a trained model from a checkpoint path and returns a dictionary of 
    # related objects for further model training
    model_dic = load(in_arg.checkpoint)
    
    model = model_dic['model']
    
    # Loads a mapping from category label to category name
    cat_to_name = get_label_map(in_arg.category_names)
    
    # Predicts the image input and returns the flower name and class probability
    print(predict(img, model, cat_to_name, in_arg.top_k, in_arg.gpu))
    
# Call to main function to run the program
if __name__ == "__main__":
    main()    
