#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# */AIPND-project/pyproject-classify-flower-images/classifier.py
                                                                            
# PROGRAMMER: Chimela Caesar
# DATE CREATED: 09/06/2023                                 
# REVISED DATE: 
# PURPOSE: This set of functions and classes are relating to the model.
#
##

import time
import os
from tempfile import TemporaryDirectory

import torch
from torch import nn
from torch import optim
from torchvision import models
from torch.optim import lr_scheduler

resnet152 = models.resnet152(pretrained=True)
densenet201 = models.densenet201(pretrained=True)
vgg19 = models.vgg19(pretrained=True)

models = {'densenet201': densenet201, 'vgg19': vgg19}

def get_classifier(arch, hidden_units):
    ''' Create a classifier for a chosen model architecture.
    '''  
    input_units = {'densenet201': 1920, 'vgg19': 25088}   
    classifier = nn.Sequential(nn.Linear(input_units[arch], hidden_units), 
                                     nn.ReLU(),
                                     nn.Dropout(0.2),
                                     nn.Linear(hidden_units, 102),
                                     nn.LogSoftmax(dim=1))  
    return classifier
    

# Build the network
def build(arch, learning_rate, hidden_units, gpu):
    """
    Builds the neural network and creates a dictionary of parameters useful for
    model training.
    Parameters:
     arch - CNN model architecture (string)
     learning_rate - the model learning rate hyperparameter (float)
     hidden_units - the model hidden units hyperparameter (int)
     gpu - use GPU for training (bool)
    Returns:
      train_params_dic - Dictionary with keys 'model', 'criterion', 'optimizer', 
      'scheduler', and 'device', and values as objects with corresponding names.
    """
    
    # Use GPU if it's chosen
    device = torch.device("cuda" if (gpu and torch.cuda.is_available()) else "cpu")

    model = models[arch]

    # Freeze parameters so we don't backprop through them
    for param in model.parameters():
        param.requires_grad = False
    
    model.classifier = get_classifier(arch, hidden_units)

    criterion = nn.NLLLoss()

    # Only train the classifier parameters, feature parameters are frozen
    optimizer = optim.Adam(model.classifier.parameters(), learning_rate)

    # Decay LR by a factor of 0.1 every 4 epochs
    scheduler = lr_scheduler.StepLR(optimizer, step_size=4, gamma=0.1)

    model.to(device);
    
    train_params_dic = {'model': model,
                        'criterion': criterion,
                        'optimizer': optimizer,
                        'scheduler': scheduler,
                        'device': device}     
    
    return train_params_dic

# Train the network
# this code was taken from this page with modification:
# https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html#load-data
# Author: Sasank Chilamkurthy
def train(data, train_params, num_epochs):
    """
    Trains the neural network and prepares a model for prediction.
    Parameters:
     data - Dictionary with keys 'image_datasets', 'dataloaders', 
            'dataset_sizes', and 'class_names', and values as objects with 
            corresponding names.
     train_params - Dictionary with keys 'model', 'criterion', 'optimizer', 
                    'scheduler', and 'device', and values as objects with corresponding names.
     num_epochs - the training epochs hyperparameter (int)
    Returns:
      train_params_dic - Dictionary with keys 'model', 'criterion', 'optimizer', 
      'scheduler', and 'device', and values as objects with corresponding names.
    """
    
    start = time.time()
    
    model = train_params['model']
    criterion = train_params['criterion']
    optimizer = train_params['optimizer']
    scheduler = train_params['scheduler']
    device = train_params['device']
    
    dataloaders = data['dataloaders']
    dataset_sizes = data['dataset_sizes']
    
    # Create a temporary directory to save training checkpoints
    with TemporaryDirectory() as tempdir:
        best_model_params_path = os.path.join(tempdir, 'best_model_params.pth')

        torch.save(model.state_dict(), best_model_params_path)
        best_accuracy = 0.0

        for epoch in range(num_epochs):    
            print(f'Epoch {epoch}/{num_epochs - 1}')
            print('-' * 10)

            # Each epoch has a training and validation phase
            for phase in ['train', 'valid']:
                if phase == 'train':
                    model.train()  # Set model to training mode
                else:
                    model.eval()   # Set model to evaluate mode

                running_loss = 0.0
                running_corrects = 0

                # Iterate over data.
                for inputs, labels in dataloaders[phase]:
                    inputs = inputs.to(device)
                    labels = labels.to(device)

                    # zero the parameter gradients
                    optimizer.zero_grad()

                    # forward
                    # track history if only in train
                    with torch.set_grad_enabled(phase == 'train'):
                        outputs = model(inputs)
                        _, preds = torch.max(outputs, 1)
                        loss = criterion(outputs, labels)

                        # apply backward and optimize only if in training phase
                        if phase == 'train':
                            loss.backward()
                            optimizer.step()

                    # statistics
                    running_loss += loss.item() * inputs.size(0)
                    running_corrects += torch.sum(preds == labels.data)   
                if phase == 'train':
                    scheduler.step()

                epoch_loss = running_loss / dataset_sizes[phase]
                epoch_accuracy = running_corrects.double() / dataset_sizes[phase]

                print(f'{phase} Loss: {epoch_loss:.4f} Accuracy: {epoch_accuracy:.4f}')

                # deep copy the model
                if phase == 'valid' and epoch_accuracy > best_accuracy:
                    best_accuracy = epoch_accuracy
                    torch.save(model.state_dict(), best_model_params_path)

            print()

        time_elapsed = time.time() - start
        print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
        print(f'Best Validation Accuracy: {best_accuracy:4f}')

        # load best model weights
        model.load_state_dict(torch.load(best_model_params_path))
        
    return model

# Save the trained model
def save(model, filepath, params):
    """
    Saves a trained model to the folder specified in the filepath parameter.
    Parameters:
     model - model to be saved
     filepath - the (full) path to the folder to store model checkpoint (string)
     params - Dictionary with variable keys, and values as objects with 
              corresponding names. It provides extra parameters for the checkpoint.
    Returns:
      None
    """
        
    image_datasets = params['image_datasets']
    optimizer = params['optimizer']
    arch = params['arch']
    hidden_units = params['hidden_units']
    learning_rate = params['learning_rate']
    epochs = params['epochs']

    model.class_to_idx = image_datasets['train'].class_to_idx
    model.cpu()
    
    checkpoint = {'arch': arch,
                  'hidden_units': hidden_units,
                  'learning_rate': learning_rate,
                  'epochs': epochs,
                  'class_to_idx': model.class_to_idx,
                  'optimizer_state_dict': optimizer.state_dict(),
                  'state_dict': model.state_dict()}
    
    if not os.path.exists(filepath):
        os.mkdir(filepath)

    torch.save(checkpoint, filepath + '/checkpoint.pth')
    
# Load a trained model    
def load(filepath):
    """
    Loads a trained model from checkpoint specified in the filepath parameter.
    Parameters:
     filepath - the (full) path to the model checkpoint (string)
    Returns:
      model_dic - Dictionary with keys 'model', 'criterion', 'optimizer', 
      'scheduler', and 'checkpoint', and values as objects with corresponding names.
    """
    
    checkpoint = torch.load(filepath)
        
    arch = checkpoint['arch']
    hidden_units = checkpoint['hidden_units']
    learning_rate = checkpoint['learning_rate']
    
    model = models[arch]    
    
    # Freeze parameters so we don't backprop through them
    for param in model.parameters():
        param.requires_grad = False
    
    model.classifier = get_classifier(arch, hidden_units)

    criterion = nn.NLLLoss()

    # Only train the classifier parameters, feature parameters are frozen
    optimizer = optim.Adam(model.classifier.parameters(), learning_rate)
    
    # Decay LR by a factor of 0.1 every 4 epochs
    scheduler = lr_scheduler.StepLR(optimizer, step_size=4, gamma=0.1)

    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    model.load_state_dict(checkpoint['state_dict'])
    model.class_to_idx = checkpoint['class_to_idx']

    model_dic = {'model': model,
                 'criterion': criterion,
                 'optimizer': optimizer,
                 'scheduler': scheduler,
                 'checkpoint': checkpoint}
    
    return model_dic

def predict(img, model, cat_to_name, topk, gpu):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    
    # Use GPU if it's chosen
    device = torch.device("cuda" if (gpu and torch.cuda.is_available()) else "cpu")
    model.to(device);
    
    # Convert Numpy to Tensor
    image_tensor = torch.from_numpy(img).type(torch.FloatTensor)
    image_tensor = image_tensor.cuda() if gpu else image_tensor
    
    # Add batch of size 1 to image
    model_input = image_tensor.unsqueeze(0)
    
    # Probabilities
    probs = torch.exp(model.forward(model_input))
    
    # Top probabilities
    top_probs, top_labs = probs.topk(topk)
    top_probs = top_probs.detach().cpu().numpy().tolist()[0] 
    top_labs = top_labs.detach().cpu().numpy().tolist()[0]
    
    # Convert indices to classes
    idx_to_class = {val: key for key, val in    
                                      model.class_to_idx.items()}
    
    top_labels = [idx_to_class[lab] for lab in top_labs]
    top_flowers = [cat_to_name[idx_to_class[lab]] for lab in top_labs]
    
    return top_probs, top_labels, top_flowers