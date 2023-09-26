# image-classifier
## Developing an Image Classifier with Deep Learning ##
In the first part of the project, a Jupyter notebook is used to implement an image classifier with PyTorch. The second part of the project involves building a command line application to train the built image classifier and to make predictions with the classifier.
## Description ##
Going forward, AI algorithms will be incorporated into more and more everyday applications. For example, you might want to include an image classifier in a smart phone app. To do this, you'd use a deep learning model trained on hundreds of thousands of images as part of the overall application architecture. A large part of software development in the future will be using these types of models as common parts of applications.

In this project, an image classifier will be trained to recognize different species of flowers. You can imagine using something like this in a phone app that tells you the name of the flower your camera is looking at. In practice you'd train this classifier, then export it for use in your application. We'll be using a dataset of 102 flower categories. If you do not find the flowers/ dataset in the current directory, you can download it using the following commands.

!wget 'https://s3.amazonaws.com/content.udacity-data.com/nd089/flower_data.tar.gz'
!unlink flowers
!mkdir flowers && tar -xzf flower_data.tar.gz -C flowers

The project is broken down into multiple steps:
* Load and preprocess the image dataset
* Train the image classifier on your dataset
* Use the trained classifier to predict image content

With the completed project, you'll have an application that can be trained on any set of labeled images. Here the network will be learning about flowers and end up as a command line application. But, what can be done with the new skills depends on your imagination and effort in building a dataset. For example, imagine an app where you take a picture of a car, it tells you what the make and model is, then looks up information about it.
## Specifications ##
The project includes files `train.py` and `predict.py`. The first file, train.py, will train a new network on a dataset and save the model as a checkpoint. The second file, predict.py, uses a trained network to predict the class for an input image. A file just for functions and classes relating to the model has been created; and another one, for utility functions like loading data and preprocessing images.
### 1. Train ###
Train a new network on a data set with `train.py`

* Basic usage: `python train.py data_directory`
* Prints out training loss, validation loss, and validation accuracy as the network trains
* Options: (+) Set directory to save checkpoints: `python train.py data_dir --save_dir save_directory` (+) Choose architecture: `python train.py data_dir --arch "vgg19"` (+) Set hyperparameters: `python train.py data_dir --learning_rate 0.01 --hidden_units 512 --epochs 20` (*) Use GPU for training: `python train.py data_dir --gpu`
### 2. Predict ###
Predict flower name from an image with predict.py along with the probability of that name. That is, you'll pass in a single image /path/to/image and return the flower name and class probability.

* Basic usage: `python predict.py /path/to/image checkpoint`
* Options: (+) Return top _K_ most likely classes: `python predict.py input checkpoint --top_k 3` (+) Use a mapping of categories to real names: `python predict.py input checkpoint --category_names cat_to_name.json` (+) Use GPU for inference: `python predict.py input checkpoint --gpu`
