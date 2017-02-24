# Tensorflow Workshop
Repository of scripts for the Introduction to Deep Learning and Tensorflow Workshop, 
one of multiple workshops help in preparation for the 2017 Datathon Competition at Brown University.

## Setup ##

This workshop covers the basics of Google's Tensorflow library, a state of the art tool for building deep neural network models.
To get setup, it is important that you have the following things installed on your machines:

+ A working Python installation. The scripts in this repository are all written for Python Version 2.7, but is should
      be fairly straightforward to refactor the code to work with Python 3.3.

+ The latest version of Tensorflow (version 1.0). You can download and install Tensorflow by following the instructions
      here: [Installing Tensorflow](https://www.tensorflow.org/install/)

Once you have the above installed on your machine, clone this repository locally. The master branch contains all the skeleton 
files you will be using during the workshop, and the solutions branch has all of the solutions (for your reference).

## Repository Structure ##

The repository is structured in the following way: 

+ feedforward.py - Contains the code skeleton for the first model described in the workshop - A Two-Layer Feed-Forward Network 
                   for MNIST Handwritten Digit Classification.

+ convolution.py - Contains the code skeleton for the second model described in the workshop - A Convolutional Neural Network (CNN) 
                   again for MNIST Handwritten Digit Classification.

+ recurrent.py - Contains the code skeleton for the last model described in the workshop - A Recurrent Neural Network (RNN) Language Model, for 
                 Language Modeling the Penn Treebank. 

+ data/ - Contains the data files for the Recurrent Neural Network language model described above. This directory has the following subdirectories:
    - raw_data/ - Directory containing the raw train and test files. These are standard, pre-UNKed files from the Penn Treebank.

    - processed_data/ - Directory containing the processed (pickled) train and test files. These store the vectorized format required for the 
                        RNN Language Model.

+ preprocessor/ - Contains the preprocessing script for converting the Penn Treebank data to the format required for the RNN Language Model. 

