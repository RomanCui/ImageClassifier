Image recognition project using supervised learning

Scope: 
This personal project applies different supervised learning methods on datasets MNIST and CIFAR10.

Tools used:
Python, Pytorch, numpy, Google Colab

Timeline: 
Sep - Dec 2022

Introduction:
Different methods have different accuracy and taining time.
Here is a summary of the methods used so far :

1. Approach 1: 
   kNN on MNIST.
   Accuracy: 96%
2. Approach 2:
   Multiple logistic regression on both MNIST and CIFAR10.
   Accuracy: 92% on MNIST and 38% on CIFAR10
   
Details:
Approach 1 
   The program read MNIST data from torchvision datasets
   It flattens 28x28 images into 784 sized vectors
   For each test image, the program find the top 3 closest in the training set
   and take the mode of their labels as the predition (kNN, k = 3)
   The accuracy is 96% for all the test set images
   Limatation is that it is not a deep learner, even larger data set is provided,
   the performance will not improve much
   
   
   
   
   
   
