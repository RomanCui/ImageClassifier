Image recognition project using supervised learning

Scope: 
This personal project applies different supervised learning methods on datasets MNIST and CIFAR10.

Tools used:
Python, Pytorch, numpy, Google Colab

Timeline: 
Sep - Dec 2022

Different methods have different accuracy and taining time.
Here is a summary of the methods used so far :

1. kNN on MNIST.
   Accuracy: 96%

2. Multiple logistic regression on both MNIST and CIFAR10.
   Accuracy: 92% on MNIST and 38% on CIFAR10
   
3. Fully Connected Network on MNIST
   Accuracy: 95% on MNIST with L2 loss
   Accuracy: 97% on MNIST with cross entropy loss
   Part 1 uses built-in pytorch model 
   Part 2 manually implemented a training model that 
   computes forward pass, backward pass, and loss function manually
   with only a very basic set of pytorch functions
   
4. Convolution Nurual Network on CIFAR10
