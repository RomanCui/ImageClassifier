# Introduction

- This project applied **supervised learning** techniques on **image recognition** from a beginning level to an advanced level
- Various learning methods and **many different learning models** are developed
- **Different datasets** are used to train and test the models, such as CIFAR10
- In this project, I use **pytorch library** and **google colab**
- Timeline: Sep - Dec 2022

# Summary of programs with corresponding models and datasets

1. kNN model with MNIST dataset (Accuracy: 96%)
2. **Multiple logistic regression** on MNIST (Accuracy: 92%) and CIFAR10 (Accuracy: 38%)
3. **Fully Connected Network** on MNIST (Accuracy: 95%)
4. **Convolution Nurual Network** on CIFAR10 (Accuracy: 65%)
5. **Tranfer Learning** on FashionMNIST (Accuracy: 83%)

# Design and running results of each program above

## kNN model with MNIST dataset
useful link to understand kNN method https://en.wikipedia.org/wiki/K-nearest_neighbors_algorithm

### Design:
There is no training for the kNN model <br>
In testing, the program find the closes 3 neighbor and compute the mode of their labels as the predition

### Result:
![alt text](https://github.com/RomanCui/BigBrainModels/blob/main/images/knn_result.png?raw=true)

## Multiple logistic regression on MNIST and CIFAR10

## Fully Connected Network on MNIST

## Convolution Nurual Network on CIFAR10

## Tranfer Learning on FashionMNIST



3. Fully Connected Network on MNIST
   Accuracy: 95% on MNIST with L2 loss
   Accuracy: 97% on MNIST with cross entropy loss
   Part 1 uses built-in pytorch model 
   Part 2 manually implemented a training model that 
   computes forward pass, backward pass, and loss function manually
   with only a very basic set of pytorch functions
   
