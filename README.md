# Introduction

- This project applied **supervised learning** techniques on **image recognition** from a beginning level to an advanced level
- Various learning methods and **many different learning models** are developed
- **Different datasets** are used to train and test the models, such as CIFAR10
- In this project, I use **pytorch library** and **google colab**
- Timeline: Sep - Dec 2022

# Summary of programs

1. **kNN** model with MNIST dataset (Accuracy: 96%)
2. **Multiple logistic regression** on MNIST (Accuracy: 91%) and CIFAR10 (Accuracy: 38%)
3. **Fully Connected Network** on MNIST (Accuracy: 95%)
4. **Convolution Nurual Network** on CIFAR10 (Accuracy: 65%)
5. **Tranfer Learning** on FashionMNIST (Accuracy: 83%)

# Design and running results of each program above

### 1. kNN model with MNIST dataset
Codebase: https://github.com/RomanCui/BigBrainModels/tree/main/K-NN <br>
Running instruction: upload knn_on_MNIST.ipynb to google colab and run with gpu

#### Design:
There is no training for the kNN model <br>
In testing, the program find the closes 3 neighbor and compute the mode of their labels as the predition

#### Result:
Correct Predictions: 962/1000 total <br>
Accuracy: 0.962000 <br>
Time: 4.204589

### 2. Multiple logistic regression on MNIST and CIFAR10
Codebase: https://github.com/RomanCui/BigBrainModels/tree/main/LogisticRegression <br>
Running instruction: upload multiple_logistic_regression_with_tuning.ipynb to google colab and run with gpu

#### Design:
First, set up logistic regression and regularization with a starting leaerning rate and regularization hyperparameter <br>
Then, I improved the performance by using grid-search to find the best learning rate and hyperparmeters

#### Result:
Best lambda for adam:  0.005 <br>
Best learning rate for adam:  0.001 <br>
Best lambda for sgd :  0.005 <br>
Best learning rate for sgd:  0.001 <br>
Accuracy of the network on the 10000 test images: 91 % <br>
Accuracy of the network on the 10000 test images: 38 %

### 3. Fully Connected Network on MNIST
Codebase: https://github.com/RomanCui/BigBrainModels/tree/main/FCNet <br>
Running instruction: run main.py and specify implementation type (builin or manual) and loss type (l2 or ce), where ce stands for cross entropy loss <br>
example: python3 main.py impl_type=builtin loss_type=ce python3 main.py impl_type=manual loss_type=l2

#### Design:
Part 1: I used built-in pytorch model to construct the fully connected network <br>
Part 2: I manually implemented a training model that computes forward pass, backward pass, and loss function manually with basic math funtions

#### Result:
Accuracy: 95% on MNIST with L2 loss <br>
Accuracy: 97% on MNIST with cross entropy loss

### 4. Convolution Nurual Network on CIFAR10
Codebase: https://github.com/RomanCui/BigBrainModels/tree/main/ConvolutionalNet <br>
Running instruction: upload CNN_CIFAR10.ipynb to google colab and run with gpu

#### Design:
The model has convolutional layers, fully connected layers, activation functions, max-pooling and up-sampling.

#### Result:
Accuracy: 65% on CIFAR-10

### 5. Tranfer Learning on FashionMNIST

### 6. Object Detection
Codebase: https://github.com/RomanCui/BigBrainModels/tree/main/ConvolutionalNet <br>


