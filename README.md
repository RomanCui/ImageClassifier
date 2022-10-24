# Clone and Use 📋

- The website is completely built on `react-js` library of `javascript` and that's why we need `nodejs` and `npm` installed
- While installing `nodejs` and `npm`, try to install versions which are equal or greater than the versions mentioned in badges above
- In case you want to help developing it or simply saving it, you can fork the repository just by clicking the button on the top-right corner of this page
- After the successful installation of `nodejs` and `npm`, clone the repository into your local system using below command:
    ```bash
     git clone https://github.com/ashutosh1919/masterPortfolio.git
    ```
    This will clone the whole repository in your system.
- To download required dependencies to your system, navigate to the directory where the cloned repository resides and execute following command:
    ```node
    npm install
    ```
- Now the project is ready to use
- You can check it using `npm start`, it will open the website locally on your browser.

# Customize it to make your own portfolio ✏️

In this project, there are basically 4 things that you need to change to customize this to anyone else's portfolio: **package.json**, **Personal Information**, **Github Information** and **Splash Logo**.


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
