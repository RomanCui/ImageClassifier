## <div align="center">MNIST and CIFAR-10 Image Classification</div>

<div align="center">
  <p>
     This project applied supervised learning techniques on image classification from a beginning level to an advanced level. Various learning methods and many different learning models are developed. Different datasets such as <a href="https://en.wikipedia.org/wiki/MNIST_database">MNIST</a> and <a href="https://en.wikipedia.org/wiki/CIFAR-10">CIFAR-10</a> are used to train and test the models. This project is a personal project that developed to apply my skills that I gained from CMPUT 328 at University of Alberta with instructor Nilanjan Ray. <br><br>
    Technology used: pytorch library,  google colab <br><br> Timeline: Sep - Dec 2022
  </p>
 
  
</div>

## <div align="center">Documentation</div>

## Summary of programs

1. **kNN** model with MNIST dataset (Accuracy: 96%)
2. **Multiple logistic regression** on MNIST (Accuracy: 91%) and CIFAR10 (Accuracy: 38%)
3. **Fully Connected Network** on MNIST (Accuracy: 95%)
4. **Convolution Nurual Network** on CIFAR10 (Accuracy: 65%)
5. **Object Detection** on MNISTDD-RGB (Accuracy: Classification 98%, IOU 89%)
6. **Image Segmentaion** on MNISTDD-RGB (Accuracy: 94%)

## 1. kNN model with MNIST dataset
Codebase: https://github.com/RomanCui/BigBrainModels/tree/main/K-NN <br>
Running instruction: upload knn_on_MNIST.ipynb to google colab and run with gpu

#### Design:
There is no training for the kNN model <br>
In testing, the program find the closes 3 neighbor and compute the mode of their labels as the predition

#### Result:
MNIST dataset illustration:
<div align="left">
  <p>
    <img width="300" src="https://github.com/RomanCui/BigBrainModels/blob/main/assets/MnistExamples.png"></a>
  </p>
</div>
Correct Predictions: 962/1000 total <br>
Accuracy: 0.962000 <br>
Time: 4.204589

## 2. Multiple logistic regression on MNIST and CIFAR10
Codebase: https://github.com/RomanCui/BigBrainModels/tree/main/LogisticRegression <br>
Running instruction: upload multiple_logistic_regression_with_tuning.ipynb to google colab and run with gpu

#### Design:
First, set up logistic regression and regularization with a starting leaerning rate and regularization hyperparameter <br>
Then, I improved the performance by using grid-search to find the best learning rate and hyperparmeters

#### Result:
CIFAR-10 dataset illustration:
<div align="left">
  <p>
    <img width="300" src="https://github.com/RomanCui/BigBrainModels/blob/main/assets/Cifar10Examples.png"></a>
  </p>
</div>
Tuned best lambda for adam:  0.005 <br>
Tuned best learning rate for adam:  0.001 <br>
Tuned best lambda for sgd :  0.005 <br>
Tuned best learning rate for sgd:  0.001 <br>
Accuracy of the network on the 10000 test images: 91 % <br>
Accuracy of the network on the 10000 test images: 38 %

## 3. Fully Connected Network on MNIST
Codebase: https://github.com/RomanCui/BigBrainModels/tree/main/FCNet <br>
Running instruction: Run main.py and specify implementation type (builin or manual) and loss type (l2 or ce), where ce stands for cross entropy loss <br>
example: ```python3 main.py impl_type=builtin loss_type=ce``` ```python3 main.py impl_type=manual loss_type=l2```

#### Design:
Part 1: I used built-in pytorch model to construct the fully connected network <br>
Part 2: I manually implemented a training model that computes forward pass, backward pass, and loss function manually with basic math funtions

#### Result:
Accuracy: 95% on MNIST with L2 loss <br>
Accuracy: 97% on MNIST with cross entropy loss

## 4. Convolution Nurual Network on CIFAR10
Codebase: https://github.com/RomanCui/BigBrainModels/tree/main/ConvolutionalNet <br>
Running instruction: upload CNN_CIFAR10.ipynb to google colab and run with gpu

#### Design:
The model has convolutional layers, fully connected layers, activation functions, max-pooling and up-sampling.

#### Result:
Accuracy: 65% on CIFAR-10

## 5. Object Detection
Codebase: https://github.com/RomanCui/BigBrainModels/tree/main/ObjectDetection <br>
Running instruction: Download the dataset from https://drive.google.com/drive/u/0/folders/1cBuMzH4ysD_pnJi8MjeDFyy5cH_DJQwK and upload the folder in your google drive. Upload object_detection_train.ipynb to google colab for training and object_detection_test.ipynb for testing. Run with gpu in colab. 

#### Design:
The model uses DarkNet53

#### Result:
MNISTDD-RGB dataset illustration:
<div align="left">
  <p>
    <img width="300" src="https://github.com/RomanCui/BigBrainModels/blob/main/assets/%20MNISTDDExamples.png"></a>
  </p>
</div>
Accuracy: 98% on the classification task <br> 
Accuracy: 89% on the IOU task (success is only true if the IOU predction is exactly the same as the label, all pixels have to overlab. success rate is not calculated based on area)

## 6. Image Segmentation
Codebase: https://github.com/RomanCui/BigBrainModels/tree/main/Segmentation <br>
Running instruction: Download the dataset from https://drive.google.com/drive/u/0/folders/1cBuMzH4ysD_pnJi8MjeDFyy5cH_DJQwK and upload the folder in your google drive. Upload segmentation_train.ipynb to google colab for training and segmentation_test.ipynb for testing. Run with gpu in colab. 

#### Design:
The model uses U-Net

#### Result:
Accuracy: 94% on correct prediction for object pixels. Background pixels are not included in the accuracy calculation.




