# 🌐 ERA1 Session 11 Assignment 🌐

## 📌 Table of Contents

1. [Problem Statement](#problem-statement)
2. [Solution](#Solution)
3. [Grad-CAM concepts](#Grad-CAM-concepts)
4. [Training Loss](#Training-Loss)
5. [Results](#results)
6. [Classwise Accuracy](#classwise-accuracy)
7. [Misclassified Images](#Misclassified-Images)
8. [Grad-CAM Heatmaps on Misclassified Images](#Grad-CAM-Heatmaps-on-Misclassified-Images)

## 🎯 Problem Statement

1. Check this Repo out: https://github.com/kuangliu/pytorch-cifar  
2. (Optional) You are going to follow the same structure for your Code (as a reference). So Create:
    1. models folder - this is where you'll add all of your future models. Copy resnet.py into this folder, this file should only have ResNet 18/34 models. Delete Bottleneck Class
    2. main.py - from Google Colab, now onwards, this is the file that you'll import (along with the model). Your main file shall be able to take these params or you should be able to pull functions from it and then perform operations, like (including but not limited to):
         1. training and test loops
         2. data split between test and train
         3. epochs
         4. batch size
         5. which optimizer to run
         6. do we run a scheduler?

    3. utils.py file (or a folder later on when it expands) - this is where you will add all of your utilities like:  
        1. image transforms,
        2. gradcam,
        3. misclassification code,
        4. tensorboard related stuff
        5. advanced training policies, etc
        6. etc

3. Your assignment is to build the above training structure. Train ResNet18 on Cifar10 for 20 Epochs. The assignment must:
    1. pull your Github code to google colab (don't copy-paste code)
    2. prove that you are following the above structure
    3. that the code in your google collab notebook is NOTHING.. barely anything. There should not be any function or class that you can define in your Google Colab Notebook. Everything must be imported from all of your other files
    4. your colab file must:
         1. train resnet18 for 20 epochs on the CIFAR10 dataset
         2. show loss curves for test and train datasets
         3. show a gallery of 10 misclassified images
         4. show gradcamLinks to an external site. output on 10 misclassified images. Remember if you are applying GradCAM on a channel that is less than 5px, then please don't bother to submit the assignment. 😡🤬🤬🤬🤬  

5. Once done, upload the code to GitHub, and share the code. This readme must link to the main repo so we can read your file structure.
6. Train for 20 epochs
7. Get 10 misclassified images
8. Get 10 GradCam outputs on any misclassified images (remember that you MUST use the library we discussed in the class)
9. Apply these transforms while training:
          1. RandomCrop(32, padding=4)
          2. CutOut(16x16)

## 📚 Solution

The goal of this assignment is to use the ResNet18 & ResNet34 architecture code by the authors of ResNet and apply it on CIFAR10 dataset by meeting the requirements defined below:
### 1. Data Augmentation: 
RandomCrop 32, 32 (after padding of 4), Followed by CutOut(16, 16) --> Here the cutout is almost half the size of the image. 
### 2. Functions to be implemented in utility.py
- gradcam --> which helps in explaining where the deep learning model is looking at in the image inorder to classify in to a particular class.
- misclassification code --> to identify and flag the images which are predicted wrongly by the model
### 3. Training conditions:
- train resnet18 for 20 epochs on the CIFAR10 dataset
- Show loss curves for train & testsets
- Since we are using the ResNet18 original architecture the size of the channel after the last convolution layer is 7x7.  

The CIFAR10 dataset consists of 60,000 32x32 color training images and 10,000 test images, labeled into 10 classes. The 10 classes represent airplanes, cars, birds, cats, deer, dogs, frogs, horses, ships, and trucks. The dataset is divided into 50,000 training images and 10,000 validation images.

## Grad-CAM Concept & Steps
1. Where can we apply Grad-CAM?
    - Irrespective of the downstream task as long as the network involves Convolutional layers and the input is an image we can use Grad-CAM to identify where in a particular image the algorithm has looked in to make a particular decision.
    - Grad-CAM is applicable to a wide variety of CNN model-families:
          (1) CNNs with fully-connected layers (e.g. VGG),
          (2) CNNs used for structured outputs (e.g. captioning),
          (3) CNNs used in tasks with multi-modal inputs (e.g. visual question answering) or reinforcement learning, all without architectural changes or re-training.

2. What are the benefits of Grad-CAM?
    - It can be applied to any CNN-based architecture without any changes
    - It helps in interpreting the results and understanding the failure modes of the current architecture and provides direction for the improvement of the architecture.
    - Localizes class-discriminative regions --> If there are multiple objects within an image and we are interested in finding the location in the image where a particular class is present then Grad-CAM very accurately shows only the location of the class of interest.

3. What are the broad steps involved in Grad-CAM? - Source: https://coderzcolumn.com/tutorials/artificial-intelligence/pytorch-grad-cam  
      1. Capture the output of the last convolution layer of the network.
      2. Take gradient of last convolution layer with respect to prediction probability. (We can take predictions with respect to any class we want. In our case, we'll take prediction with the highest probability. We can look at other probabilities as well).
      3. Average gradients calculated in the previous step at axis which has the same dimension as output channels of last convolution layer. The output of this step will be 1D array that has the same numbers as that of output channels of the last convolution layer.
      4. Multiply convolution layer output with averaged gradients from the previous step at output channel level, i.e. first channel output should be multiplied with first averaged value, second should be multiplied with the second value, and so on.
      5. Average output from the previous step at channel level to create 2D heatmap that has the same dimension as that of image.
      6. Normalize heatmap (Optional step but recommended as it helps improve results).
     
<img width="859" alt="Screenshot 2023-07-29 at 12 29 47 AM" src="https://github.com/phaninandula/ERA-Session11/assets/30425824/82bbc340-1baa-423a-b4d8-e9b2697c290a">

## Training Loss
<img width="623" alt="Screenshot 2023-07-29 at 1 01 59 AM" src="https://github.com/phaninandula/ERA-Session11/assets/30425824/1f3cbf41-38a2-48f4-bfa9-71ab95da8b5c">

## 📈 Results

The model was trained for `18 epochs` and achieved an accuracy of `87.26 %` on the test set. The training logs, as well as the output of the torch summary, are included in the notebook.

Training accuracy: `84.046 %`
Test accuracy: `87.81 %`

## 📊 Classwise Accuracy

<img width="491" alt="Screenshot 2023-07-29 at 1 06 27 AM" src="https://github.com/phaninandula/ERA-Session11/assets/30425824/94e497c8-d91e-4c54-9251-b8cf6d7adc1c">

## Misclassified Images
<img width="1145" alt="Screenshot 2023-07-29 at 1 07 53 AM" src="https://github.com/phaninandula/ERA-Session11/assets/30425824/ccf55c64-76af-4dad-9da1-928aa4adabc0">

## Grad-CAM heatmaps for 10 misclassified Images
<img width="1160" alt="Screenshot 2023-07-29 at 1 08 08 AM" src="https://github.com/phaninandula/ERA-Session11/assets/30425824/558ba599-18b0-48d4-b1d4-8c067f9be09d">

