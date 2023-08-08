# 🌐 ERA1 Session 12 Assignment 🌐

## 📌 Table of Contents

1. [Problem Statement](#problem-statement)
2. [Learnings](#learnings)
3. [Model Architecture](#model-architecture)
4. [Data Augmentation](#data-augmentation)
5. [Results](#results)
6. [Misclassified Images](#misclassified-images)

## 🎯 Problem Statement

1. Move your S10 assignment to Lightning first and then to Spaces such that:  
    1. (You have retrained your model on Lightning)  
    2. You are using Gradio  
    3. Your spaces app has these features:  
        1. ask the user whether he/she wants to see GradCAM images and how many, and from which layer, allow opacity change as well  
        2. ask whether he/she wants to view misclassified images, and how many  
        3. allow users to upload new images, as well as provide 10 example images  
        4. ask how many top classes are to be shown (make sure the user cannot enter more than 10)  
    4. Add the full details on what your App is doing to Spaces README   
2. Then:  
    1. Submit the Spaces App Link  
    2. Submit the Spaces README link (Space must not have a training code)
    3. Submit the GitHub Link where Lightning Code can be found along with detailed README with log, loss function graphs, and 10 misclassified images

## Learnings - Converting the pytorch code to Lightning Code
As we move the code from pytorch to Lightning code most of the code gets reorganized or already written for us by the Lightning module so the code looks very simple. Below are the comparison of files that are used between Pytorch approach and Lightning.

| Sno.| In Lightning code we dont see |
|-----|-----------------------------------------|
|1.  | Looping over Epochs|
|2.  | Looping over Datasets|
|3.  | Setting `model` to `eval` or `train`|
|4.  | Enabling or Disabling Gradients|
|-|-------Lightning Trainer Automatically handles all the above---------|-|

Below is the Lightning example code showing what all functions are needed in the lightning class.
1. `__init__` function
2. `forward` function --> This defines the prediction or inference actions
3. `training_step` function --> training loop and capture losses
4. `validation_step` function --> capture validation set losses
5. `test_step` function --> captures testset losses
6. `configure_optimizers` function --> here the learning_rate, optimizers, schedulers are defined

![PyTorch lightning-59](https://github.com/phaninandula/ERA-Session12/assets/30425824/f30307c2-bf52-4a81-9b3b-d3b619bb8d5c)

## 🏗 Model Architecture

The model should have an architecture as defined above in the problem statement. The objective is to use Residual blocks, OneCyclePolicy, Adam optimizer, cross-entropy loss, batch size of 512, and achieve 90% accuracy. 

The CIFAR10 dataset consists of 60,000 32x32 color training images and 10,000 test images, labeled into 10 classes. The 10 classes represent airplanes, cars, birds, cats, deer, dogs, frogs, horses, ships, and trucks. The dataset is divided into 50,000 training images and 10,000 validation images.


## 🎨 Data augmentation 
The model uses data augmentation techniques to improve robustness and prevent overfitting by increasing data diversity. This includes RandomCrop (32, 32), applied after a 4-pixel padding, to enhance positional robustness by randomly cropping images. FlipLR is used for introducing orientation robustness by mirroring images along the vertical axis. Lastly, CutOut (8, 8) randomly masks parts of the image, promoting the model's ability to learn from various regions, thereby improving its robustness to occlusions.

Sample images,  
![augmentation](./images/dataloader_preview.png)

## 📈 Results

The model was trained for 24 epochs and achieved an test accuracy of 86.11 % on the test set. 

## ❌ Misclassified Images

Few Samples of misclassified images can be seen in the folder `misclassified_images`  
