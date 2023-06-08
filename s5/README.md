# Session 5 : Readme
This project contains 3 files and files are orgnaised for better readabilty 

| FileName      | Usage         |
| ------------- |:-------------:| 
| model.py      | Used for model object creation |
| utils.py      | All the utils for the dataset and dataloader creation      |
| s5.ipynb | main starting file for the model execution    |

## Model Architecture
Below image explain about network architecture with params used in the s5.ipnb
![Alt text](./resource/model_architecture.jpg)  

## model.py
model.py is model file used in the s5.ipynb for creating a model,lets look at the architecture.
![Alt text](./resource/model_architecture.jpg)  

## utils.py
utils.py as lib for project which helps to move the reusable code to one place as mentioned below.
[^1]: getMNIST_TestData helps to get the Test data.
[^2]: getMNIST_TrainData helps to get the Trian data.
[^2]: getMNIST_DataLoader helps to get the dataset as iteratable data by batch
