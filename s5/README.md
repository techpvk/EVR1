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
> getMNIST_TestData helps to get the Test data.


> getMNIST_TrainData helps to get the Trian data.


> getMNIST_DataLoader helps to get the dataset as iteratable data by batch


## s5.ipynb
s5.ipynb is the starting file uses utils.py and model.py let check step by step

### CODE BLOCK: 1
Google cloab driver mount and basic lib installtions

![Alt text](./resource/model_architecture.jpg)  

### CODE BLOCK: 2
check the code is available or not

> cuda = torch.cuda.is_available()

> print("CUDA Available?", cuda)

### CODE BLOCK: 5
 Get Dataset as itertable data from utils.py

![Alt text](./resource/model_architecture.jpg)  

### CODE BLOCK: 6
 Get Dataloder as itertable data from utils.py

![Alt text](./resource/model_architecture.jpg)  

### CODE BLOCK: 6
Plot the sample data from dataloader.

![Alt text](./resource/model_architecture.jpg)  


### CODE BLOCK: 7
Create model object using model.py and list summary 

![Alt text](./resource/model_architecture.jpg)  


### CODE BLOCK: 8,9
 Run model with required batch size and desired epoc size.


### CODE BLOCK: 11
Plot the loss for train and test and accuracy for train and test.

![Alt text](./resource/model_architecture.jpg)  

