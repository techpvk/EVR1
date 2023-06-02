import torch
from torchvision import datasets, transforms


def getMNIST_TestData():
  test_transforms = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.1307,), (0.3081,))])
  test_data = datasets.MNIST('../data', train=False, download=True, transform=test_transforms)
  return test_data


def getMNIST_TrainData():
  train_transforms = transforms.Compose([transforms.RandomApply([transforms.CenterCrop(22), ], p=0.1),transforms.Resize((28, 28)),transforms.RandomRotation((-15., 15.), fill=0),transforms.ToTensor(),transforms.Normalize((0.1307,), (0.3081,)),])
  train_data = datasets.MNIST('../data', train=True, download=True, transform=train_transforms)
  return train_data

def getMNIST_DataLoader(dataset):
  kwargs = {'batch_size': 512, 'shuffle': True, 'num_workers': 2, 'pin_memory': True}
  train_loader = torch.utils.data.DataLoader(dataset, **kwargs)
  return train_loader

    