import torch
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self, dropout_value=0.01):
        super(Net, self).__init__()
        
        # Prep Layer Conv 3x3 s1, p1) >> BN >> RELU [64k]
        self.PREPLAYER = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=(3, 3), padding=1, stride=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        # CONV_BLOCK_1 -Conv 3x3 (s1, p1) >> MaxPool2D >> BN >> RELU [128k] 
        self.CONV_BLOCK_1 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(3, 3), padding=1, stride=1, bias=False),
            nn.MaxPool2d(2,2),
            nn.BatchNorm2d(128),
            nn.ReLU(),
        )
        # RES_BLOCK_1 (Conv-BN-ReLU-Conv-BN-ReLU))(X) [128k]  
        self.RES_BLOCK_1 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 3), padding=1, stride=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 3), padding=1, stride=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(),
        ) # output_size = 30
        
        # CONV_BLOCK_2 -Conv 3x3 (s1, p1) >> MaxPool2D >> BN >> RELU [256k] 
        self.CONV_BLOCK_2 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(3, 3), padding=1, stride=1, bias=False),
            nn.MaxPool2d(2,2),
            nn.BatchNorm2d(256),
            nn.ReLU(),
        )
        # CONV_BLOCK_3 -Conv 3x3 (s1, p1) >> MaxPool2D >> BN >> RELU [128k] 
        self.CONV_BLOCK_3 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=(3, 3), padding=1, stride=1, bias=False),
            nn.MaxPool2d(2,2),
            nn.BatchNorm2d(512),
            nn.ReLU(),
        )
        # RES_BLOCK_2 (Conv-BN-ReLU-Conv-BN-ReLU))(X) [128k]  
        self.RES_BLOCK_2 = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(3, 3), padding=1, stride=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(3, 3), padding=1, stride=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(),
        ) # output_size = 30

        self.MAX_POOL = nn.Sequential(
            nn.MaxPool2d(4,2)
        ) # output_size = 1

        self.FULLY_CONNECTED = nn.Sequential(
            nn.Linear(512, 10)
        ) # output_size = 1

    def forward(self, x):
        x = self.PREPLAYER(x)
        x = self.CONV_BLOCK_1(x)
        r1 = self.RES_BLOCK_1(x)
        x = x+r1
        x = self.CONV_BLOCK_2(x)
        x = self.CONV_BLOCK_3(x)
        r2 = self.RES_BLOCK_2(x)
        x = x+r2
        x = self.MAX_POOL(x)
        x = self.FULLY_CONNECTED(torch.squeeze(x))
        x = x.view(-1, 10)
        return F.log_softmax(x, dim=-1)
