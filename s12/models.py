import torch.nn.functional as F
import torch.nn as nn
from pytorch_lightning import LightningModule
from torch.optim.lr_scheduler import OneCycleLR
from torchmetrics.functional import accuracy
import torch

dropout_value = 0.1

class X(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(X, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels,out_channels,kernel_size=3,stride=1, padding=1,bias = False),
            nn.MaxPool2d(kernel_size=2,stride=2),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )

    def forward(self, x):
        return self.conv1(x)

class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResBlock, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels,out_channels,kernel_size=3,stride=1, padding=1,bias = False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )

    def forward(self, x):
        out = self.conv(x)
        out = self.conv(out)
        out = out + x
        return out

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        # Prep Layer
        self.preplayer = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3,stride=1, padding=1,bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU()
        ) ## 32x32

        # Layer 1
        self.X1 = X(in_channels=64,out_channels=128) # 16x16
        self.R1 = ResBlock(in_channels=128,out_channels=128) # 32x32

        # Layer 2
        self.X2 = X(in_channels=128,out_channels=256)

        # Layer 3
        self.X3 = X(in_channels=256,out_channels=512)
        self.R3 = ResBlock(in_channels=512,out_channels=512)

        # Max Pool
        self.maxpool = nn.MaxPool2d(kernel_size=4, stride=1)

        # FC
        self.fc = nn.Linear(512,10)

    def forward(self, x):
        batch_size = x.shape[0]

        out = self.preplayer(x)

        # Layer 1
        X = self.X1(out) ## 16x16
        R1 = self.R1(X)  


        out = X + R1

        # Layer 2
        out = self.X2(out)

        # Layer 3
        X = self.X3(out)
        R2 = self.R3(X)  

        out = X + R2

        out = self.maxpool(out)

        # FC
        out = out.view(out.size(0),-1)
        out = self.fc(out)

        return F.softmax(out, dim=1)

class Custom_ResNet_for_CIFAR10(LightningModule):
  def __init__(self,lr = 1,batch_size=512):
    """
    Inputs:
      model_class -> Name of the class that contains all the details of the architecture
      optimizer_name --> Name of the optimizer to be used
      optimizer_hparams --> Hyperparameters of the optimizer as dictionary. This includes Learning rate, weight decay etc
    """
    super().__init__()
    self.model = Net()
    self.save_hyperparameters()
    self.BATCH_SIZE=batch_size

    self.loss_module = nn.CrossEntropyLoss()
    self.example_input_array = torch.zeros((1,3,32,32), dtype=torch.float32)
    #self.optimizer_hparams = {"lr": 1e-3, "weight_decay": 1e-4}

  def forward(self,imgs):
    # Forward function that is run when visualizing the graph
    return self.model(imgs)

  # def configure_optimizers(self):
  #   #self.hparams.optimizer_hparams = {"lr": 1e-3, "weight_decay": 1e-4}
  #   #optimizer = optim.AdamW(self.parameters(), lr=1e-3)
  #   # introduce LR Scheduler
  #   #scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=5.4E-3, steps_per_epoch=len(train_loader),
  #   #                                            epochs=24,pct_start=500/2400,
  #   #                                            anneal_strategy='linear')#0.01
  #   return optim.Adam(self.parameters(), lr=1e-3)

  def configure_optimizers(self):
    optimizer = torch.optim.Adam(
        self.parameters(),
        lr=self.hparams.lr,
    )
    steps_per_epoch = 45000 // self.BATCH_SIZE
    scheduler_dict = {
        "scheduler": OneCycleLR(
            optimizer,
            1e-03,
            epochs=self.trainer.max_epochs,
            steps_per_epoch=steps_per_epoch,
        ),
        "interval": "step",
    }
    return {"optimizer": optimizer, "lr_scheduler": scheduler_dict}

  def training_step(self, batch, batch_idx):
    # Batch is the output of the training data loader
    imgs, labels = batch
    preds = self.model(imgs)
    loss = self.loss_module(preds,labels)
    acc = (preds.argmax(dim=-1) == labels).float().mean()

    #logs the accuracy per epoch to tensorboard (weighted average over batches)
    self.log("train_acc", acc, on_step=False, on_epoch=True, prog_bar=True)
    self.log("train_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
    return loss

  def validation_step(self, batch, batch_idx):
    imgs, labels = batch
    preds = self.model(imgs)
    loss = self.loss_module(preds,labels)
    preds = preds.argmax(dim=-1)
    acc = (labels == preds).float().mean()
    # By default logs it per epoch (weighted average over batches)
    self.log("val_acc", acc, on_step=False, on_epoch=True, prog_bar=True)
    self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)

  def test_step(self, batch, batch_idx):

    imgs, labels = batch
    preds = self.model(imgs)
    loss = self.loss_module(preds,labels)
    preds = preds.argmax(dim=-1)
    acc = (labels == preds).float().mean()
    self.log("test_acc", acc, on_step=False, on_epoch=True, prog_bar=True)
    self.log("test_loss", loss, on_step=False, on_epoch=True, prog_bar=True)

  def on_training_epoch_end(self, outputs):
    avg_loss = torch.stack([x['loss'] for x in outputs]).mean()
    avg_acc = torch.stack([x['train_acc'] for x in outputs]).mean()
    self.log('avg_train_loss', avg_loss)
    self.log('avg_train_acc', avg_acc)
