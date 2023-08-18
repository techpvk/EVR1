"""
Implementation of YOLOv3 architecture
"""

import model
from model import *
import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_lightning import LightningModule
from torch.optim.lr_scheduler import OneCycleLR
from torchmetrics.functional import accuracy
import config
from loss import YoloLoss
from utils import get_loaders, load_checkpoint
import torch.optim as optim

class Custom_YOLOv3_for_PASCAL(LightningModule):
    
    def __init__(self,lr_value=0):
        super().__init__()
        self.model = YOLOv3(num_classes=config.NUM_CLASSES)
        #tore all the provided arguments under the self.hparams attribute. These hyperparameters will also be stored within the model checkpoint, which simplifies model re-instantiation after training.
        self.save_hyperparameters()
        self.loss_fn = YoloLoss()
        self.scaler = torch.cuda.amp.GradScaler()
        self.example_input_array = torch.randn((2, 3, 416, 416))
        self.scaled_anchors = (
            torch.tensor(config.ANCHORS)
            * torch.tensor(config.S).unsqueeze(1).unsqueeze(1).repeat(1, 3, 2)
        ).to(self.device)
        if lr_value == 0:
            self.learning_rate = config.LEARNING_RATE
        else:
            self.learning_rate = lr_value
    
    def forward(self,imgs):
        return self.model(imgs)
    
    def configure_optimizers(self):
        optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate, weight_decay=config.WEIGHT_DECAY)
        EPOCHS = config.NUM_EPOCHS * 2
        scheduler = OneCycleLR(
        optimizer,
        max_lr=1E-3,
        steps_per_epoch=len(self.train_dataloader()),
        epochs=EPOCHS,
        pct_start=5/EPOCHS,
        div_factor=100,
        three_phase=False,
        final_div_factor=100,
        anneal_strategy='linear')
       #scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=1e-3,total_steps=self.trainer.estimated_stepping_batches    )
        return {"optimizer": optimizer, "lr_scheduler": scheduler}
    
    def train_dataloader(self):
        train_loader, _, _ = get_loaders(train_csv_path=config.DATASET + "/train.csv", test_csv_path=config.DATASET + "/test.csv")
        return train_loader
        
    def on_train_start(self):
        if config.LOAD_MODEL:
            load_checkpoint(config.CHECKPOINT_FILE, self.model, self.optimizers(), config.LEARNING_RATE)
        self.scaled_anchors = (
            torch.tensor(config.ANCHORS)
            * torch.tensor(config.S).unsqueeze(1).unsqueeze(1).repeat(1, 3, 2)
        ).to(self.device)
        
    def training_step(self, batch, batch_idx):
        losses = []
        x, y = batch
        #x = x.to(config.DEVICE)
        y0, y1, y2 = (y[0].to(config.DEVICE),y[1].to(config.DEVICE),y[2].to(config.DEVICE),)
        losses = []
        with torch.cuda.amp.autocast():
            out = self.model(x)
            loss = (
                self.loss_fn(out[0], y0, self.scaled_anchors[0])
                + self.loss_fn(out[1], y1, self.scaled_anchors[1])
                + self.loss_fn(out[2], y2, self.scaled_anchors[2])
            )

        self.log('train_loss', loss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
        return loss
    def on_training_epoch_end(self, outputs):
        self.log('on_training_epoch_end',prog_bar=True, logger=True, on_step=True, on_epoch=True)
        
    def val_dataloader(self):
        _, _, train_eval_loader = get_loaders(train_csv_path=config.DATASET + "/train.csv", test_csv_path=config.DATASET + "/test.csv")
        return train_eval_loader
        
    def validation_step(self, batch, batch_idx):
        losses = []
        x, y = batch
        #x = x.to(config.DEVICE)
        y0, y1, y2 = (y[0].to(config.DEVICE),y[1].to(config.DEVICE),y[2].to(config.DEVICE),)
        losses = []
        with torch.cuda.amp.autocast():
            out = self.model(x)
            loss = (
                self.loss_fn(out[0], y0, self.scaled_anchors[0])
                + self.loss_fn(out[1], y1, self.scaled_anchors[1])
                + self.loss_fn(out[2], y2, self.scaled_anchors[2])
            )

        self.log('validation_loss', loss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
    
    def test_dataloader(self):
        _, test_loader, _ = get_loaders(train_csv_path=config.DATASET + "/train.csv", test_csv_path=config.DATASET + "/test.csv")
        return test_loader
        
    def test_step(self, batch, batch_idx):
        losses = []
        x, y = batch
        #x = x.to(config.DEVICE)
        y0, y1, y2 = (y[0].to(config.DEVICE),y[1].to(config.DEVICE),y[2].to(config.DEVICE),)
        losses = []
        with torch.cuda.amp.autocast():
            out = self.model(x)
            loss = (
                self.loss_fn(out[0], y0, self.scaled_anchors[0])
                + self.loss_fn(out[1], y1, self.scaled_anchors[1])
                + self.loss_fn(out[2], y2, self.scaled_anchors[2])
            )

        self.log('test_loss', loss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
        
    
if __name__ == "__main__":
    num_classes = 20
    IMAGE_SIZE = 416
    model = Custom_YOLOv3_for_PASCAL(num_classes=num_classes)
    x = torch.randn((2, 3, IMAGE_SIZE, IMAGE_SIZE))
    out = model(x)
    assert model(x)[0].shape == (2, 3, IMAGE_SIZE//32, IMAGE_SIZE//32, num_classes + 5)
    assert model(x)[1].shape == (2, 3, IMAGE_SIZE//16, IMAGE_SIZE//16, num_classes + 5)
    assert model(x)[2].shape == (2, 3, IMAGE_SIZE//8, IMAGE_SIZE//8, num_classes + 5)
    print("Success!")