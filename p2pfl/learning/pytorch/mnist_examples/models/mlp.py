import torch
from torch import nn
from torch.nn import functional as F
import pytorch_lightning as pl
from torchmetrics import Accuracy


###############################
#    Multilayer Perceptron    #
###############################

class MLP(pl.LightningModule):
    """
    Multilayer Perceptron (MLP) to solve MNIST with PyTorch Lightning.
    """

    def __init__(self, lr_rate=0.001): # low lr to avoid overfitting
        
        # Set seed for reproducibility iniciialization
        seed = 666
        torch.manual_seed(seed)

        super().__init__()
        self.lr_rate = lr_rate
        self.accuracy = Accuracy()

        # 10 clases
        self.l1 = torch.nn.Linear(28 * 28, 128)
        self.l2 = torch.nn.Linear(128, 256)
        self.l3 = torch.nn.Linear(256, 10)

    def forward(self, x):
        """
        """
        batch_size, channels, width, height = x.size()
        
        # (b, 1, 28, 28) -> (b, 1*28*28)
        x = x.view(batch_size, -1)
        x = self.l1(x)
        x = torch.relu(x)
        x = self.l2(x)
        x = torch.relu(x)
        x = self.l3(x)
        x = torch.log_softmax(x, dim=1)
        return x

    def configure_optimizers(self):
        """
        """
        return torch.optim.Adam(self.parameters(), lr=self.lr_rate)

    def training_step(self, batch, batch_id):
        """
        """
        x, y = batch
        loss = F.cross_entropy(self(x), y)
        self.log("train_loss", loss, prog_bar=True)
        return loss 

    def validation_step(self, batch, batch_idx):
        """
        """
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(self(x), y)
        out = torch.argmax(logits, dim=1)
        acc = self.accuracy(out, y) 
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_acc", acc, prog_bar=True)    
        return loss

    def test_step(self, batch, batch_idx):
        """
        """
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(self(x), y)
        out = torch.argmax(logits, dim=1)
        acc = self.accuracy(out, y) 
        self.log("test_loss", loss, prog_bar=True)
        self.log("test_acc", acc, prog_bar=True)
        return loss
        