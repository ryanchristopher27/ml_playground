# Imports
import torch.nn as nn
import torch.nn.init as init

import torch
import lightning as L
from torch.optim import Adam, SGD
from torch.optim.lr_scheduler import CosineAnnealingLR


class Network(L.LightningModule):
    """
    Purpose: Custom Network
    """
    
    def __init__(self, config):
        """
        Purpose:
        - Define network architecture
        
        Arguments:
        - config (dict[any]): user defined parameters
        """
        
        super().__init__()

        self.alpha = config["hyper_parameters"]["learning_rate"]
        self.num_epochs = config["hyper_parameters"]["num_epochs"]
        
        self.model_type = config["model"]["type"]
        self.hidden_size = config["model"]["hidden_size"]
        self.num_layers = config["model"]["num_layers"]
        
        self.input_size = config["data"]["num_features"]

        self.objective_function = config["hyper_parameters"]["objective"]

        self.task = config["experiment"]

        self.optimizer = config["hyper_parameters"]["optimizer"]

        # Create: LSTM Architecture
        self.lstm_arch = torch.nn.LSTM(batch_first=True,
                                  num_layers=self.num_layers,
                                  input_size=self.input_size, 
                                  hidden_size=self.hidden_size)
        
        # Create: RNN Architecture
        self.rnn_arch = torch.nn.RNN(batch_first=True,
                                 num_layers=self.num_layers,
                                 input_size=self.input_size,
                                 hidden_size=self.hidden_size)

        self.linear = torch.nn.Linear(self.hidden_size, 1)

        self.test_predictions = []
     
    def objective(self, labels, preds):
        """
        Purpose:
        - Define network objective function
        
        Arguments:
        - labels (torch.tensor[float]): ground truths
        - preds (torch.tensor[float]): model predictions
        
        Returns:
        - (torch.tensor[float]): error between ground truths and predictions
        """
        if self.objective_function == "mse_loss":
            obj = torch.nn.functional.mse_loss(labels, preds)
        elif self.objective_function == "mae_loss":
            obj = torch.nn.functional.l1_loss(labels, preds)
        elif self.objective_function == "cross_entropy":
            obj = torch.nn.functional.cross_entropy(labels, preds)

        return obj

    def configure_optimizers(self):
        """
        Purpose: 
        - Define network optimizers and schedulars

        Returns:
        - (dict[str, any]): network optimizers and schedulars
        """

        # Create: Optimzation Routine
        if self.optimizer == "Adam":
            optimizer = Adam(self.parameters(), lr=self.alpha)
        elif self.optimizer == "SGD":
            optimizer = SGD(self.parameters(), lr=self.alpha)
        

        # Create: Learning Rate Schedular

        lr_scheduler = CosineAnnealingLR(optimizer, T_max=self.num_epochs)
        lr_scheduler_config = {"scheduler": lr_scheduler,
                               "interval": "epoch", "frequency": 1}

        return {"optimizer": optimizer, "lr_scheduler": lr_scheduler_config}
    
    def forward(self, x, target_seq=None):
        """
        Purpose: 
        - Define network forward pass

        Arguments:
        - x (torch.tensor[float]): network input observation
        - target_seq (int): number of sequence predictions

        Returns:
        - (torch.tensor[float]): network predictions
        """

        batch_size = x.size()[0]

        # LSTM Model
        if self.model_type == "LSTM":

            # Create: Hidden & Cell States
            hidden = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(x.device)
            cell = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(x.device)
            
            # Task: Many To One
            if self.task == 0:
                features, (hidden, cell) = self.lstm_arch(x, (hidden, cell))
                features = features[:, -1].view(batch_size, -1)
                preds = self.linear(features)

            # Task: Many To Many
            else:
                
                preds = torch.zeros(batch_size, target_seq).to(x.device)

                for i in range(target_seq):
                    features, (hidden, cell) = self.lstm_arch(x, (hidden, cell))
                    features = features[:, -1].view(batch_size, -1)
                    output = self.linear(features).view(-1)
                    preds[:, i] = output

        # RNN Model
        elif self.model_type == "RNN":
            # Create: Hidden State
            hidden = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(x.device)
            
            # Task: Many To One
            if self.task == 0:
                features, hidden = self.rnn_arch(x, hidden)
                features = features[:, -1].view(batch_size, -1)
                preds = self.linear(features)
            
            # Task: Many To Many
            else:
                preds = torch.zeros(batch_size, target_seq).to(x.device)
                
                for i in range(target_seq):
                    features, hidden = self.rnn_arch(x, hidden)
                    features = features[:, -1].view(batch_size, -1)
                    output = self.linear(features).view(-1)
                    preds[:, i] = output

        return preds

    def shared_step(self, batch, batch_idx, tag):
        """
        Purpose: 
        - Define network batch processing procedure

        Arguments: 
        - batch (tuple[any]): network observations (i.e., samples, labels)
        - batch_idx (int): batch iteration counter
        - tag (str): tag for labeling analytics

        Returns:
        - (torch.tensor[any]): error between ground-truth and predictions
        """
        
        samples, labels = batch
        batch_size = samples.size()[0]

        # Gather: Predictions
        
        if self.task == 0:
            preds = self(samples)
        else:
            target_seq = labels.size()[1]
            preds = self(samples, target_seq)

        # Calculate: Objective
 
        loss = self.objective(preds, labels)

        self.log(tag, loss, batch_size=batch_size,
                 on_step=True, on_epoch=True, sync_dist=True)

        return loss

    def training_step(self, batch, batch_idx):
        """
        Purpose: 
        - Define network training iteration

        Arguments: 
        - batch (tuple[any]): network observations (i.e., samples, labels)
        - batch_idx (int): batch iteration counter

        Returns:
        - (Function): batch processing procedure that returns training error
        """

        return self.shared_step(batch, batch_idx, "train_error")

    def validation_step(self, batch, batch_idx):
        """
        Purpose: 
        - Define network validation iteration

        Arguments: 
        - batch (tuple[any]): network observations (i.e., samples, labels)
        - batch_idx (int): batch iteration counter
        """

        self.shared_step(batch, batch_idx, "valid_error")

    def test_step(self, batch, batch_idx):
        """
        Purpose: 
        - Define network test iteration

        Arguments: 
        - batch (tuple[any]): network observations (i.e., samples, labels)
        - batch_idx (int): batch iteration counter
        """

        self.shared_step(batch, batch_idx, "test_error")

    def predict_step(self, batch, batch_idx):
        """
        Purpose: 
        - Define network predict iteration

        Arguments: 
        - batch (tuple[any]): network observations (i.e., samples, labels)
        - batch_idx (int): batch iteration counter
        """

        samples, labels = batch
        preds = self(samples)

        self.test_predictions.append(preds.cpu().numpy())

        # return preds.cpu().numpy()

        # self.shared_step(batch, batch_idx, "predict_error")