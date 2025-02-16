import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable

from inputs import INPUTS

def Net():

    class Flatten(nn.Module):

        def forward(self, x):
            return x.view(x.size()[0], -1)
    
    class extract_tensor(nn.Module):
        def forward(self,x):
            # Output shape (batch, features, hidden)
            out, hn = x
            # Reshape shape (batch, hidden)
            return out


    model = torch.nn.Sequential(
        torch.nn.Conv2d(3, 32, 4, stride=2, bias=True),
        torch.nn.ReLU(),
        torch.nn.Conv2d(32, 64, 4, stride=2, bias=True),
        torch.nn.ReLU(),   
        torch.nn.Conv2d(64, 64, 3, stride=1, bias=True),
        torch.nn.ReLU(),
        torch.nn.BatchNorm2d(64),
        
        torch.nn.Conv2d(64, 64, 3, stride=1, bias=True),
        torch.nn.ReLU(),
        torch.nn.Dropout(p=0.4),

        Flatten(),

        torch.nn.BatchNorm1d(64 * 26 * 26),

        nn.Linear(in_features=64 * 26 * 26, out_features=512, bias=True),
        nn.ReLU(),
        nn.Linear(in_features=512, out_features=len(INPUTS), bias=True)
    )

    model.double()

    return model
    
class RNNModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, layer_dim, output_dim):
        super(RNNModel, self).__init__()
        
        # Number of hidden dimensions
        self.hidden_dim = hidden_dim
        
        # Number of hidden layers
        self.layer_dim = layer_dim
        
        # RNN
        self.rnn = nn.RNN(input_dim, hidden_dim, layer_dim, batch_first=True, nonlinearity='relu')
        
        # Readout layer
        self.fc = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, x):
        
        # Initialize hidden state with zeros
        h0 = Variable(torch.zeros(self.layer_dim, x.size(0), self.hidden_dim))
            
        # One time step
        out, hn = self.rnn(x, h0)
        out = self.fc(out[:, -1, :]) 
        return out