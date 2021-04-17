from typing import OrderedDict

import torch
import torch.nn as nn
from torch.autograd import Variable


class ConvBlock(nn.Module):

    def __init__(self, 
                 channel_in, 
                 channel_out, 
                 activation_fn, 
                 use_batchnorm, 
                 kernel_size = 3):
        
        super(ConvBlock, self).__init__()

        self.conv1 = nn.Conv2d(channel_in, channel_out, kernel_size)
        self.conv2 = nn.Conv2d(channel_out, channel_out, kernel_size)

        self.use_batchnorm = use_batchnorm
        if self.use_batchnorm:
            self.batchnorm1 = nn.BatchNorm2d()
            self.batchnorm2 = nn.BatchNorm2d()
        
        if activation_fn == "relu":
            self.a_fn = nn.ReLU()
        elif activation_fn == "leaky_relu":
            self.a_fn = nn.LeakyReLU()
        elif activation_fn == "param_relu":
            self.a_fn = nn.PReLU()
        else:
            raise ValueError("please use a valid activation function argument ('relu'; 'leaky_relu'; 'param_relu')")
    
    def forward(self, x):
        out = self.conv1(x)
        if self.use_batchnorm:
            out = self.batchnorm1(out)
        out = self.a_fn(out)

        out = self.conv2(out)
        if self.use_batchnorm:
            out = self.batchnorm2(out)
        out = self.a_fn(out)
        return out

class CNN(nn.Module):
    
    def __init__(self, 
                 channel_out, 
                 n_layers, 
                 intermediate_act_fn="relu", 
                 use_batchnorm=True, 
                 channel_in=3):
        
        super(CNN, self).__init__()

        channels = [64, 64, 128, 256, 512]

        if n_layers < 1 or n_layers >= len(channels):
            raise ValueError(f"please use a valid int for the n_layers param (1-{len(channels)-1})")

        self.conv1 = nn.Conv2d(channel_in, channels[0], 3, stride=2)
        self.maxpool = nn.MaxPool2d(3, 2)

        layers =  OrderedDict()
        for i in range(n_layers):
            layers[str(i)] = ConvBlock(channels[i], channels[i+1], intermediate_act_fn, use_batchnorm=use_batchnorm)
        
        self.layers = nn.Sequential(layers)
        self.avgpool = nn.AdaptiveAvgPool2d(10)
        self.fc = nn.Linear(channels[n_layers+1], channel_out)
        
    def forward(self, x):
        out = self.conv1(x)
        out = self.layers(out)
        out = self.avgpool(out)
        out = torch.flatten(out, 1)
        out = self.fc(out)
        return out

class RNN(nn.Module):

    def __init__(self, 
                 input_size, 
                 hidden_size, 
                 channel_out, 
                 n_layers, 
                 intermediate_act_fn="relu", 
                 bidirectional=False):
        
        super(RNN, self).__init__()

        self.lstm = nn.LSTM(input_size, 
                            hidden_size, 
                            num_layers=n_layers, 
                            bidirectional=bidirectional)

        self.fc1 =  nn.Linear(hidden_size, 128) #fully connected 1
        self.fc2 = nn.Linear(128, channel_out) #fully connected last layer

        

        if intermediate_act_fn == "relu":
            self.a_fn = nn.ReLU()
        elif intermediate_act_fn == "leaky_relu":
            self.a_fn = nn.LeakyReLU()
        elif intermediate_act_fn == "param_relu":
            self.a_fn = nn.PReLU()
        else:
            raise ValueError("please use a valid activation function argument ('relu'; 'leaky_relu'; 'param_relu')")

    def forward(self, x):
        # h_0 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size)) #hidden state
        # c_0 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size)) #internal state

        # Propagate input through LSTM
        output, (hn, cn) = self.lstm(x) #lstm with input, hidden, and internal state
        # hn = hn.view(-1, self.hidden_size) #reshaping the data for Dense layer next

        out = self.a_fn(hn)
        out = self.fc1(out)
        out = self.a_fn(out)
        out = self.fc2(output)
        return out

class CNN_RNN(nn.Module):

    def __init__(self, 
                 n_classes, 
                 latent_size, 
                 n_cnn_layers, 
                 n_rnn_layers, 
                 channel_in=3, 
                 cnn_act_fn="relu", 
                 rnn_act_fn="relu", 
                 cnn_bn=False, 
                 bidirectional=False):
        
        super(CNN_RNN, self).__init__()

        self.CNN = CNN(latent_size, n_cnn_layers, intermediate_act_fn=cnn_act_fn, use_batchnorm=cnn_bn, channel_in=channel_in)
        self.RNN = RNN(latent_size, 64, n_classes, n_rnn_layers, intermediate_act_fn=rnn_act_fn, bidirectional=bidirectional)

        self.log_softmax = nn.LogSoftmax(dim=1)

    def forward(self, x):
        latent_var = self.CNN(x)
        out = self.RNN(latent_var)
        out = self.log_softmax(out)
        return out
    