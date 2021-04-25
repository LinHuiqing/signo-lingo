from collections import OrderedDict

import torch
import torch.nn as nn
from torch.autograd import Variable
from torchvision.models import resnet18, vgg11


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
            self.batchnorm1 = nn.BatchNorm2d(channel_out)
            self.batchnorm2 = nn.BatchNorm2d(channel_out)
        
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

        pool_size = 20
        self.avgpool = nn.AdaptiveAvgPool2d(pool_size)
        
        fc_in = channels[n_layers] * pool_size * pool_size
        self.fc = nn.Linear(fc_in, channel_out)
        
    def forward(self, x):
        out = self.conv1(x)
        out = self.layers(out)
        out = self.avgpool(out)
        out = torch.flatten(out, 1)
        out = self.fc(out)
        return out

class RNN(nn.Module):

    def __init__(self, 
                 embed_dim, 
                 hidden_dim, 
                 channel_out, 
                 n_layers, 
                 intermediate_act_fn="relu", 
                 bidirectional=False, 
                 device="cuda"):
        
        super(RNN, self).__init__()

        self.num_layers = n_layers
        self.hidden_dim = hidden_dim
        self.device = device
        self.bidirectional = bidirectional

        self.lstm = nn.LSTM(embed_dim, 
                            self.hidden_dim, 
                            num_layers=self.num_layers, 
                            bidirectional=self.bidirectional, 
                            batch_first=True)

        if self.bidirectional:
            fc1_in = self.hidden_dim * 2
        else:
            fc1_in = self.hidden_dim
        self.fc1 =  nn.Linear(fc1_in, channel_out) #fully connected 1
        # self.fc2 = nn.Linear(128, channel_out) #fully connected last layer

        if intermediate_act_fn == "relu":
            self.a_fn = nn.ReLU()
        elif intermediate_act_fn == "leaky_relu":
            self.a_fn = nn.LeakyReLU()
        elif intermediate_act_fn == "param_relu":
            self.a_fn = nn.PReLU()
        else:
            raise ValueError("please use a valid activation function argument ('relu'; 'leaky_relu'; 'param_relu')")

    def forward(self, x):
        
        if self.bidirectional:
            h_0_size = c_0_size = self.num_layers * 2
        else:
            h_0_size = c_0_size = self.num_layers
        h = torch.zeros(h_0_size, x.size(0), self.hidden_dim).to(self.device)
        c = torch.zeros(c_0_size, x.size(0), self.hidden_dim).to(self.device)

        # Propagate input through LSTM
        output, (h, c) = self.lstm(x, (h, c)) #lstm with input, hidden, and internal state
        # hn = hn.view(-1, self.hidden_size) #reshaping the data for Dense layer next
        
        last_out = output[:, -1, :]
        out = self.a_fn(last_out)
        out = self.fc1(out)
        
        return out

# class AttnDecoderRNN(nn.Module):
    
#     def __init__(self, 
#                  hidden_size, 
#                  output_size, 
#                  max_length=30, 
#                  dropout_p=0.1):
        
#         super(AttnDecoderRNN, self).__init__()
        
#         self.hidden_size = hidden_size
#         self.output_size = output_size
#         self.dropout_p = dropout_p
#         self.max_length = max_length

#         # self.embedding = nn.Embedding(self.output_size, self.hidden_size)
#         self.attn = nn.Linear(self.hidden_size * 2, self.max_length)
#         self.attn_combine = nn.Linear(self.hidden_size * 2, self.hidden_size)
#         self.dropout = nn.Dropout(self.dropout_p)
#         self.gru = nn.GRU(self.hidden_size, self.hidden_size)
#         self.out = nn.Linear(self.hidden_size, self.output_size)

#     def forward(self, input, hidden, encoder_outputs):
#         # embedded = self.embedding(input).view(1, 1, -1)
#         embedded = self.dropout(embedded)

#         attn_weights = F.softmax(
#             self.attn(torch.cat((embedded[0], hidden[0]), 1)), dim=1)
#         attn_applied = torch.bmm(attn_weights.unsqueeze(0),
#                                  encoder_outputs.unsqueeze(0))

#         output = torch.cat((embedded[0], attn_applied[0]), 1)
#         output = self.attn_combine(output).unsqueeze(0)

#         output = F.relu(output)
#         output, hidden = self.gru(output, hidden)

#         output = F.log_softmax(self.out(output[0]), dim=1)
#         return output, hidden, attn_weights

#     def initHidden(self):
#         return torch.zeros(1, 1, self.hidden_size, device=device)

class CNN_LSTM(nn.Module):

    def __init__(self, 
                 n_classes, 
                 latent_size, 
                 n_cnn_layers, 
                 n_rnn_layers, 
                 channel_in=3, 
                 cnn_act_fn="relu", 
                 rnn_act_fn="relu", 
                 dropout_rate=0.1,
                 cnn_bn=False, 
                 bidirectional=False):
        
        super(CNN_LSTM, self).__init__()

        self.CNN = CNN(latent_size, n_cnn_layers, intermediate_act_fn=cnn_act_fn, use_batchnorm=cnn_bn, channel_in=channel_in)
        self.RNN = RNN(latent_size, 3, n_classes, n_rnn_layers, intermediate_act_fn=rnn_act_fn, bidirectional=bidirectional)

        self.dropout = nn.Dropout(p=dropout_rate)

        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        batch_size, timesteps, C, H, W = x.size()

        cnn_in = x.view(batch_size * timesteps, C, H, W)
        latent_var = self.CNN(cnn_in)

        rnn_in = latent_var.view(batch_size, timesteps, -1)
        rnn_in = self.dropout(rnn_in)
        out = self.RNN(rnn_in)

        out = self.softmax(out)
        return out

class CNN_AttnRNN(nn.Module):

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
        
        super(CNN_AttnRNN, self).__init__()

        self.CNN = CNN(latent_size, n_cnn_layers, intermediate_act_fn=cnn_act_fn, use_batchnorm=cnn_bn, channel_in=channel_in)
        self.RNN = RNN(latent_size, 3, n_classes, n_rnn_layers, intermediate_act_fn=rnn_act_fn, bidirectional=bidirectional)

        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        batch_size, timesteps, C, H, W = x.size()

        cnn_in = x.view(batch_size * timesteps, C, H, W)
        latent_var = self.CNN(cnn_in)

        rnn_in = latent_var.view(batch_size, timesteps, -1)
        out = self.RNN(rnn_in)

        out = self.softmax(out)
        return out

class ResNet_LSTM(nn.Module):

    def __init__(self, 
                 n_classes, 
                 latent_size, 
                 n_rnn_layers, 
                 channel_in=3, 
                 rnn_act_fn="relu", 
                 bidirectional=False, 
                 resnet_opt="resnet18"):
        
        super(ResNet_LSTM, self).__init__()

        if resnet_opt == "resnet18":
            self.CNN = resnet18(pretrained=True)
        elif resnet_opt == "resnet101":
            self.CNN = resnet101(pretrained=True)
        
        self.CNN.fc = nn.Sequential(nn.Linear(self.CNN.fc.in_features, latent_size))

        # self.CNN = CNN(latent_size, n_cnn_layers, intermediate_act_fn=cnn_act_fn, use_batchnorm=cnn_bn, channel_in=channel_in)
        self.RNN = RNN(latent_size, 3, n_classes, n_rnn_layers, intermediate_act_fn=rnn_act_fn, bidirectional=bidirectional)

        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        batch_size, timesteps, C, H, W = x.size()

        cnn_in = x.view(batch_size * timesteps, C, H, W)
        latent_var = self.CNN(cnn_in)

        rnn_in = latent_var.view(batch_size, timesteps, -1)
        out = self.RNN(rnn_in)

        out = self.softmax(out)
        return out

class VGG_LSTM(nn.Module):

    def __init__(self, 
                 n_classes, 
                 latent_size, 
                 n_rnn_layers, 
                 rnn_act_fn="relu", 
                 bidirectional=False):
        
        super(VGG_LSTM, self).__init__()
        
        self.CNN = vgg11(pretrained=True)
        self.CNN.classifier[6] = nn.Linear(4096, latent_size)

        # self.CNN = CNN(latent_size, n_cnn_layers, intermediate_act_fn=cnn_act_fn, use_batchnorm=cnn_bn, channel_in=channel_in)
        self.RNN = RNN(latent_size, 3, n_classes, n_rnn_layers, intermediate_act_fn=rnn_act_fn, bidirectional=bidirectional)

        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        batch_size, timesteps, C, H, W = x.size()

        cnn_in = x.view(batch_size * timesteps, C, H, W)
        latent_var = self.CNN(cnn_in)

        rnn_in = latent_var.view(batch_size, timesteps, -1)
        out = self.RNN(rnn_in)

        out = self.softmax(out)
        return out
    
# # source: https://github.com/pranoyr/cnn-lstm/blob/master/models/cnnlstm.py
# import torch
# import torch.nn as nn
# import torchvision.models as models
# from torch.nn.utils.rnn import pack_padded_sequence
# import torch.nn.functional as F
# from torchvision.models import resnet18, resnet101


# class CNNLSTM(nn.Module):
#     def __init__(self, 
#                  num_classes, 
#                  latent_size, 
#                  lstm_hidden_size=256, 
#                  lstm_n_layers=3):
#         super(CNNLSTM, self).__init__()
#         self.resnet = resnet101(pretrained=True)
#         self.resnet.fc = nn.Sequential(nn.Linear(self.resnet.fc.in_features, latent_size))
#         self.lstm = nn.LSTM(input_size=latent_size, hidden_size=lstm_hidden_size, num_layers=lstm_n_layers)
#         self.fc1 = nn.Linear(lstm_hidden_size, 128)
#         self.fc2 = nn.Linear(128, num_classes)
       
#     def forward(self, x_3d):
#         hidden = None
#         for t in range(x_3d.size(1)):
#             with torch.no_grad():
#                 x = self.resnet(x_3d[:, t, :, :, :])  
#             out, hidden = self.lstm(x.unsqueeze(0), hidden)         

#         x = self.fc1(out[-1, :, :])
#         x = F.relu(x)
#         x = self.fc2(x)
#         return x

# class TimeDistributed(nn.Module):
#     def __init__(self, 
#                  module, 
#                  batch_first=False):
        
#         super(TimeDistributed, self).__init__()

#         self.module = module
#         self.batch_first = batch_first

#     def forward(self, x):

#         if len(x.size()) <= 2:
#             return self.module(x)

#         # Squash samples and timesteps into a single axis
#         x_reshape = x.contiguous().view(-1, x.size(-1))  # (samples * timesteps, input_size)

#         y = self.module(x_reshape)

#         # We have to reshape Y
#         if self.batch_first:
#             y = y.contiguous().view(x.size(0), -1, y.size(-1))  # (samples, timesteps, output_size)
#         else:
#             y = y.view(-1, x.size(1), y.size(-1))  # (timesteps, samples, output_size)

#         return y