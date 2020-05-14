import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

__weights_dict = dict()

def load_weights(weight_file):
    if weight_file == None:
        return

    try:
        weights_dict = np.load(weight_file, allow_pickle=True).item()
        print("weight_dict: ", weights_dict)
    except:
        weights_dict = np.load(weight_file, allow_pickle=True, encoding='bytes').item()
        print("weight_dict: ", weights_dict)


    return weights_dict

class KitModel(nn.Module):

    
    def __init__(self, weight_file):
        super(KitModel, self).__init__()
        global __weights_dict
        __weights_dict = load_weights(weight_file)

        self.data_bn = self.__batch_normalization(2, 'data/bn', num_features=3, eps=9.999999747378752e-06, momentum=0.0)
        self.conv1_1 = self.__conv(2, name='conv1_1', in_channels=3, out_channels=64, kernel_size=(3, 3), stride=(1, 1), groups=1, bias=True)
        self.conv1_1_bn = self.__batch_normalization(2, 'conv1_1/bn', num_features=64, eps=9.999999747378752e-06, momentum=0.0)
        self.conv1_2 = self.__conv(2, name='conv1_2', in_channels=64, out_channels=64, kernel_size=(3, 3), stride=(1, 1), groups=1, bias=True)
        self.conv1_2_bn = self.__batch_normalization(2, 'conv1_2/bn', num_features=64, eps=9.999999747378752e-06, momentum=0.0)
        self.conv2_1 = self.__conv(2, name='conv2_1', in_channels=64, out_channels=128, kernel_size=(3, 3), stride=(1, 1), groups=1, bias=True)
        self.conv2_1_bn = self.__batch_normalization(2, 'conv2_1/bn', num_features=128, eps=9.999999747378752e-06, momentum=0.0)
        self.conv2_2 = self.__conv(2, name='conv2_2', in_channels=128, out_channels=128, kernel_size=(3, 3), stride=(1, 1), groups=1, bias=True)
        self.conv2_2_bn = self.__batch_normalization(2, 'conv2_2/bn', num_features=128, eps=9.999999747378752e-06, momentum=0.0)
        self.conv3_1 = self.__conv(2, name='conv3_1', in_channels=128, out_channels=256, kernel_size=(3, 3), stride=(1, 1), groups=1, bias=True)
        self.conv3_1_bn = self.__batch_normalization(2, 'conv3_1/bn', num_features=256, eps=9.999999747378752e-06, momentum=0.0)
        self.conv3_2 = self.__conv(2, name='conv3_2', in_channels=256, out_channels=256, kernel_size=(3, 3), stride=(1, 1), groups=1, bias=True)
        self.conv3_2_bn = self.__batch_normalization(2, 'conv3_2/bn', num_features=256, eps=9.999999747378752e-06, momentum=0.0)
        self.conv3_3 = self.__conv(2, name='conv3_3', in_channels=256, out_channels=256, kernel_size=(3, 3), stride=(1, 1), groups=1, bias=True)
        self.conv3_3_bn = self.__batch_normalization(2, 'conv3_3/bn', num_features=256, eps=9.999999747378752e-06, momentum=0.0)
        self.conv3_4 = self.__conv(2, name='conv3_4', in_channels=256, out_channels=256, kernel_size=(3, 3), stride=(1, 1), groups=1, bias=True)
        self.conv3_4_bn = self.__batch_normalization(2, 'conv3_4/bn', num_features=256, eps=9.999999747378752e-06, momentum=0.0)
        self.conv4_1 = self.__conv(2, name='conv4_1', in_channels=256, out_channels=512, kernel_size=(3, 3), stride=(1, 1), groups=1, bias=True)
        self.conv4_1_bn = self.__batch_normalization(2, 'conv4_1/bn', num_features=512, eps=9.999999747378752e-06, momentum=0.0)
        self.conv4_2 = self.__conv(2, name='conv4_2', in_channels=512, out_channels=512, kernel_size=(3, 3), stride=(1, 1), groups=1, bias=True)
        self.conv4_2_bn = self.__batch_normalization(2, 'conv4_2/bn', num_features=512, eps=9.999999747378752e-06, momentum=0.0)
        self.conv4_3 = self.__conv(2, name='conv4_3', in_channels=512, out_channels=512, kernel_size=(3, 3), stride=(1, 1), groups=1, bias=True)
        self.conv4_3_bn = self.__batch_normalization(2, 'conv4_3/bn', num_features=512, eps=9.999999747378752e-06, momentum=0.0)
        self.conv4_4 = self.__conv(2, name='conv4_4', in_channels=512, out_channels=512, kernel_size=(3, 3), stride=(1, 1), groups=1, bias=True)
        self.conv4_4_bn = self.__batch_normalization(2, 'conv4_4/bn', num_features=512, eps=9.999999747378752e-06, momentum=0.0)
        self.conv5_1 = self.__conv(2, name='conv5_1', in_channels=512, out_channels=512, kernel_size=(3, 3), stride=(1, 1), groups=1, bias=True)
        self.conv5_1_bn = self.__batch_normalization(2, 'conv5_1/bn', num_features=512, eps=9.999999747378752e-06, momentum=0.0)
        self.conv5_2 = self.__conv(2, name='conv5_2', in_channels=512, out_channels=512, kernel_size=(3, 3), stride=(1, 1), groups=1, bias=True)
        self.conv5_2_bn = self.__batch_normalization(2, 'conv5_2/bn', num_features=512, eps=9.999999747378752e-06, momentum=0.0)
        self.conv5_3 = self.__conv(2, name='conv5_3', in_channels=512, out_channels=512, kernel_size=(3, 3), stride=(1, 1), groups=1, bias=True)
        self.conv5_3_bn = self.__batch_normalization(2, 'conv5_3/bn', num_features=512, eps=9.999999747378752e-06, momentum=0.0)
        self.conv5_4 = self.__conv(2, name='conv5_4', in_channels=512, out_channels=512, kernel_size=(3, 3), stride=(1, 1), groups=1, bias=True)
        self.conv5_4_bn = self.__batch_normalization(2, 'conv5_4/bn', num_features=512, eps=9.999999747378752e-06, momentum=0.0)

    def forward(self, x):
        data_bn         = self.data_bn(x)
        conv1_1_pad     = F.pad(data_bn, (1, 1, 1, 1))
        conv1_1         = self.conv1_1(conv1_1_pad)
        conv1_1_bn      = self.conv1_1_bn(conv1_1)
        relu1_1         = F.relu(conv1_1_bn, inplace=True)
        conv1_2_pad     = F.pad(relu1_1, (1, 1, 1, 1))
        conv1_2         = self.conv1_2(conv1_2_pad)
        conv1_2_bn      = self.conv1_2_bn(conv1_2)
        relu1_2         = F.relu(conv1_2_bn)
        pool1_pad       = F.pad(relu1_2, (0, 1, 0, 1), value=float('-inf'))
        pool1, pool1_idx = F.max_pool2d(pool1_pad, kernel_size=(2, 2), stride=(2, 2), padding=0, ceil_mode=False, return_indices=True)
        conv2_1_pad     = F.pad(pool1, (1, 1, 1, 1))
        conv2_1         = self.conv2_1(conv2_1_pad)
        conv2_1_bn      = self.conv2_1_bn(conv2_1)
        relu2_1         = F.relu(conv2_1_bn)
        conv2_2_pad     = F.pad(relu2_1, (1, 1, 1, 1))
        conv2_2         = self.conv2_2(conv2_2_pad)
        conv2_2_bn      = self.conv2_2_bn(conv2_2)
        relu2_2         = F.relu(conv2_2_bn)
        pool2_pad       = F.pad(relu2_2, (0, 1, 0, 1), value=float('-inf'))
        pool2, pool2_idx = F.max_pool2d(pool2_pad, kernel_size=(2, 2), stride=(2, 2), padding=0, ceil_mode=False, return_indices=True)
        conv3_1_pad     = F.pad(pool2, (1, 1, 1, 1))
        conv3_1         = self.conv3_1(conv3_1_pad)
        conv3_1_bn      = self.conv3_1_bn(conv3_1)
        relu3_1         = F.relu(conv3_1_bn)
        conv3_2_pad     = F.pad(relu3_1, (1, 1, 1, 1))
        conv3_2         = self.conv3_2(conv3_2_pad)
        conv3_2_bn      = self.conv3_2_bn(conv3_2)
        relu3_2         = F.relu(conv3_2_bn)
        conv3_3_pad     = F.pad(relu3_2, (1, 1, 1, 1))
        conv3_3         = self.conv3_3(conv3_3_pad)
        conv3_3_bn      = self.conv3_3_bn(conv3_3)
        relu3_3         = F.relu(conv3_3_bn)
        conv3_4_pad     = F.pad(relu3_3, (1, 1, 1, 1))
        conv3_4         = self.conv3_4(conv3_4_pad)
        conv3_4_bn      = self.conv3_4_bn(conv3_4)
        relu3_4         = F.relu(conv3_4_bn)
        pool3_pad       = F.pad(relu3_4, (0, 1, 0, 1), value=float('-inf'))
        pool3, pool3_idx = F.max_pool2d(pool3_pad, kernel_size=(2, 2), stride=(2, 2), padding=0, ceil_mode=False, return_indices=True)
        conv4_1_pad     = F.pad(pool3, (1, 1, 1, 1))
        conv4_1         = self.conv4_1(conv4_1_pad)
        conv4_1_bn      = self.conv4_1_bn(conv4_1)
        relu4_1         = F.relu(conv4_1_bn)
        conv4_2_pad     = F.pad(relu4_1, (1, 1, 1, 1))
        conv4_2         = self.conv4_2(conv4_2_pad)
        conv4_2_bn      = self.conv4_2_bn(conv4_2)
        relu4_2         = F.relu(conv4_2_bn)
        conv4_3_pad     = F.pad(relu4_2, (1, 1, 1, 1))
        conv4_3         = self.conv4_3(conv4_3_pad)
        conv4_3_bn      = self.conv4_3_bn(conv4_3)
        relu4_3         = F.relu(conv4_3_bn)
        conv4_4_pad     = F.pad(relu4_3, (1, 1, 1, 1))
        conv4_4         = self.conv4_4(conv4_4_pad)
        conv4_4_bn      = self.conv4_4_bn(conv4_4)
        relu4_4         = F.relu(conv4_4_bn)
        pool4_pad       = F.pad(relu4_4, (0, 1, 0, 1), value=float('-inf'))
        pool4, pool4_idx = F.max_pool2d(pool4_pad, kernel_size=(2, 2), stride=(2, 2), padding=0, ceil_mode=False, return_indices=True)
        conv5_1_pad     = F.pad(pool4, (1, 1, 1, 1))
        conv5_1         = self.conv5_1(conv5_1_pad)
        conv5_1_bn      = self.conv5_1_bn(conv5_1)
        relu5_1         = F.relu(conv5_1_bn)
        conv5_2_pad     = F.pad(relu5_1, (1, 1, 1, 1))
        conv5_2         = self.conv5_2(conv5_2_pad)
        conv5_2_bn      = self.conv5_2_bn(conv5_2)
        relu5_2         = F.relu(conv5_2_bn)
        conv5_3_pad     = F.pad(relu5_2, (1, 1, 1, 1))
        conv5_3         = self.conv5_3(conv5_3_pad)
        conv5_3_bn      = self.conv5_3_bn(conv5_3)
        relu5_3         = F.relu(conv5_3_bn)
        conv5_4_pad     = F.pad(relu5_3, (1, 1, 1, 1))
        conv5_4         = self.conv5_4(conv5_4_pad)
        conv5_4_bn      = self.conv5_4_bn(conv5_4)
        relu5_4         = F.relu(conv5_4_bn)
        pool5_pad       = F.pad(relu5_4, (0, 1, 0, 1), value=float('-inf'))
        pool5, pool5_idx = F.max_pool2d(pool5_pad, kernel_size=(2, 2), stride=(2, 2), padding=0, ceil_mode=False, return_indices=True)
        return pool5


    @staticmethod
    def __batch_normalization(dim, name, **kwargs):
        if   dim == 0 or dim == 1:  layer = nn.BatchNorm1d(**kwargs)
        elif dim == 2:  layer = nn.BatchNorm2d(**kwargs)
        elif dim == 3:  layer = nn.BatchNorm3d(**kwargs)
        else:           raise NotImplementedError()

        print("weight_dict: ", __weights_dict)

        if 'scale' in __weights_dict[name]:
            layer.state_dict()['weight'].copy_(torch.from_numpy(__weights_dict[name]['scale']))
        else:
            layer.weight.data.fill_(1)

        if 'bias' in __weights_dict[name]:
            layer.state_dict()['bias'].copy_(torch.from_numpy(__weights_dict[name]['bias']))
        else:
            layer.bias.data.fill_(0)

        layer.state_dict()['running_mean'].copy_(torch.from_numpy(__weights_dict[name]['mean']))
        layer.state_dict()['running_var'].copy_(torch.from_numpy(__weights_dict[name]['var']))
        return layer

    @staticmethod
    def __conv(dim, name, **kwargs):
        if   dim == 1:  layer = nn.Conv1d(**kwargs)
        elif dim == 2:  layer = nn.Conv2d(**kwargs)
        elif dim == 3:  layer = nn.Conv3d(**kwargs)
        else:           raise NotImplementedError()

        layer.state_dict()['weight'].copy_(torch.from_numpy(__weights_dict[name]['weights']))
        if 'bias' in __weights_dict[name]:
            layer.state_dict()['bias'].copy_(torch.from_numpy(__weights_dict[name]['bias']))
        return layer


    @staticmethod
    def __relu(name, **kwargs):
        layer = nn.ReLU(inplace=True)
        return layer
