import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import cv2


cfg = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M']


def make_layers(cfg, batch_norm=False):
    layers = []
    in_channels = 3
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)



class VGGGray(nn.Module):

    def __init__(self):
        super(VGGGray, self).__init__()

        self.data_bn = nn.BatchNorm2d(num_features=3, momentum=0.0)

        self.conv1_1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, padding=1)
        self.conv1_1_bn = nn.BatchNorm2d(num_features=64, momentum=0.0)
        self.relu1_1 = nn.ReLU(inplace=True)
        self.conv1_2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1)
        self.conv1_2_bn = nn.BatchNorm2d(num_features=64, momentum=0.0)
        self.relu1_2 = nn.ReLU(inplace=True)
        self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv2_1 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.conv2_1_bn = nn.BatchNorm2d(num_features=128, momentum=0.0)
        self.relu2_1 = nn.ReLU(inplace=True)
        self.conv2_2 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1)
        self.conv2_2_bn = nn.BatchNorm2d(num_features=128, momentum=0.0)
        self.relu2_2 = nn.ReLU(inplace=True)
        self.maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv3_1 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1)
        self.conv3_1_bn = nn.BatchNorm2d(num_features=256, momentum=0.0)
        self.relu3_1 = nn.ReLU(inplace=True)
        self.conv3_2 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1)
        self.conv3_2_bn = nn.BatchNorm2d(num_features=256, momentum=0.0)
        self.relu3_2 = nn.ReLU(inplace=True)
        self.conv3_3 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1)
        self.conv3_3_bn = nn.BatchNorm2d(num_features=256, momentum=0.0)
        self.relu3_3 = nn.ReLU(inplace=True)
        self.conv3_4 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1)
        self.conv3_4_bn = nn.BatchNorm2d(num_features=256, momentum=0.0)
        self.relu3_4 = nn.ReLU(inplace=True)
        self.maxpool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv4_1 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, padding=1)
        self.conv4_1_bn = nn.BatchNorm2d(num_features=512, momentum=0.0)
        self.relu4_1 = nn.ReLU(inplace=True)
        self.conv4_2 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1)
        self.conv4_2_bn = nn.BatchNorm2d(num_features=512, momentum=0.0)
        self.relu4_2 = nn.ReLU(inplace=True)
        self.conv4_3 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1)
        self.conv4_3_bn = nn.BatchNorm2d(num_features=512, momentum=0.0)
        self.relu4_3 = nn.ReLU(inplace=True)
        self.conv4_4 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1)
        self.conv4_4_bn = nn.BatchNorm2d(num_features=512, momentum=0.0)
        self.relu4_4 = nn.ReLU(inplace=True)
        self.maxpool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv5_1 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1)
        self.conv5_1_bn = nn.BatchNorm2d(num_features=512, momentum=0.0)
        self.relu5_1 = nn.ReLU(inplace=True)
        self.conv5_2 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1)
        self.conv5_2_bn = nn.BatchNorm2d(num_features=512, momentum=0.0)
        self.relu5_2 = nn.ReLU(inplace=True)
        self.conv5_3 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1)
        self.conv5_3_bn = nn.BatchNorm2d(num_features=512, momentum=0.0)
        self.relu5_3 = nn.ReLU(inplace=True)
        self.conv5_4 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1)
        self.conv5_4_bn = nn.BatchNorm2d(num_features=512, momentum=0.0)
        self.relu5_4 = nn.ReLU(inplace=True)
        self.maxpool5 = nn.MaxPool2d(kernel_size=2, stride=2)


    def forward(self, x):
        data_bn = self.data_bn(x)
        conv1_1 = self.conv1_1(data_bn)
        conv1_1_bn = self.conv1_1_bn(conv1_1)
        relu1_1 = self.relu1_1(conv1_1_bn)
        conv1_2 = self.conv1_2(relu1_1)
        conv1_2_bn = self.conv1_2_bn(conv1_2)
        relu1_2 = self.relu1_2(conv1_2_bn)
        maxpool1 = self.maxpool1(relu1_2)

        conv2_1 = self.conv2_1(maxpool1)
        conv2_1_bn = self.conv2_1_bn(conv2_1)
        relu2_1 = self.relu2_1(conv2_1_bn)
        conv2_2 = self.conv2_2(relu2_1)
        conv2_2_bn = self.conv2_2_bn(conv2_2)
        relu2_2 = self.relu2_2(conv2_2_bn)
        maxpool2 = self.maxpool2(relu2_2)

        conv3_1 = self.conv3_1(maxpool2)
        conv3_1_bn = self.conv3_1_bn(conv3_1)
        relu3_1 = self.relu3_1(conv3_1_bn)
        conv3_2 = self.conv3_2(relu3_1)
        conv3_2_bn = self.conv3_2_bn(conv3_2)
        relu3_2 = self.relu3_2(conv3_2_bn)
        conv3_3 = self.conv3_3(relu3_2)
        conv3_3_bn = self.conv3_3_bn(conv3_3)
        relu3_3 = self.relu3_3(conv3_3_bn)
        conv3_4 = self.conv3_4(relu3_3)
        conv3_4_bn = self.conv3_4_bn(conv3_4)
        relu3_4 = self.relu3_4(conv3_4_bn)
        maxpool3 = self.maxpool3(relu3_4)

        conv4_1 = self.conv4_1(maxpool3)
        conv4_1_bn = self.conv4_1_bn(conv4_1)
        relu4_1 = self.relu4_1(conv4_1_bn)
        conv4_2 = self.conv4_2(relu4_1)
        conv4_2_bn = self.conv4_2_bn(conv4_2)
        relu4_2 = self.relu4_2(conv4_2_bn)
        conv4_3 = self.conv4_3(relu4_2)
        conv4_3_bn = self.conv4_3_bn(conv4_3)
        relu4_3 = self.relu4_3(conv4_3_bn)
        conv4_4 = self.conv4_4(relu4_3)
        conv4_4_bn = self.conv4_4_bn(conv4_4)
        relu4_4 = self.relu4_4(conv4_4_bn)
        maxpool4 = self.maxpool4(relu4_4)

        conv5_1 = self.conv5_1(maxpool4)
        conv5_1_bn = self.conv5_1_bn(conv5_1)
        relu5_1 = self.relu5_1(conv5_1_bn)
        conv5_2 = self.conv5_2(relu5_1)
        conv5_2_bn = self.conv5_2_bn(conv5_2)
        relu5_2 = self.relu5_2(conv5_2_bn)
        conv5_3 = self.conv5_3(relu5_2)
        conv5_3_bn = self.conv5_3_bn(conv5_3)
        relu5_3 = self.relu5_3(conv5_3_bn)
        conv5_4 = self.conv5_4(relu5_3)
        conv5_4_bn = self.conv5_4_bn(conv5_4)
        relu5_4 = self.relu5_4(conv5_4_bn)
        maxpool5 = self.maxpool5(relu5_4)

        return maxpool5


def load_image(imgfile):
    image = cv2.imread(imgfile)
    img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    img_scaled = cv2.resize(img_rgb, dsize=(224, 224), interpolation=cv2.INTER_LINEAR)

    img = np.transpose(img_scaled, (2, 0, 1))
    data = np.expand_dims(img, 0).copy()

    # data most probably doesnt need normalization since there is already a normalization
    # layer present in the model
    data = np.asarray(data, np.float)/255.0

    # data = np.asarray(data, np.float)
    return data


if __name__ == '__main__':
    weights_file = "/home/ninja/PycharmProjects/MedicalColorTransfer/models/vgg19_gray/vgg19_gray.pth"

    vgg = VGGGray()

    weights_dict = torch.load(weights_file)
    vgg.load_state_dict(weights_dict)


    # ********************************************************************************
    imgfile = "/home/ninja/PycharmProjects/MedicalColorTransfer/models/cat.jpg"
    img = load_image(imgfile)
    vgg.eval()
    data = torch.from_numpy(img).float()
    output = vgg(data)
    print(output)

