from VGG19 import VGG19
import torch
import torchvision.models as models
import copy
import numpy as np
import cv2


img_gallery = "/home/ninja/PycharmProjects/MedicalColorTransfer/toy_dataset/head-no-bg"
mri_img = "/home/ninja/PycharmProjects/MedicalColorTransfer/toy_dataset/radiological/mri/src-0027.png"
use_cuda = True

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Read Image
img = cv2.imread(mri_img)
print(img.shape)
assert(img.shape[2] == 3)


# ************************************OPTION 1************************************
img_tensor = torch.FloatTensor(img.transpose(2, 0, 1))
img_tensor = img_tensor.to(device)

img_tensor = img_tensor.unsqueeze(0)

# compute 5 feature maps
model = VGG19(device=device)
# data, data_size = model.get_features(img_tensor=img_tensor.clone(), layers=[29])
data, data_size = model.get_features(img_tensor=img_tensor.clone(), layers=[34])

features,  = copy.deepcopy(data[:-1])
features_size = data_size[:-1]
print(features, features_size)
print(torch.max(features), torch.min(features))

# ************************************************************************


# ************************************OPTION 1************************************
# img_tensor = torch.FloatTensor(img_mri.transpose(2, 0, 1))
# img_tensor = img_tensor.to(device)
#
# img_tensor = img_tensor.unsqueeze(0)
#
# # compute 5 feature maps
# vgg19_model = models.vgg19(pretrained=True, progress=True)
#
# features = vgg19_model.features(img_tensor)
# print(features)

# ************************************************************************

