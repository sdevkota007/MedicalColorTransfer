import torchvision.models as models
import torch

import torch.nn as nn


#******************************* VGG19 STUFF ********************************************

vgg19_model = models.vgg19(pretrained=True, progress=True)
print(vgg19_model)
new_classifier = nn.Sequential(*list(vgg19_model.classifier.children())[:1])
vgg19_model.classifier = new_classifier
print(vgg19_model)
vgg19_model.eval()





#******************************* RESNET STUFF ********************************************

#
# # Load the pretrained model
# resnet_model = models.resnet18(pretrained=True)
# # Use the model object to select the desired layer
#
# print(resnet_model)
#
# layer = resnet_model._modules.get('avgpool')
# print("\n\n\n\n")
# print(layer)