import torch
import imp
import numpy as np
import argparse
import cv2


print("[INFO] loading model...")

MainModel = imp.load_source('MainModel', "/home/ninja/PycharmProjects/MedicalColorTransfer/models/vgg19_gray/vgg19_gray_kitmodel.py")
model_pytorch = "/home/ninja/PycharmProjects/MedicalColorTransfer/models/vgg19_gray/vgg19_gray_kitmodel.pth"

imgfile = "/home/ninja/PycharmProjects/MedicalColorTransfer/models/cat.jpg"

parser = argparse.ArgumentParser(description='convert caffe to pytorch')
parser.add_argument('--height', default=224, type=int)
parser.add_argument('--width', default=224, type=int)


args = parser.parse_args()
print(args)


def load_image(imgfile):
    image = cv2.imread(imgfile)
    img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    img_scaled = cv2.resize(img_rgb, dsize=(args.height, args.width), interpolation=cv2.INTER_LINEAR)

    img = np.transpose(img_scaled, (2, 0, 1))
    data = np.expand_dims(img, 0).copy()
    data = np.asarray(data, np.float)/255.0
    return data


img = load_image(imgfile)

the_model = torch.load(model_pytorch)
the_model.eval()

state_dict = the_model.state_dict()

# save state dictionary
# torch.save(state_dict, "/home/ninja/PycharmProjects/MedicalColorTransfer/models/vgg19_gray/vgg19_gray_state_dict.pth")

layers = list(the_model._modules.items())
for i, layer in enumerate(layers):
    print(i, layer)
    print(layer)


data = torch.from_numpy(img).float()
output = the_model(data)

# print(output)


