# loads caffe model and tries to save it to numpy ...
import torch
import numpy as np
import caffe
import argparse
import cv2
from vgg19_gray.vgg19_gray import VGGGray


print("[INFO] loading model...")
prototxt = "/home/ninja/PycharmProjects/MedicalColorTransfer/models/vgg19_gray/deploy.prototxt"
model_caffe = "/home/ninja/PycharmProjects/MedicalColorTransfer/models/vgg19_gray/vgg19_bn_gray_ft_iter_150000.caffemodel"

model_pytorch = "/home/ninja/PycharmProjects/MedicalColorTransfer/models/vgg19_gray/vgg19_gray.pth"

imgfile = "/home/ninja/PycharmProjects/MedicalColorTransfer/models/cat.jpg"

parser = argparse.ArgumentParser(description='compare caffe and pytorch')
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

    # data most probably doesnt need normalization since there is already a normalization
    # layer present in the model
    # data = np.asarray(data, np.float)/255.0

    data = np.asarray(data, np.float)
    return data

# Reference from:
def forward_caffe(protofile, weightfile, image):
    caffe.set_mode_cpu()

    net = caffe.Net(protofile, 1, weights=weightfile)
    net.blobs['data'].reshape(1, 3, args.height, args.width)
    net.blobs['data'].data[...] = image
    output = net.forward()
    return output['pool5'], net.blobs, net.params

def forward_pytorch(weightfile, image):
    vgg = VGGGray()

    weights_dict = torch.load(weightfile)
    vgg.load_state_dict(weights_dict)

    vgg.eval()

    data = torch.from_numpy(image).float()
    output = vgg(data)
    return output



img = load_image(imgfile)

output_caffe, blobs_caffe, params_caffe = forward_caffe(prototxt, weightfile=model_caffe, image=img)

output_torch = forward_pytorch(weightfile=model_pytorch, image=img)

print(output_caffe)
print("\n**********************\n")
print(output_torch)

