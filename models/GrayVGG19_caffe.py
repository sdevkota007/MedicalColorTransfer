# loads caffe model and tries to save it to numpy ...

import numpy as np
import caffe
import argparse
import cv2


print("[INFO] loading model...")
prototxt = "vgg19_gray/deploy.prototxt"
model = "vgg19_gray/vgg19_bn_gray_ft_iter_150000.caffemodel"
imgfile = "cat.jpg"

parser = argparse.ArgumentParser(description='convert caffe to pytorch')
parser.add_argument('--height', default=224, type=int)
parser.add_argument('--width', default=224, type=int)
parser.add_argument('--meanB', default=104, type=float)
parser.add_argument('--meanG', default=117, type=float)
parser.add_argument('--meanR', default=123, type=float)
parser.add_argument('--scale', default=255, type=float)
parser.add_argument('--synset_words', default='', type=str)

args = parser.parse_args()
print(args)


def load_image_caffe(imgfile):
    image = caffe.io.load_image(imgfile)
    transformer = caffe.io.Transformer({'data': (1, 3, args.height, args.width)})
    transformer.set_transpose('data', (2, 0, 1))
    # transformer.set_mean('data', np.array([args.meanB, args.meanG, args.meanR]))
    # transformer.set_raw_scale('data', args.scale)
    transformer.set_channel_swap('data', (2, 1, 0))

    image = transformer.preprocess('data', image)
    image = image.reshape(1, 3, args.height, args.width)
    return image


def load_image_cv(imgfile):
    image = cv2.imread(imgfile)
    img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    img_scaled = cv2.resize(img_rgb, dsize=(args.height, args.width), interpolation=cv2.INTER_LINEAR)

    img = np.transpose(img_scaled, (2, 0, 1))
    data = np.expand_dims(img, 0).copy()
    data = np.asarray(data, np.float32)/255.0
    return data

# Reference from:
def forward_caffe(protofile, weightfile, image):
    caffe.set_mode_cpu()

    net = caffe.Net(protofile, 1, weights=weightfile)
    net.blobs['data'].reshape(1, 3, args.height, args.width)
    net.blobs['data'].data[...] = image
    output = net.forward()
    return output, net.blobs, net.params


img_caffe = load_image_caffe(imgfile)
img_cv = load_image_cv(imgfile)


output, caffe_blobs, caffe_params = forward_caffe(prototxt, weightfile=model, image=img_cv)
print(output)
print("\n**********************\n")
print(caffe_blobs)
print("\n**********************\n")
print(caffe_params)
