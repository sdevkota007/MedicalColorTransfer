import numpy as np
import cv2


print("[INFO] loading model...")
prototxt = "vgg19-gray/deploy.prototxt"
model = "vgg19-gray/vgg19_bn_gray_ft_iter_150000.caffemodel"
net = cv2.dnn.readNetFromCaffe(prototxt, model)

print("aa")