import cv2
import numpy as np
import argparse

parser = argparse.ArgumentParser()

# parser.add_argument('--resize_ratio', type=float, default=0.5)
# parser.add_argument('--weight', type=int, default=2, choices=[2, 3])
parser.add_argument('--img_mri', type=str, default='data/7/content1.png')


args = parser.parse_args()
img = cv2.imread(args.img)
dim = (448, 448)
img_out = cv2.resize(img, dsize = dim)
img_name = args.img.split(".")
img_name = "".join(img_name[:-1])+ "-r."+ img_name[-1]
print(img_name)
cv2.imwrite(img_name, img_out)
