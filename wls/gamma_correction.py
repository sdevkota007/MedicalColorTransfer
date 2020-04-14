# import the necessary packages
from __future__ import print_function
import numpy as np
import cv2


def gamma_corr(img, gamma=1.0):
    #normalize to [0,1]
    img_norm = np.asarray(img, dtype=np.float32) / 255.0

    #apply gamma corr
    img_gamma = np.power(img_norm, gamma)

    # convert back to [0,255]
    img_u8 = np.asarray(img_gamma * 255, dtype=np.uint8)

    return img_u8


# load the original image
original = cv2.imread("src-0028.png", cv2.IMREAD_GRAYSCALE)

# loop over various values of gamma
for gamma in np.arange(0.0, 3.5, 0.5):
    # ignore when gamma is 1 (there will be no change to the image)
    if gamma == 1:
        continue
    # apply gamma correction and show the images

    gamma = gamma if gamma > 0 else 0.1

    img_inv_gamma = gamma_corr(original, gamma=gamma)
    img_gamma = gamma_corr(original, gamma=1/gamma)


    cv2.imshow("Images-Gamma-{}".format(gamma), np.hstack([original, img_inv_gamma, img_gamma]))
    cv2.waitKey(0)