# original paper implementation
# Paper: https://www.cse.huji.ac.il/~danix/epd/
# Site: https://www.cse.huji.ac.il/~danix/epd/wlsFilter.m

# python implementation
# Site: https://github.com/harveyslash/Deep-Image-Analogy-TF/blob/master/src/WLS.py

#######################################################################################
# %   Input arguments:
# %   ----------------
# %     IN              Input image (2-D, double, N-by-M matrix).
# %
# %     lambda          Balances between the data term and the smoothness
# %                     term. Increasing lambda will produce smoother images.
# %                     Default value is 1.0
# %
# %     alpha           Gives a degree of control over the affinities by non-
# %                     lineary scaling the gradients. Increasing alpha will
# %                     result in sharper preserved edges. Default value: 1.2
# %
# %     L               Source image for the affinity matrix. Same dimensions
# %                     as the input image IN. Default: log(IN)
#######################################################################################


import cv2
import os
import numpy as np
from scipy.sparse import spdiags
from scipy.sparse.linalg import spsolve

# values used in Deep Image Analogy paper
LAMBDA = 0.8
ALPHA = 1.0

#for Gamma correction
GAMMA = 2.2

# values used in WLS paper
# LAMBDA = 1.0
# ALPHA = 1.2

def replaceZeroes(data):
  min_nonzero = np.min(data[np.nonzero(data)])
  data[data == 0] = min_nonzero
  return data


def gamma_corr(img, gamma=1.0):
    #normalize to [0,1]
    img_norm = np.asarray(img, dtype=np.float32) / 255.0

    #apply gamma corr
    img_gamma = np.power(img_norm, gamma)

    # convert back to [0,255]
    img_u8 = np.asarray(img_gamma * 255, dtype=np.uint8)

    return img_u8


def weightedLeastSquare(img_color, img_guide=None, _lambda=1.0 , alpha=1.2):
    output = np.zeros_like(img_color)
    if img_guide is None:
        img_guide = np.log(img_color)

    def wlsFilter(IN_img, L_img, _lambda=1.0, alpha=1.2):

        smallNum = 0.0001

        r,c = IN_img.shape
        k = r*c

        dy = np.diff(L_img, 1, 0)

        dy = -_lambda / (np.abs(dy) ** alpha + smallNum)
        dy = np.pad(dy, ((0 ,1),(0,0)),mode='constant')
        dy = dy.flatten()

        dx = np.diff(L_img, 1, 1)
        dx = -_lambda / (np.abs(dx) ** alpha + smallNum)
        dx = np.pad(dx, ((0 ,0),(0,1)),mode='constant')
        dx = dx.flatten()

        B = np.zeros(shape=(dx.shape[0],2))
        B[:,0] = dx
        B[:,1] = dy
        d = np.array([-r,-1])

        A = spdiags(B.T,d,k,k)

        e = dx
        w = np.pad(dx, ((r,0)) ,mode= 'constant')
        w = w[0:-r]

        s = dy
        n = np.pad(dy, ((1,0)), mode= 'constant')
        n = n[0:-1]

        D = 1-(e+w+s+n)
        A = A + A.T + spdiags(D.T, 0, k, k)
        OUT = spsolve(A, IN_img.flatten())

        OUT = np.reshape(OUT, (r, c))

        return OUT


    # output[:,:,0] = wlsFilter( img_color[:, :, 0], L_img=img_guide[:, :, 0], _lambda=_lambda, alpha=alpha)
    # output[:,:,1] = wlsFilter( img_color[:, :, 1], L_img=img_guide[:, :, 1], _lambda=_lambda, alpha=alpha)
    # output[:,:,2] = wlsFilter( img_color[:, :, 2], L_img=img_guide[:, :, 2], _lambda=_lambda, alpha=alpha)

    output = wlsFilter(img_color, L_img=img_guide, _lambda=_lambda, alpha=alpha)

    return output




if __name__ == '__main__':
    imgA_gray = cv2.imread("src-0028.png", cv2.IMREAD_GRAYSCALE)
    imgAP_bgr = cv2.imread("img_B-ref-0028-src-0028.png")

    #perform inverse gamma operation
    imgA_gray = gamma_corr(imgA_gray, gamma=1/GAMMA)
    imgAP_bgr = gamma_corr(imgAP_bgr, gamma=1/GAMMA)

    imgAP_lab = cv2.cvtColor(imgAP_bgr, cv2.COLOR_BGR2LAB)
    imgAP_l = imgAP_lab[:, :, 0]
    imgAP_a = imgAP_lab[:, :, 1]
    imgAP_b = imgAP_lab[:, :, 2]

    # if imgA.shape != imgAP.shape:
    #     imgA = cv2.resize(imgA , dsize=imgAP.shape[:2] ,  interpolation=cv2.INTER_CUBIC)

    imgA_gray = np.asarray(imgA_gray, dtype=np.float32)
    imgA_gray = imgA_gray / 255
    imgA_gray = replaceZeroes(imgA_gray)

    imgAP_l = np.asarray(imgAP_l, dtype=np.float32)
    imgAP_l = imgAP_l / 255
    imgAP_l = replaceZeroes(imgAP_l)


    print("WLS 1")
    wls_AP_A = weightedLeastSquare(imgAP_l, imgA_gray, alpha =ALPHA, _lambda = LAMBDA)

    print("WLS 2")
    wls_A_A = weightedLeastSquare(imgA_gray, imgA_gray, alpha =ALPHA, _lambda = LAMBDA)

    imgOut_lP = imgA_gray + wls_AP_A - wls_A_A

    imgOut_lP = cv2.normalize(src=imgOut_lP, dst = imgOut_lP, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
    imgOut_lP = np.asarray(imgOut_lP, dtype=np.uint8)

    imgOut_lP_a_b = np.zeros_like(imgAP_lab)
    imgOut_lP_a_b[:, :, 0] = imgOut_lP
    imgOut_lP_a_b[:, :, 1] = imgAP_a
    imgOut_lP_a_b[:, :, 2] = imgAP_b

    imgOut_lP_a_b = cv2.cvtColor(imgOut_lP_a_b, cv2.COLOR_LAB2BGR)

    #apply gamma at the end
    imgOut_lP_a_b = gamma_corr(imgOut_lP_a_b, gamma=GAMMA)

    cv2.imwrite("wlsOut_lab_gamma.png".format(LAMBDA, ALPHA), imgOut_lP_a_b)

    print("Image saved.")