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
import wls_lab_gamma


# # values used in Deep Image Analogy paper
LAMBDA = 0.8
ALPHA = 1.0

# mod
# LAMBDA = 1.0
# ALPHA = 1.2

GAMMA = 2.0

def processAllImages(imgAP_path, imgA_path, img_wlsOut_path):
    count = 0
    for imgAP_name in os.listdir(imgAP_path):
        imgA_name = "src-" + imgAP_name.split("-")[-1]
        imgAP_file = os.path.join(imgAP_path, imgAP_name)
        imgA_file = os.path.join(imgA_path, imgA_name)

        imgA_gray = cv2.imread(imgA_file, cv2.IMREAD_GRAYSCALE)
        imgAP_bgr = cv2.imread(imgAP_file)

        # perform inverse gamma operation
        imgA_gray = wls_lab_gamma.gamma_corr(imgA_gray, gamma=1 / GAMMA)
        imgAP_bgr = wls_lab_gamma.gamma_corr(imgAP_bgr, gamma=1 / GAMMA)

        imgAP_lab = cv2.cvtColor(imgAP_bgr, cv2.COLOR_BGR2LAB)
        imgAP_l = imgAP_lab[:, :, 0]
        imgAP_a = imgAP_lab[:, :, 1]
        imgAP_b = imgAP_lab[:, :, 2]

        imgA_gray = np.asarray(imgA_gray, dtype=np.float32)
        imgA_gray = imgA_gray / 255
        imgA_gray = wls_lab_gamma.replaceZeroes(imgA_gray)

        imgAP_l = np.asarray(imgAP_l, dtype=np.float32)
        imgAP_l = imgAP_l / 255
        imgAP_l = wls_lab_gamma.replaceZeroes(imgAP_l)

        print("WLS 1")
        wls_AP_A = wls_lab_gamma.weightedLeastSquare(imgAP_l, imgA_gray, _lambda=LAMBDA, alpha=ALPHA)
        #
        print("WLS 2")
        wls_A_A = wls_lab_gamma.weightedLeastSquare(imgA_gray, imgA_gray, _lambda=LAMBDA, alpha=ALPHA)

        imgOut_lP = imgA_gray + wls_AP_A - wls_A_A

        imgOut_lP = cv2.normalize(src=imgOut_lP, dst=imgOut_lP, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
        imgOut_lP = np.asarray(imgOut_lP, dtype=np.uint8)

        imgOut_lP_a_b = np.zeros_like(imgAP_lab)
        imgOut_lP_a_b[:, :, 0] = imgOut_lP
        imgOut_lP_a_b[:, :, 1] = imgAP_a
        imgOut_lP_a_b[:, :, 2] = imgAP_b

        imgOut_lP_a_b = cv2.cvtColor(imgOut_lP_a_b, cv2.COLOR_LAB2BGR)

        imgOut_lP_a_b = wls_lab_gamma.gamma_corr(imgOut_lP_a_b, gamma=GAMMA)

        img_wlsOut_name = "wlsout-" + imgAP_name.split("-")[-1]
        img_wlsOut_file = os.path.join(img_wlsOut_path, img_wlsOut_name)

        if not os.path.exists(img_wlsOut_path):
            os.makedirs(img_wlsOut_path)

        cv2.imwrite(img_wlsOut_file, imgOut_lP_a_b)

        print("Guide Image: ", imgA_file)
        print("Color Image: ", imgAP_file)
        print("WLSOut Image", img_wlsOut_file)
        count += 1

    print("{} files have been processed".format(count))


if __name__ == '__main__':
    imgAP_path = "../results/expr_5/AP"
    imgA_path = "../toy_dataset/radiological/mri"
    wls_out_path = "../results/expr_5/wls-lab-gamma_out"
    processAllImages(imgAP_path, imgA_path, wls_out_path)