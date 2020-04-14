# original paper implementation
# Paper: https://www.cse.huji.ac.il/~danix/epd/
# Site: https://www.cse.huji.ac.il/~danix/epd/wlsFilter.m

# python implementation
# Site: https://github.com/harveyslash/Deep-Image-Analogy-TF/blob/master/src/WLS.py

#matplot lib plot multiple images
# Site: https://stackoverflow.com/questions/46615554/how-to-display-multiple-images-in-one-figure-correctly/46616645

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
import wls_lab
import matplotlib.pyplot as plt



def plotLambdaVsAlpha(imgA_gray, imgAP_bgr, lambdas, alphas, out_path):
    if not os.path.exists(out_path):
        os.makedirs(out_path)

    imgAP_lab = cv2.cvtColor(imgAP_bgr, cv2.COLOR_BGR2LAB)
    imgAP_l = imgAP_lab[:, :, 0]
    imgAP_a = imgAP_lab[:, :, 1]
    imgAP_b = imgAP_lab[:, :, 2]

    # if imgA.shape != imgAP.shape:
    #     imgA = cv2.resize(imgA , dsize=imgAP.shape[:2] ,  interpolation=cv2.INTER_CUBIC)

    imgA_gray = np.asarray(imgA_gray, dtype=np.float32)
    imgA_gray = imgA_gray / 255
    imgA_gray = wls_lab.replaceZeroes(imgA_gray)

    imgAP_l = np.asarray(imgAP_l, dtype=np.float32)
    imgAP_l = imgAP_l / 255
    imgAP_l = wls_lab.replaceZeroes(imgAP_l)

    ncols = len(lambdas)
    nrows = len(alphas)
    figsize = [nrows*ncols, nrows*ncols]
    fig, ax = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize)

    img_count = 0
    for i, LAMBDA in enumerate(lambdas):
        for j, ALPHA in enumerate(alphas):
            print("WLS 1")
            wls_AP_A = wls_lab.weightedLeastSquare(imgAP_l, imgA_gray, alpha=ALPHA, _lambda=LAMBDA)

            print("WLS 2")
            wls_A_A = wls_lab.weightedLeastSquare(imgA_gray, imgA_gray, alpha=ALPHA, _lambda=LAMBDA)

            imgOut_lP = imgA_gray + wls_AP_A - wls_A_A

            imgOut_lP = cv2.normalize(src=imgOut_lP, dst=imgOut_lP, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
            imgOut_lP = np.asarray(imgOut_lP, dtype=np.uint8)

            imgOut_lP_a_b = np.zeros_like(imgAP_lab)
            imgOut_lP_a_b[:, :, 0] = imgOut_lP
            imgOut_lP_a_b[:, :, 1] = imgAP_a
            imgOut_lP_a_b[:, :, 2] = imgAP_b

            imgOut_lP_a_b = cv2.cvtColor(imgOut_lP_a_b, cv2.COLOR_LAB2BGR)

            #saveimage
            imgOut_path = os.path.join(out_path, "wlsOut_lamb-{}_alph-{}.png".format(LAMBDA, ALPHA))
            print(imgOut_path)
            cv2.imwrite(imgOut_path, imgOut_lP_a_b)


            #convert BRG to RBG for matplotlib and plot image
            imgOut_rgb = cv2.cvtColor(imgOut_lP_a_b, cv2.COLOR_BGR2RGB)
            ax[i][j].set_title("Lambda:{}, Alpha: {}".format(LAMBDA, ALPHA), fontsize=8)
            ax[i][j].imshow(imgOut_rgb)
            ax[i][j].axis('off')

            img_count+=1
            print("Image no.{} plotted out of {}".format(img_count, ncols*nrows))

    plt.tight_layout(True)
    plt.savefig(os.path.join(out_path, "LambdaVsAlpha.png"))
    plt.show("LambdaVsAlpha_lab.png")


    print("Image saved.")



if __name__ == '__main__':
    lambdas = [0.1, 1, 2.0, 10.0]
    alphas = [0.6, 1, 1.4, 2.0]

    imgA = cv2.imread("src-0028.png", cv2.IMREAD_GRAYSCALE)
    imgAP = cv2.imread("img_B-ref-0028-src-0028.png")
    plotLambdaVsAlpha(imgA, imgAP, lambdas, alphas, out_path = "LambdaVsAlpha_lab" )