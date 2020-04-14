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
import wls
import matplotlib.pyplot as plt



def plotLambdaVsAlpha(imgA, imgAP, lambdas, alphas, out_path):
    if not os.path.exists(out_path):
        os.makedirs(out_path)

    imgA = cv2.resize(imgA, dsize=imgAP.shape[:2], interpolation=cv2.INTER_CUBIC)

    imgA = np.asarray(imgA, dtype=np.float32)
    imgA = imgA / 255
    imgA = wls.replaceZeroes(imgA)

    imgAP = np.asarray(imgAP, dtype=np.float32)
    imgAP = imgAP / 255
    imgAP = wls.replaceZeroes(imgAP)

    ncols = len(lambdas)
    nrows = len(alphas)
    figsize = [nrows*ncols, nrows*ncols]
    fig, ax = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize)

    img_count = 0
    for i, LAMBDA in enumerate(lambdas):
        for j, ALPHA in enumerate(alphas):
            print("WLS 1")
            filtered_AB = wls.weightedLeastSquare(imgAP, imgA, alpha=ALPHA, _lambda=LAMBDA)

            print("WLS 2")
            filtered_A = wls.weightedLeastSquare(imgA, imgA, alpha=ALPHA, _lambda=LAMBDA)

            refine_AB = imgA + filtered_AB - filtered_A

            refine_AB = cv2.normalize(src=refine_AB, dst=refine_AB, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
            refine_AB = np.asarray(refine_AB, dtype=np.uint8)

            #saveimage
            refine_AB_path = os.path.join(out_path, "wlsOut_lamb-{}_alph-{}.png".format(LAMBDA, ALPHA))
            print(refine_AB_path)
            cv2.imwrite(refine_AB_path, refine_AB)


            #convert BRG to RBG for matplotlib and plot image
            refine_AB = cv2.cvtColor(refine_AB, cv2.COLOR_BGR2RGB)
            ax[i][j].set_title("Lambda:{}, Alpha: {}".format(LAMBDA, ALPHA), fontsize=8)
            ax[i][j].imshow(refine_AB)
            ax[i][j].axis('off')

            img_count+=1
            print("Image no.{} plotted out of {}".format(img_count, ncols*nrows))

    plt.tight_layout(True)
    plt.savefig(os.path.join(out_path, "LambdaVsAlpha.png"))
    plt.show("LambdaVsAlpha.png")

    print("Image saved.")


if __name__ == '__main__':
    lambdas = [0.1, 1, 2.0, 10.0]
    alphas = [0.6, 1, 1.4, 2.0]

    imgA = cv2.imread("src-0028.png")
    imgAP = cv2.imread("img_B-ref-0028-src-0028.png")
    plotLambdaVsAlpha(imgA, imgAP, lambdas, alphas, out_path = "LambdaVsAlpha" )