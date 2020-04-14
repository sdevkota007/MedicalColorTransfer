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
import wls


# values used in Deep Image Analogy paper
LAMBDA = 0.8
ALPHA = 1.0


def processAllImages(img_color_path, img_guide_path, img_wlsOut_path):
    count = 0
    for img_color_name in os.listdir(img_color_path):
        img_guide_name = "src-" + img_color_name.split("-")[-1]
        img_color_file = os.path.join(img_color_path, img_color_name)
        img_guide_file = os.path.join(img_guide_path, img_guide_name)

        img_guide = cv2.imread(img_guide_file)
        img_color = cv2.imread(img_color_file)

        img_guide = np.asarray(img_guide, dtype=np.float32)
        img_guide = img_guide / 255
        img_guide = wls.replaceZeroes(img_guide)

        img_color = np.asarray(img_color, dtype=np.float32)
        img_color = img_color / 255
        img_color = wls.replaceZeroes(img_color)

        print("WLS 1")
        wls_AP_A = wls.weightedLeastSquare(img_color, img_guide, _lambda=LAMBDA, alpha=ALPHA)
        #
        print("WLS 2")
        wls_A_A = wls.weightedLeastSquare(img_guide, img_guide, _lambda=LAMBDA, alpha=ALPHA)

        refine_AB = img_guide + wls_AP_A - wls_A_A

        refine_AB = cv2.normalize(src=refine_AB, dst=refine_AB, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
        refine_AB = np.asarray(refine_AB, dtype=np.uint8)

        img_wlsOut_name = "wlsout-" + img_color_name.split("-")[-1]
        img_wlsOut_file = os.path.join(img_wlsOut_path, img_wlsOut_name)

        if not os.path.exists(img_wlsOut_path):
            os.makedirs(img_wlsOut_path)

        cv2.imwrite(img_wlsOut_file, refine_AB)

        print("Guide Image: ", img_guide_file)
        print("Color Image: ", img_color_file)
        print("WLSOut Image", img_wlsOut_file)
        count += 1

    print("{} files have been processed".format(count))


if __name__ == '__main__':
    img_color_path = "../results/toy_expr/img_B"
    img_guide_path = "../toy_dataset/radiological/mri"
    wls_out_path = "../results/toy_expr/wls_out"
    processAllImages(img_color_path, img_guide_path, wls_out_path)