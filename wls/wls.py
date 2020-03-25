# original paper implementation
# Paper: https://www.cse.huji.ac.il/~danix/epd/
# Site: https://www.cse.huji.ac.il/~danix/epd/wlsFilter.m

# python implementation
# Site: https://github.com/harveyslash/Deep-Image-Analogy-TF/blob/master/src/WLS.py

import cv2
import os
import numpy as np
from scipy.sparse import spdiags
from scipy.sparse.linalg import spsolve

# values used in Deep Image Analogy paper
ALPHA = 1.0
LAMBDA = 0.8


def replaceZeroes(data):
  min_nonzero = np.min(data[np.nonzero(data)])
  data[data == 0] = min_nonzero
  return data



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


    output[:,:,0] = wlsFilter( img_color[:, :, 0], L_img=img_guide[:, :, 0], alpha=alpha, _lambda=_lambda)
    output[:,:,1] = wlsFilter( img_color[:, :, 1], L_img=img_guide[:, :, 1], alpha=alpha, _lambda=_lambda)
    output[:,:,2] = wlsFilter( img_color[:, :, 2], L_img=img_guide[:, :, 2], alpha=alpha, _lambda=_lambda)
    return output


def processAllImages(img_color_path, img_guide_path, img_wlsOut_path):
    count = 0
    for img_color_name in os.listdir(img_color_path):
        img_guide_name = "src-"+img_color_name.split("-")[-1]
        img_color_file = os.path.join(img_color_path, img_color_name)
        img_guide_file = os.path.join(img_guide_path, img_guide_name)

        img_guide = cv2.imread(img_guide_file)
        img_color = cv2.imread(img_color_file)

        img_guide = np.asarray(img_guide, dtype=np.float32)
        img_guide = img_guide / 255
        img_guide = replaceZeroes(img_guide)

        img_color = np.asarray(img_color, dtype=np.float32)
        img_color = img_color / 255
        img_color = replaceZeroes(img_color)

        print("WLS 1")
        filtered_AB = weightedLeastSquare(img_color, img_guide, alpha =ALPHA, _lambda = LAMBDA)
        #
        print("WLS 2")
        filtered_A = weightedLeastSquare(img_guide, img_guide, alpha =ALPHA, _lambda = LAMBDA)

        refine_AB = img_guide + filtered_AB - filtered_A

        refine_AB = cv2.normalize(src=refine_AB, dst = refine_AB, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
        refine_AB = np.asarray(refine_AB, dtype=np.uint8)


        img_wlsOut_name = "wlsout-" + img_color_name.split("-")[-1]
        img_wlsOut_file = os.path.join(img_wlsOut_path, img_wlsOut_name)


        if not os.path.exists(img_wlsOut_path):
            os.makedirs(img_wlsOut_path)

        cv2.imwrite(img_wlsOut_file, refine_AB)

        print("Guide Image: ", img_guide_file)
        print("Color Image: ", img_color_file)
        print("WLSOut Image", img_wlsOut_file)
        count+=1


    print("{} files have been processed". format(count))



if __name__ == '__main__':
    # imgA = cv2.imread("../data/5/content1.png")
    # imgAP = cv2.imread("../results/expr_5/img_AP.png")
    #
    # imgA = cv2.resize(imgA , dsize=imgAP.shape[:2] ,  interpolation=cv2.INTER_CUBIC)
    #
    # imgA = np.asarray(imgA, dtype=np.float32)
    # imgA = imgA / 255
    # imgA = replaceZeroes(imgA)
    # imgAP = np.asarray(imgAP, dtype=np.float32)
    # imgAP = imgAP / 255
    # imgAP = replaceZeroes(imgAP)
    #
    #
    # print("WLS 1")
    # filtered_AB = weightedLeastSquare(imgAP, imgA, alpha =ALPHA, _lambda = LAMBDA)
    #
    # print("WLS 2")
    # filtered_A = weightedLeastSquare(imgA, imgA, alpha =ALPHA, _lambda = LAMBDA)
    #
    # refine_AB = imgA + filtered_AB - filtered_A
    #
    # refine_AB = cv2.normalize(src=refine_AB, dst = refine_AB, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
    # refine_AB = np.asarray(refine_AB, dtype=np.uint8)
    #
    # cv2.imwrite("wlsOut.png", refine_AB)
    #
    # print("Image saved.")
    #
    #
    # ### Save Filtered_AB
    # filtered_AB = cv2.normalize(src=filtered_AB, dst=filtered_AB, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
    # filtered_AB = np.asarray(filtered_AB, dtype=np.uint8)
    # cv2.imwrite("filtered_AB.png", filtered_AB)
    # ### Save Filtered A
    # filtered_A = cv2.normalize(src=filtered_A, dst=filtered_A, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
    # filtered_A = np.asarray(filtered_A, dtype=np.uint8)
    # cv2.imwrite("filtered_A.png", filtered_A)



    img_color_path = "../results/toy_expr/img_B"
    img_guide_path = "../toy_dataset/radiological/mri"
    wls_out_path = "../results/toy_expr/wls_out"
    processAllImages(img_color_path, img_guide_path, wls_out_path)