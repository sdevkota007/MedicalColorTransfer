import cv2
import numpy as np
import scipy
from scipy import sparse
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import spsolve
from scipy.sparse import spdiags


def weightedLeastSquare(img_color, img_guide, alpha = 1.0, _lambda = 0.8):

    epsilon = 0.0001
    grayImgF = cv2.cvtColor(img_guide, cv2.COLOR_BGR2GRAY)
    gradWeightX = np.zeros(grayImgF.shape, dtype=np.float16)
    gradWeightY = np.zeros(grayImgF.shape, dtype=np.float16)

    rows, cols = img_guide.shape[:2]
    for y in range(0, rows-1):
        for x in range(0, cols-1):
            if x+1 < cols:
                gx = grayImgF[y, x+1] - grayImgF[y,x]
                gradWeightX[y,x] = _lambda / (pow(abs(gx), alpha) + epsilon)

            if (y + 1 < rows):
                gy = grayImgF[y+1, x] - grayImgF[y,x]
                gradWeightY[y,x] = _lambda / (pow(abs(gy), alpha) + epsilon)


    height, width = img_color.shape[:2]
    size = width * height
    n = width * height

    A = scipy.sparse.rand(n, n).toarray()

    bs = np.zeros(shape=(3,n))
    xs = np.zeros(shape=(3,n))


    for y in range(0, height):
        for x in range(0, width):
            a = np.zeros(shape=(5,))
            ii = y*width + x

            #doesnt execute for top row
            if y-1>=0:
                gyw = gradWeightY[y - 1, x]
                a[2] += 1 * gyw
                a[0] -= 1 * gyw
                A[ii, ii - width] = a[0]

            #doesnt execute for left row
            if x-1>=0:
                gxw = gradWeightX[y, x - 1]
                a[2] += 1 * gxw
                a[1] -= 1 * gxw
                A[ii, ii - 1] = a[1]

            #right
            if (x + 1 < width):
                gxw = gradWeightX[y, x]
                a[2] += 1 * gxw
                a[3] -= 1 * gxw
                A[ii, ii + 1] = a[3]

            #bottom
            if (y + 1 < height):
                gyw = gradWeightY[y, x]
                a[2] += 1 * gyw
                a[4] -= 1 * gyw
                A[ii, ii + width] = a[4]

            a[2] += 1
            A[ii, ii] = a[2]

            col = img_color[y,x]
            xs[0][ii] = 0
            xs[1][ii] = 0
            xs[2][ii] = 0
            bs[0][ii] = col[0]
            bs[1][ii] = col[1]
            bs[2][ii] = col[2]

    for ch in range(0,3):
        xs[ch] = spsolve(csr_matrix(A), bs[ch])

    resImg = np.zeros(shape=(height, width, 3))

    for y in range(0, height):
        for x in range(0, width):
            resImg[y,x] = np.array( [xs[0][y * width + x],
				                     xs[1][y * width + x],
				                     xs[2][y * width + x]] )

    return resImg



if __name__ == '__main__':
    imgA = cv2.imread("../data/8/content1-r.png")
    imgAP = cv2.imread("../results/expr_8/img_AP.png")

    # imgA = cv2.resize(imgA , dsize=imgAP.shape[:2] ,  interpolation=cv2.INTER_CUBIC)
    # cv2.imshow('a', imgA)
    # cv2.imshow('ap', imgAP)

    imgA = cv2.resize(imgA, dsize=(50,50), interpolation=cv2.INTER_CUBIC)
    imgAP = cv2.resize(imgAP, dsize=(50, 50), interpolation=cv2.INTER_CUBIC)


    imgA = np.asarray(imgA, dtype=np.float32)
    imgA = imgA / 255
    imgAP = np.asarray(imgAP, dtype=np.float32)
    imgAP = imgAP / 255


    print("WLS 1")
    filtered_AB = weightedLeastSquare(imgAP, imgA)
    print("WLS 2")
    filtered_A = weightedLeastSquare(imgA, imgA)

    refine_AB = imgA + filtered_AB - filtered_A
    # refine_AB = refine_AB * 255

    refine_AB = cv2.normalize(src=refine_AB, dst=refine_AB, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
    refine_AB = np.asarray(refine_AB, dtype=np.uint8)

    cv2.imwrite("wls-slow.png", refine_AB)

    print("Image saved.")

