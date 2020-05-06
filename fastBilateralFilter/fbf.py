import cv2
import numpy as np


imgA_gray = cv2.imread("src-0028.png", cv2.IMREAD_GRAYSCALE)
imgAP_bgr = cv2.imread("img_B-ref-0028-src-0028.png")
img_out = np.zeros_like(imgAP_bgr)

# fbs = cv2.ximgproc_FastBilateralSolverFilter()
# print(fbs.filter(src=imgAP_bgr, confidence=imgA_gray))
# print(np.array(img_out))


fbs = cv2.ximgproc.createFastBilateralSolverFilter(imgA_gray, 8,8,8)


cv2.imshow("out", img_out)

cv2.waitKey(0)
cv2.destroyAllWindows()
