import cv2
import numpy as np

imgAP_bgr = cv2.imread("img_B-ref-0028-src-0028.png")
imgA_gray = cv2.imread("src-0028.png")
img_conf = cv2.cvtColor(imgAP_bgr, cv2.COLOR_BGR2GRAY)

img_out = np.zeros_like(imgAP_bgr)

# fbs = cv2.ximgproc_FastBilateralSolverFilter()
# img_out = fbs.filter(src=imgAP_bgr, confidence=imgA_gray, dst=img_out)
# # # print(fbs.filter())
# # print(fbs)
# print(np.array(img_out))



# img_out = cv2.ximgproc_FastBilateralSolverFilter.filter(src=imgAP_bgr, confidence=imgA_gray, dst=img_out)
# print(np.array(img_out))


def fbsfilter(img_color, img_guide, confidence, sigma_spatial=32, sigma_luma=4, sigma_chroma=8):
    fbs = cv2.ximgproc.createFastBilateralSolverFilter(guide=img_guide,
                                                       sigma_spatial=sigma_spatial,
                                                       sigma_luma=sigma_luma,
                                                       sigma_chroma=sigma_chroma)
    img_out = fbs.filter(src=img_color,
                         confidence=confidence)
    return img_out


fbs_AP_A = fbsfilter(imgAP_bgr, imgA_gray, confidence=img_conf)
fbs_A_A = fbsfilter(imgA_gray, imgA_gray, confidence=img_conf)

refine_AB = imgA_gray + fbs_AP_A - fbs_A_A


cv2.imshow("out", refine_AB)

cv2.waitKey(0)
cv2.destroyAllWindows()
