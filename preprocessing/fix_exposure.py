import cv2
import numpy as np
import subprocess
import os

# Bring your packages onto the path

import sys
sys.path.append(os.path.abspath("../"))

from config.settings import DOWNLOAD_REQUIRED, DATASET_RAW_PATH, SCAN_TYPES, SUPPORTED_IMAGE_FORMATS


URL = "ftp://lhcftp.nlm.nih.gov/Open-Access-Datasets/Visible-Human/Male-Images/PNG_format/radiological/"

def download(url):
    cmd = ["wget", "-m", url , "-P", "../dataset/"]
    subprocess.call(cmd)
    print("\nDownload Completed!!!")


def fixExposure(img, dtype='uint8'):

    if dtype == 'uint8':
        img8 = np.zeros_like(img)
        img8 = cv2.normalize(src=img, dst=img8, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
        img8 = img8.astype('uint8')
        img = img8

    if dtype == 'uint16' :
        assert img.dtype == "uint16"
        img16 = np.zeros_like(img)
        img16 = cv2.normalize(src=img, dst=img16, alpha=0, beta=65536, norm_type=cv2.NORM_MINMAX)
        img = img16

    return img

def isImage(str):
    extension = str.split(".")[-1]
    return True if extension in SUPPORTED_IMAGE_FORMATS else False


def processAllImages(in_path, out_path=None):
    if out_path is None:
        in_path_lst = in_path.split("/")
        root_dir = "/".join(in_path_lst[:-1])
        dir_name = in_path_lst[-1]+ "-fixed"
        out_path = os.path.join(root_dir, dir_name)
        if not os.path.exists(out_path):
            os.makedirs(out_path)


    count = 0
    for file in os.listdir(in_path):
        if isImage(file):
            path = os.path.join(in_path, file)
            img = cv2.imread(path, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
            img = fixExposure(img, dtype = 'uint8')

            filename = os.path.join(out_path, file)
            cv2.imwrite(filename, img)

            count += 1
            print("\nIn: ", path)
            print("Out: ", filename)

    print("{} files have been processed". format(count))


if __name__ == '__main__':
    scantype = SCAN_TYPES[1]
    if DOWNLOAD_REQUIRED:
        download(os.path.join(URL, scantype))


    path = os.path.join(DATASET_RAW_PATH, scantype)
    print(path, os.getcwd())
    processAllImages(path)
