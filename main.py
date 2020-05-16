import os
from utils import load_image
import argparse
from DeepAnalogy import analogy
import cv2
import numpy as np
import json
import re

def str2bool(v):
    return v.lower() in ('true')


def dumpConfig(content, path):
    output = json.dumps(content, indent=4)
    with open(os.path.join(path, "config.json"), 'w') as f:
        f.write(output)


def main():
    # default
    WEIGHT_CHOICE = 2
    MODEL_CHOICE = 2

    # higher weight choice for photo to photo transfer
    # didnt work out that well
    # WEIGHT_CHOICE = 4

    # setting parameters
    config = dict()

    if WEIGHT_CHOICE == 2:
        config['weights'] = [1.0, 0.8, 0.7, 0.6, 0.1, 0.0]
    elif WEIGHT_CHOICE == 3:
        config['weights'] = [1.0, 0.9, 0.8, 0.7, 0.2, 0.0]
    elif WEIGHT_CHOICE == 4:
        config['weights'] = [3.0, 3.0, 3.0, 3.0, 3.0, 3.0]

    # depending upon the choice of model, select layers:
    # relu_5_1, relu_4_1, relu_3_1, relu_3_1, and relu_1_1
    if MODEL_CHOICE ==1:
        config['model'] = "VGG19"
        layers = [29, 20, 11, 6, 1]
    elif MODEL_CHOICE ==2:
        config['model'] = "VGG19Gray"
        layers = [47, 30, 17, 10, 3]

    params = {
        'layers': layers,

        # default
        # 'propagate_iter': 10,

        # test
        # 'propagate_iter': 1,

        # mod - better results
        'propagate_iter': 20,
    }
    config['params'] = params

    config['sizes'] = [3, 3, 3, 5, 5, 3]
    config['rangee'] = [32, 6, 6, 4, 4, 2]
    config['lr'] = [0.1, 0.005, 0.005, 0.00005]


    # default
    config['deconv_iters'] = 400

    # mod - testing
    # config['deconv_iters'] = 1


    # make the required directories
    content = os.listdir('results')
    count = 1
    for c in content:
        if os.path.isdir('results/' + c):
            count += 1
    save_path = 'results/expr_{}'.format(count)
    save_path_AP = os.path.join(save_path, "AP")
    save_path_B = os.path.join(save_path, "B")

    os.makedirs(save_path)
    os.makedirs(save_path_AP)
    os.makedirs(save_path_B)

    # radiological images
    images_A_path = "toy_dataset/radiological/ctscans/1-denoised"
    # frozen cross section images
    images_BP_path = "toy_dataset/thorax-abdomen-fixed/1-denoised"

    # for all images
    target_images = os.listdir(images_A_path)

    # mod - testing
    # target_images = ['src-0028.png']

    dumpConfig(config, save_path)
    print("Config dumped")

    for img_A_name in target_images:
        img_A_path = os.path.join(images_A_path, img_A_name)
        img_BP_name = "ref-" + img_A_name.split("-")[1]
        img_BP_path = os.path.join(images_BP_path, img_BP_name)

        # load images
        print('Loading images...', end='')
        img_A = cv2.imread(img_A_path)
        img_BP = cv2.imread(img_BP_path)
        print('\rImages loaded successfully!')

        # Deep-Image-Analogy
        print("\n##### Deep Image Analogy - start #####")
        img_AP, img_B, elapse = analogy(img_A, img_BP, config)
        print("##### Deep Image Analogy - end | Elapse:" + elapse + " #####")

        img_AP_name = "img_AP-{0}-{1}.png".format(img_A_name.split(".")[0],
                                                  img_BP_name.split(".")[0])
        img_B_name = "img_B-{0}-{1}.png".format(img_A_name.split(".")[0],
                                                img_BP_name.split(".")[0])

        cv2.imwrite(os.path.join(save_path_AP, img_AP_name), img_AP)
        cv2.imwrite(os.path.join(save_path_B, img_B_name), img_B)

        print('Image saved!')


if __name__=="__main__":
    main()