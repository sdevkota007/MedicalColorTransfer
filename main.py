import os
from utils import load_image
import argparse
from DeepAnalogy import analogy
import cv2
import numpy as np
import json

def str2bool(v):
    return v.lower() in ('true')


def dumpConfig(content, path):
    json.dumps(content, indent=4)
    with open(os.path.join(path, "config.json"), 'w') as f:
        f.write(content)


def main():
    # default
    WEIGHT_CHOICE = 2

    # higher weight choice for photo to photo transfer
    # didnt work out that well
    # WEIGHT_CHOICE = 4

    # setting parameters
    config = dict()

    params = {
        'layers': [29, 20, 11, 6, 1],
        'iter': 10,
        # 'iter': 1,
    }
    config['params'] = params

    if WEIGHT_CHOICE == 2:
        config['weights'] = [1.0, 0.8, 0.7, 0.6, 0.1, 0.0]
    elif WEIGHT_CHOICE == 3:
        config['weights'] = [1.0, 0.9, 0.8, 0.7, 0.2, 0.0]
    elif WEIGHT_CHOICE == 4:
        config['weights'] = [3.0, 3.0, 3.0, 3.0, 3.0, 3.0]

    config['sizes'] = [3, 3, 3, 5, 5, 3]
    config['rangee'] = [32, 6, 6, 4, 4, 2]
    config['lr'] = [0.1, 0.005, 0.005, 0.00005]

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

    images_A_path = "toy_dataset/radiological/mri"
    images_BP_path = "toy_dataset/head-no-bg"

    # target_images = os.listdir(images_A_path)
    target_images = ['toy_dataset/radiological/mri/src-0028.png']

    for img_A_name in target_images:
        img_A_path = os.path.join(images_A_path, img_A_name)
        img_BP_name = "ref-" + img_A_name.split("-")[1]
        img_BP_path = os.path.join(images_BP_path, img_BP_name)

        # load images
        print('Loading images...', end='')
        img_A = load_image(img_A_path)
        img_BP = load_image(img_BP_path)
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
        dumpConfig(config, save_path)
        print("Config dumped")


if __name__=="__main__":
    main()



# if __name__ == "__main__":
    # parser = argparse.ArgumentParser()
    #
    # # parser.add_argument('--resize_ratio', type=float, default=0.5)
    # parser.add_argument('--resize_ratio', type=float, default=1)
    # parser.add_argument('--weight', type=int, default=2, choices=[2,3])
    # parser.add_argument('--img_A_path', type=str, default='data/8/content1-r.png')
    # parser.add_argument('--img_BP_path', type=str, default='data/8/style2-r.png')
    # parser.add_argument('--use_cuda', type=str2bool, default=True)
    #
    # args = parser.parse_args()
    #
    #
    # # load images
    # print('Loading images...', end='')
    # img_A = load_image(args.img_A_path, args.resize_ratio)
    # img_BP = load_image(args.img_BP_path, args.resize_ratio)
    # print('\rImages loaded successfully!')
    #
    #
    # # setting parameters
    # config = dict()
    #
    # params = {
    #     'layers': [29,20,11,6,1],
    #     'iter': 10,
    # }
    # config['params'] = params
    #
    # if args.weight == 2:
    #     config['weights'] = [1.0, 0.8, 0.7, 0.6, 0.1, 0.0]
    # elif args.weight == 3:
    #     config['weights'] = [1.0, 0.9, 0.8, 0.7, 0.2, 0.0]
    # config['sizes'] = [3,3,3,5,5,3]
    # config['rangee'] = [32,6,6,4,4,2]
    #
    # config['use_cuda'] = args.use_cuda
    # config['lr'] = [0.1, 0.005, 0.005, 0.00005]
    #
    # # Deep-Image-Analogy
    # print("\n##### Deep Image Analogy - start #####")
    # img_AP, img_B, elapse = analogy(img_A, img_BP, config)
    # print("##### Deep Image Analogy - end | Elapse:"+elapse+" #####")
    #
    # # save result
    # content = os.listdir('results')
    # count = 1
    # for c in content:
    #     if os.path.isdir('results/'+c):
    #         count += 1
    # save_path = 'results/expr_{}'.format(count)
    # os.mkdir(save_path)
    #
    # cv2.imwrite(save_path+'/img_AP.png', img_AP)
    # cv2.imwrite(save_path+'/img_B.png', img_B)
    # print('Image saved!')