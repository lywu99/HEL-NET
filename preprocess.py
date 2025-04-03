#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import os
import os.path
import cv2
import numpy as np
import glob

def clahe_gridsize(image_path, mask_path, denoise=False, contrastenhancement=False, brightnessbalance=None,
                   cliplimit=None, gridsize=8):

    bgr = cv2.imread(image_path)

    # brightness balance.
    if brightnessbalance:
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        mask_img = cv2.imread(mask_path, 0)
        brightness = gray.sum() / (mask_img.shape[0] * mask_img.shape[1] - mask_img.sum() / 255.)
        bgr = np.uint8(np.minimum(bgr * brightnessbalance / brightness, 255))

    if contrastenhancement:
        # illumination correction and contrast enhancement.
        lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
        lab_planes = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=cliplimit, tileGridSize=(gridsize, gridsize))
        lab_planes[0] = clahe.apply(lab_planes[0])
        lab = cv2.merge(lab_planes)
        bgr = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

    if denoise:
        bgr = cv2.fastNlMeansDenoisingColored(bgr, None, 10, 10, 1, 3)
        bgr = cv2.bilateralFilter(bgr, 5, 1, 1)

    return bgr

def get_preprocess_images(image_path, img_setname, preprocess, phase_name, setname):

    save_setname = ''
    if phase_name == 'train':
        save_setname = 'TrainingSet'
    elif phase_name == 'test':
        save_setname = 'TestingSet'
    elif phase_name == 'eval':
        save_setname = 'ValidingSet'

    limit = 2
    grid_size = 8
    if not os.path.exists(os.path.join(image_dir, 'Images_CLAHE' + preprocess)):
        os.mkdir(os.path.join(image_dir, 'Images_CLAHE' + preprocess))
    if not os.path.exists(os.path.join(image_dir, 'Images_CLAHE' + preprocess, setname)):
        os.mkdir(os.path.join(image_dir, 'Images_CLAHE' + preprocess, setname))

    meanbright = 0.
    images_number = 0

    #ddr
    imgs_ori = glob.glob(os.path.join(image_dir, 'OriginalImages/' + setname + '/*.jpg'))

    #泊松
    # imgs_ori = glob.glob(os.path.join(image_path, img_setname + '/*.png'))

    preprocess_dict = {'0': [False, False, None], '1': [False, False, meanbright], '2': [False, True, None],
                       '3': [False, True, meanbright], '4': [True, False, None], '5': [True, False, meanbright],
                       '6': [True, True, None], '7': [True, True, meanbright]}
    for img_path in imgs_ori:
        print(img_path)
        img_name = os.path.split(img_path)[-1].split('.')[0]
        mask_path = os.path.join(image_dir, 'Groundtruths', save_setname, 'Mask', img_name + '_MASK.tif')
        clahe_img = clahe_gridsize(img_path, mask_path, denoise=preprocess_dict[preprocess][0],
                                   contrastenhancement=preprocess_dict[preprocess][1],
                                   brightnessbalance=preprocess_dict[preprocess][2], cliplimit=limit,
                                   gridsize=grid_size)

        cv2.imwrite(os.path.join(image_dir, 'Images_CLAHE' + preprocess, setname, img_name+'.jpg'),
                    clahe_img)


if __name__ == '__main__':

    image_dir = './IDRiD'