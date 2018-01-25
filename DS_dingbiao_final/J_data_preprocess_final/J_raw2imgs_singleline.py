# -*- coding: UTF-8 -*-
## 将J项目的原始数据从文本文件转换为图像
## 分别存于_ori_imgs和_u8_imgs文件之内

import numpy as np
import os
import cv2
import time
import matplotlib.pyplot as plt

import multiprocessing as mp
import time


def showImg(img):

    print(img.max())
    plt.figure()
    plt.imshow(img,cmap=plt.cm.get_cmap('gray'))
    plt.title('')
    plt.show()

def raw2Img(file,h,w):

    print(file)
    f = open(file, 'rb')
    hf = np.fromfile(f, dtype=np.uint16)
    t = 0

    h1 = h+2  # the raw image
    img_ori = np.zeros((h1, w), dtype=np.uint16)
    for i in range(h1):
        for j in range(w):
            img_ori[i,j] = hf[t]
            t = t + 1

    img_ori_f = img_ori[1:-1, :]
    img_uin8 = np.array(img_ori_f.copy()/2047 * 255, dtype=np.uint8)
    print('origin img max value:{},8bit img max value:{}'.format(img_ori_f.max(),img_uin8.max()))
    # showImg(img_ori_f)
    # showImg(img_uin8)

    return img_ori_f,img_uin8

def worker(i,na_files):

    j = i // size
    print('=================================================', j)
    path_ori_j = path_ori + str(j) + '/'
    path_u8_j = path_u8 + str(j) + '/'

    if os.path.exists(path_ori_j):
        pass
    else:
        os.makedirs(path_ori_j, exist_ok=False)

    if os.path.exists(path_u8_j):
        pass
    else:
        os.makedirs(path_u8_j, exist_ok=False)


    for k,f in enumerate(na_files):

        img, img_u8 = raw2Img(path + f, h, w)
        cv2.imwrite(path_ori_j + f + '.png', img)
        showImg(img)
        # test = cv2.imread(path_ori_j + f + '.png',cv2.CV_16U)
        # print(np.max(test))
        cv2.imwrite(path_u8_j + f + '.png', img_u8)
        print('folder:{}, image:{},name:{}'.format(i, k, f))

if __name__=='__main__':

    fpath = '../data/TS_dingbiao/J_dingbiao/'
    # band_name = 'single_line_500ms_400-730nm_step_3nm'
    band_name = 'single_line_300ms_400-730nm_step_3nm_test'
    path_ori = fpath+band_name+'_ori_imgs/'
    path_u8 = fpath+band_name+'_u8_imgs/'

    path = fpath + band_name + '/'
    files = os.listdir(path)
    files.sort(key=lambda x: (int(x.split('_')[0]), int(x.split('_')[-1].split('.')[0])))

    size = 38
    h=2160
    w=2560
    start=0

    start_time = time.process_time()
    cpu = mp.cpu_count()
    pool = mp.Pool(cpu - 4)

    for i in range(start, len(files), size):
        na_files = files[i:i + size]
        pool.apply_async(worker,args=(i,na_files))

    pool.close()
    pool.join()

    end=time.process_time()

    print('Time Consuming: ',end-start_time)