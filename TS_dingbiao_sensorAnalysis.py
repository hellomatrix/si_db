# -*- coding: UTF-8 -*-

## 多进程执行
## 把raw转化为图像之后,再进行处理,分步执行,速度快

import numpy as np
import os
import cv2
import time
import multiprocessing as mp
import matplotlib.pyplot as plt


def worker_raw2Img(file,h,w):

    print(file)

    f = open(file, 'rb')
    hf = np.fromfile(f, dtype=np.uint16)

    t = 0
    img_ori = np.zeros((h+2, w), dtype=np.uint16)
    for i in range(h):
        for j in range(w):
            img_ori[i,j] = hf[t]
            t = t + 1

    img_uin8 = np.array(img_ori[1:-1,:].copy()/2047 * 255,dtype=np.uint8)
    print('maxmum value:', (img_ori[1:-1,:].max(),img_uin8.max()))
    # showImg(img)

    return img_ori[1:-1],img_uin8


def traversalDir_FirstDir(path):
    # 定义一个列表，用来存储结果
    list = []
    # 判断路径是否存在
    if (os.path.exists(path)):
        # 获取该目录下的所有文件或文件夹目录
        files = os.listdir(path)
        for file in files:
            # 得到该文件下所有目录的路径
            m = os.path.join(path, file)
            # 判断该路径下是否是文件夹
            if (os.path.isdir(m)):
                h = os.path.split(m)
                print(h[1])
                list.append(h[1])
        print(list)
    return list

def showImg(img):

    # img = img/65535 * 255
    print(img.max())
    plt.figure()
    plt.imshow(img,cmap=plt.cm.get_cmap('gray'))
    plt.title('')
    plt.show()

# def worker():
#
#     path_ori_j = path_ori + list_ori[j] + '/'
#     path_u8_j = path_u8 + list_u8[j] + '/'
#
#     files_ori = os.listdir(path_ori_j)
#     files_ori.sort(key=lambda x: (int(x.split('_')[0]), int(x.split('_')[-1].split('.')[0])))
#
#     for k in range(len(files_ori)):
#
#         print('The {} image:{}'.format(k, files_u8[k]))
#         img = cv2.imread(path_ori_j + files_ori[k], cv2.CV_16U)
#         img_u8 = cv2.imread(path_u8_j + files_u8[k], cv2.CV_8U)
#         print('maxmum value:', (img.max(), img_u8.max()))
#         small_mask, _, r = findCircle(img_u8)
#         # rs.append(r)
#         # ori_mask=ori_mask+ori_m
#         # showImg(ori_mask)
#
#         if k == 100:
#             print()
#         # temp_img=img.copy()
#         img[small_mask < 1] = 0
#         f_img = f_img + img
#         # showImg(f_img)
#
#         small_mask[small_mask > 0] = 1
#         devide_mask = devide_mask + small_mask
#         # showImg(small_mask)
#
#         if k%100 == 0:
#             print('band:{} folder:{} is running!'.format(band,j))
#
#     im = np.array(f_img / devide_mask, dtype=np.uint16)
#
#     # showImg(np.array(im/2047*255,dtype=np.uint8))
#     # showImg(devide_mask)
#     #
#     # plt.figure()
#     # plt.plot(range(len(rs)),rs)
#     # plt.savefig()
#
#     cv2.imwrite(path_ori + list_ori[j] + '.png', im)
#     cv2.imwrite(path_u8 + list_u8[j] + '_u8.png', np.array(im / 2047 * 255, dtype=np.uint8))

if __name__=='__main__':

    fpath = '../data/TS_dingbiao/J_dingbiao/'
    band_name = '445-463nm'
    path_ori = fpath+band_name+'_ori_imgs/'
    # path_ori = fpath+band_name+'_u8_imgs/'

    zero=[]
    max=[]
    upper=[]
    mean3=[]
    lower=[]
    fix_bad=[]

    num = 2
    for i in range(num-1):
        path0=path_ori+str(i)+'/'
        path1=path_ori+str(i+1)+'/'

        files0 = os.listdir(path0)
        files0.sort(key=lambda x: (int(x.split('_')[0]), int(x.split('_')[-1].split('.')[0])))

        files1 = os.listdir(path1)
        files1.sort(key=lambda x: (int(x.split('_')[0]), int(x.split('_')[-1].split('.')[0])))

        for j in range(27,len(files0),28):
            img0 = cv2.imread(path0 + files0[j], cv2.CV_16U)
            img1 = cv2.imread(path1 + files1[j], cv2.CV_16U)

            # hist1,_ = np.histogram(img0.ravel(), 2047, [0, 2047])
            # plt.plot(hist1 // 100000)

            img0 = np.array(img0,np.int32)
            img1 = np.array(img1,np.int32)

            hist=np.histogram(img0,256,[0,256])
            plt.plot(hist)

            # print(np.max(img0),np.max(img1))

            total = img0.shape[0]*img0.shape[1]

            mean = np.average(img0)
            print(3*mean)

            zero.append(len(np.where(img0==0)[0])/total)
            # zeros1 = len(np.where(img1==0)[0])/total

            max.append(len(np.where(img0==2047)[0])/total)
            # max1 = len(np.where(img1==2047)[0])/total

            th_upper = int(2047-2047*0.01)
            upper.append(len(np.where(img0>th_upper)[0])/total)
            # upper1 = len(np.where(img1>th_upper)[0])/total
            mean3.append(len(np.where(img0>3*mean)[0])/total)
            # if i<num:
            a = len(np.where(img0 > 3*mean)[0])
            b = len(np.where(img1 > 3*mean)[0])

            img0[img0<th_upper]=0
            img1[img1<th_upper]=0

            fix_bad.append((a - len(np.where((img0-img1)!=0)[0]))/a)

            th_lower = int(2047*0.01)
            lower.append(len(np.where(img0<th_lower)[0])/total)
            # lower1 = len(np.where(img1<th_lower)[0])/total

    #
    # plt.figure()
    # plt.plot(range(len(zero)),zero)

    hist1=np.histogram(img0.ravel(),2047,[0,2047])
    plt.plot(hist1/100000)

    plt.figure()
    plt.plot(range(len(max)),max)

    plt.figure()
    plt.plot(range(len(upper)),upper)

    plt.figure()
    plt.plot(range(len(mean3)),mean3)

    # plt.figure()
    # plt.plot(range(len(lower)),lower)

    plt.figure()
    plt.plot(range(len(fix_bad)),fix_bad)

    plt.show()
    print()