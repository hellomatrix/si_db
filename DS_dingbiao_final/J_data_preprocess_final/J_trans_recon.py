# -*- coding: UTF-8 -*-

import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
import xlrd
import h5py

COLOR_1 = (255, 0, 0)

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

def readExcel(path):
    data = xlrd.open_workbook(path,encoding_override='utf-8')
    tabel = data.sheets()[0]

    monos = [tabel.col_values(0), tabel.col_values(1)]

    return monos

def get_mask():

    path1 = '../data/TS_dingbiao/0.png'
    path2 = '../data/TS_dingbiao/20171222_101547.raw.png'
    img_show = cv2.imread(path1, cv2.CV_16U)
    img0_temp = cv2.imread(path2, cv2.CV_16U)
    # showImg(np.array(img_show / 2047 * 255, dtype=np.uint8))

    mn = 3*np.average(img0_temp)
    print(mn)
    # img_show[img0_temp >= mn] = 0
    # showImg(np.array(img_show / 2047 * 255, dtype=np.uint8))

    mask = np.zeros_like(img0_temp)
    mask[img0_temp >= mn] = 1

    # showImg(np.array(img_corr / 2047 * 255, dtype=np.uint8))

    return mask

if __name__=='__main__':

    path ='../data/TS_dingbiao/J_dingbiao/J_dingbiao_jh/'

    monos = readExcel('../data/TS_dingbiao/180116.xlsx')

    co = sio.loadmat('../data/TS_dingbiao/coords.mat')
    TT = h5py.File('../data/TS_dingbiao/image_all_new.mat')
    # TT = sio.loadmat('../data/TS_dingbiao/image_all.mat')

    folders = traversalDir_FirstDir(path)
    folders.sort()

    imgs=[]
    for f in folders:

        files_ori = os.listdir(path+f+'/')
        files_ori.sort(key=lambda x:x[0:-3])
        for i in range(len(files_ori)):
            imgs.append(cv2.imread(path+f+'/'+str(i)+'.png',cv2.CV_16U))
            print(i)

    xl = np.reshape(co['x_left'],(-1))
    xr = np.reshape(co['x_right'],(-1))
    yl = np.reshape(co['y_left'],(-1))
    yr = np.reshape(co['y_right'],(-1))

    mask = get_mask()

    ss=[]
    imgs=np.array(imgs)
    for i in range(len(yr)):
        im_temp = imgs[:,int(xl[i]):int(xr[i]),int(yl[i]):int(yr[i])]
        s=[]
        for j in range(im_temp.shape[0]):
            im_ttemp = im_temp[j]
            s.append(np.average(im_ttemp[mask[int(xl[i]):int(xr[i]),int(yl[i]):int(yr[i])]!=1]))

        ss.append(s)
        # plt.figure()
        # plt.plot(range(400,733,3),s)
        # plt.show()

    ss_final = np.array(ss)/monos[1]

    plt.figure()
    for k in range(len(ss_final)):
        plt.plot(range(400,733,3),ss_final[k])

    plt.figure()
    for k in range(len(ss)):
        plt.plot(range(400,733,3),ss[k])

    plt.figure()
    plt.plot(monos[0],monos[1])

    plt.show()