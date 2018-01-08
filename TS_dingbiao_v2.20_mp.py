# -*- coding: UTF-8 -*-

## 多进程执行
## 把raw转化为图像之后,再进行处理,分步执行,速度快

import numpy as np
import os
import matplotlib.pyplot as plt
import cv2
import scipy.io as sio
import copy
import imutils
import multiprocessing as mp
import time

COLOR_1 = (0,255, 0)
# path_ori = None
# list_ori = None

def showImg(img):

    # img = img/65535 * 255
    print(img.max())
    plt.figure()
    plt.imshow(img,cmap=plt.cm.get_cmap('gray'))
    plt.title('')
    plt.show()


def saveImg(img,fname,path_dis):

    img = img / 65535 * 255
    print(img.max())
    cv2.imwrite(path_dis+fname+'.png',img)


def getValue(sets,imgs):

    img_final = np.zeros_like(imgs[0],dtype=np.uint16)
    k=0
    for i in range(len(sets)):
        for j in range(sets[i]):
            if (not sets[i][j])==False:
                img_final = img_final+imgs[k][sets[i][j]]
                k+=1


def getExtrem(cnts,cX, cY):
    c = cnts
    extLeft = tuple(c[c[:,:,0].argmin()][0])
    extRight = tuple(c[c[:,:,0].argmax()][0])
    extTop = tuple(c[c[:,:,1].argmin()][0])
    extBot = tuple(c[c[:,:,1].argmax()][0])

    return np.min([cX-extLeft[0],extRight[0]-cX,extBot[1]-cY,cY-extTop[1]])

def findCircle(img):

    image = img
    image_show = img.copy()
    image_mask = np.zeros_like(image)

    thresh = copy.copy(cv2.medianBlur(image, 11))
    # showImg(thresh1)

    ret, thresh = cv2.threshold(thresh, 3, 255, cv2.THRESH_BINARY)
    ori_mask=thresh.copy()

    # find contours in the thresholded image
    cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if imutils.is_cv2() else cnts[1]

    # loop over the contours
    r=0
    if (not cnts) == False:
        c = max(cnts, key=cv2.contourArea)
        # compute the center of the contour
        M = cv2.moments(c)
        if(M["m00"]==0):
            print('bad mask!')
            pass
        else:
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])

            r = getExtrem(c, cX, cY)

            cv2.circle(image_mask, (cX, cY), int(0.7*r), (255, 255, 255), -1)
            print('the r:{}'.format(r))

            # showImg(image_show)
    return image_mask,ori_mask,r

def raw2Img(file,h,w):

    print(file)

    f = open(file, 'rb')
    hf = np.fromfile(f, dtype=np.uint16)

    t = 0
    img_ori = np.zeros((h+2, w), dtype=np.uint16)
    for i in range(h):
        for j in range(w):
            img_ori[i,j] = hf[t]
            t = t + 1


    img_uin8 = np.array(img_ori[1:-1,:].copy()/2048 * 255,dtype=np.uint8)
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


def worker(j,band):

    path_ori_j = path_ori + list_ori[j] + '/'
    path_u8_j = path_u8 + list_u8[j] + '/'

    files_ori = os.listdir(path_ori_j)
    files_ori.sort(key=lambda x: (int(x.split('_')[0]), int(x.split('_')[-1].split('.')[0])))

    files_u8 = os.listdir(path_u8_j)
    files_u8.sort(key=lambda x: (int(x.split('_')[0]), int(x.split('_')[-1].split('.')[0])))

    h, w = 2160, 2560
    devide_mask = np.zeros((h, w), dtype=int)
    f_img = np.zeros((h, w), dtype=np.float)
    for k in range(len(files_ori)):

        print('The {} image:{}'.format(k, files_u8[k]))
        img = cv2.imread(path_ori_j + files_ori[k], cv2.CV_16U)
        img_u8 = cv2.imread(path_u8_j + files_u8[k], cv2.CV_8U)
        print('maxmum value:', (img.max(), img_u8.max()))
        small_mask, _, r = findCircle(img_u8)
        # rs.append(r)
        # ori_mask=ori_mask+ori_m
        # showImg(ori_mask)

        if k == 100:
            print()
        # temp_img=img.copy()
        img[small_mask < 1] = 0
        f_img = f_img + img
        # showImg(f_img)

        small_mask[small_mask > 0] = 1
        devide_mask = devide_mask + small_mask
        # showImg(small_mask)

        if k%100 == 0:
            print('band:{} folder:{} is running!'.format(band,j))

    im = np.array(f_img / devide_mask, dtype=np.uint16)

    # showImg(np.array(im/2047*255,dtype=np.uint8))
    # showImg(devide_mask)
    #
    # plt.figure()
    # plt.plot(range(len(rs)),rs)
    # plt.savefig()

    cv2.imwrite(path_ori + list_ori[j] + '.png', im)
    cv2.imwrite(path_u8 + list_u8[j] + '_u8.png', np.array(im / 2047 * 255, dtype=np.uint8))


if __name__=='__main__':

    fpath = '../data/TS_dingbiao/J_dingbiao/'
    band_name = '622-730nm'
    path_ori = fpath+band_name+'_ori_imgs/'
    path_u8 = fpath+band_name+'_u8_imgs/'

    list_ori = traversalDir_FirstDir(path_ori)
    list_u8 = traversalDir_FirstDir(path_u8)

    cpus = mp.cpu_count()
    pool = mp.Pool(cpus-4)

    start = time.time()

    for j in range(24,25):
        pool.apply_async(worker,args=(j,band_name))

    pool.close()
    pool.join()

    end = time.time()
    print(end)
    print('Time consuming:',end-start)
    print(band_name + ' is ok!')