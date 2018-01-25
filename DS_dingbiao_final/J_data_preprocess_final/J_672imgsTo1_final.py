# -*- coding: UTF-8 -*-

## 重建的图像包括完整边界
## 速度较慢，可以优化

import numpy as np
import os
import matplotlib.pyplot as plt
import cv2
import scipy.io as sio
import copy
import imutils
import time

COLOR_R = (0,0,255)
COLOR_G = (0,255,0)
COLOR_B = (255,0,0)
COLOR_W = (255,255,255)


def showImg(img):

    print(img.max())
    plt.figure()
    plt.imshow(img,cmap=plt.cm.get_cmap('gray'))
    plt.title('')
    plt.show()


def saveImg(img,fname,path_dis):

    print(fname, img.max())
    cv2.imwrite(path_dis+fname+'.png',img)


def findCircle(img):

    # im_big = np.zeros((img.shape[0]+200,img.shape[1]+200),dtype=np.uint8)
    # im_big[100:100 + 2160, 100:100 + 2560] = img

    im_big = img

    img_denoise = cv2.medianBlur(im_big, 11)
    # showImg(thresh1)

    image_show = im_big.copy()

    image_mask = np.zeros_like(im_big)
    ret, thresh = cv2.threshold(img_denoise, 5, 255, cv2.THRESH_BINARY)
    # showImg(thresh)

    ori_mask=thresh.copy()

    cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if imutils.is_cv2() else cnts[1]

    # loop over the contours
    cX, cY=0,0
    r=0
    if (not cnts) == False:
        c = max(cnts, key=cv2.contourArea)

        # find contours in the thresholded image
        (x,y),radius = cv2.minEnclosingCircle(c)

        cX = int(x)
        cY = int(y)

        r =int(radius)

        cv2.circle(image_mask, (cX, cY), int(0.7 * r), (255, 255, 255), -1)
        # showImg(image_mask)

        image_show[image_mask<1]=0
        # showImg(image_show)

    return image_mask,ori_mask,r,(cX, cY)

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


if __name__=='__main__':

    fpath = '../data/TS_dingbiao/J_dingbiao/'
    band_name = '442nm'
    path_ori = fpath+band_name+'_ori_imgs/'
    path_u8 = fpath+band_name+'_u8_imgs/'

    list_ori = traversalDir_FirstDir(path_ori)
    list_u8 = traversalDir_FirstDir(path_u8)

    rs = []

    start = time.process_time()
    print(start)
    for j in range(len(list_ori)):
        path_ori_j = path_ori+list_ori[j]+'/'
        path_u8_j = path_u8+list_u8[j]+'/'

        files_ori = os.listdir(path_ori_j)
        files_ori.sort(key=lambda x: (int(x.split('_')[0]), int(x.split('_')[-1].split('.')[0])))

        files_u8 = os.listdir(path_u8_j)
        files_u8.sort(key=lambda x: (int(x.split('_')[0]), int(x.split('_')[-1].split('.')[0])))

        h, w = 2160, 2560

        cs=[]
        for k in range(len(files_ori)):

            print('The {} image:{}'.format(k,files_u8[k]))
            # img = cv2.imread(path_ori_j+files_ori[k],cv2.CV_16U)
            # im_big = np.zeros((img.shape[0]+200,img.shape[1]+200),dtype=np.uint16)
            # im_big[100:100 + 2160, 100:100 + 2560] = img
            # img = im_big
            #
            img_u8 = cv2.imread(path_u8_j+files_u8[k],cv2.CV_8U)
            # print('maxmum value:', (img.max(), img_u8.max()))

            im_big_u8 = np.zeros((img_u8.shape[0]+200,img_u8.shape[1]+200),dtype=np.uint8)
            im_big_u8[100:100 + 2160, 100:100 + 2560] = img_u8
            img_u8 = im_big_u8

            small_mask,_, r,centeriod = findCircle(img_u8)
            rs.append(r)
            cs.append(centeriod)

        rr = np.reshape(rs,(-1,28))
        # cc = np.reshape(cs,(-1,28))

        rr_new = np.array(rr.copy(),dtype=np.int16)

        # r_new=[]
        for ki,ri in enumerate(rr[2:-2]):
            # r_new.append(np.average(ri[2:-2]))
            rr_new[ki+1][2:-3] = np.average(ri[2:-3])

        rr_new[0,:]=rr_new[0,:]-0.3*rr_new[0,:]
        # rr_new[1]=rr_new[1]-0.1*rr_new[1]
        #
        # rr_new[-1]=rr_new[-1]-0.8*rr_new[-1]
        rr_new[-2,:]=rr_new[-2,:]-0.8*rr_new[-2,:]
        # rr_new[-3]=rr_new[-3]-0.4*rr_new[-3]

        # rr_new[:,-1]=rr_new[:,-1]-0.5*rr_new[:,-1]
        rr_new[1:-2,-2]=rr_new[1:-2,-2]-0.5*rr_new[1:-2,-2]
        # rr_new[:,-3]=rr_new[:,-3]-0.7*rr_new[:,-3]

        #
        rr_new[1:-2,0]=rr_new[1:-2,0]-0.6*rr_new[1:-2,0]
        # rr_new[:][1]=rr_new[:][1]-0.4*rr_new[:][1]

        rr_new = np.reshape(rr_new,(-1,))
        k=0
        f_img = np.zeros((h+200,w+200), dtype=np.float)
        devide_mask = np.zeros((h+200,w+200), dtype=np.uint8)
        for k in range(len(files_ori)):

            print('The {} image:{}'.format(k, files_u8[k]))
            img = cv2.imread(path_ori_j + files_ori[k], cv2.CV_16U)
            im_big = np.zeros((img.shape[0] + 200, img.shape[1] + 200), dtype=np.uint16)
            im_big[100:100 + 2160, 100:100 + 2560] = img
            img = im_big

            img_u8 = cv2.imread(path_u8_j + files_u8[k], cv2.CV_8U)
            print('maxmum value:', (img.max(), img_u8.max()))

            im_big_u8 = np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8)
            im_big_u8[100:100 + 2160, 100:100 + 2560] = img_u8
            img_u8 = im_big_u8
            image_mask = np.zeros_like(im_big,dtype=np.uint8)

            cv2.circle(image_mask, cs[k], int(0.7*rr_new[k]), (255, 255, 255), -1)
            small_mask=image_mask

            # rs.append(r)
            # cs.append(centeriod)
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

        im = np.array(f_img/devide_mask,dtype=np.uint16)[100:-100,100:-100]

        showImg(np.array(im/2047*255,dtype=np.uint8))
        showImg(devide_mask)

        # ret, thresh = cv2.threshold(devide_mask, 0, 255, cv2.THRESH_BINARY)
        # showImg(thresh)

        # plt.figure()
        # plt.plot(range(len(rs)),rs)
        # plt.savefig()

        end=time.process_time()
        print(end-start)
        #
        # cv2.imwrite(path_ori + list_ori[j] + '.png', im)
        # cv2.imwrite(path_u8 + list_u8[j] + '_u8.png', np.array(im/2047*255,dtype=np.uint8))

        print(band_name+' is ok!')