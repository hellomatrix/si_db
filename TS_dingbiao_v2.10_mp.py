
## from raw to final results
## problm: too slow

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

def showImg(img):

    print(img.max())
    plt.figure()
    plt.imshow(img,cmap=plt.cm.get_cmap('gray'))
    plt.title('')
    plt.show()

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


def worker(i,band):

    na_files = files[i:i + size]
    devide_mask = np.ones((h, w), dtype=int)
    f_img = np.zeros((h, w), dtype=np.float)

    for k, f in enumerate(na_files):
        img, img_u8 = raw2Img(path + f, h, w)
        small_mask, _,_= findCircle(img_u8)
        # ori_mask=ori_mask+ori_m
        # showImg(ori_mask)

        # temp_img=img.copy()
        img[small_mask < 1] = 0
        f_img = f_img + img

        small_mask[small_mask > 0] = 1
        devide_mask = devide_mask + small_mask
        # showImg(img_mask)

        if k%100 == 0:
            print('band:{}, I={} is running!'.format(band,i))

    im = np.array(f_img / devide_mask, dtype=np.uint16)
    # showImg(np.array(im, np.uint8))
    # showImg(devide_mask)

    cv2.imwrite(path_ori + str(i) + '.png', im)
    cv2.imwrite(path_u8 + str(i) + '_u8.png', np.array(im / 2047 * 255, dtype=np.uint8))
    # print('ok')


if __name__=='__main__':

    fpath = '../data/TS_dingbiao/J_dingbiao/'
    band_name = '442nm'
    path = fpath+band_name+'/'

    path_ori = fpath+band_name+'_raw2final_ori_imgs/'
    path_u8 = fpath+band_name+'_raw2final_u8_imgs/'

    if os.path.exists(path_ori):
        pass
    else:
        os.makedirs(path_ori, exist_ok=False)

    if os.path.exists(path_u8):
        pass
    else:
        os.makedirs(path_u8, exist_ok=False)

    files = os.listdir(path)
    files.sort(key=lambda x: (int(x.split('_')[0]), int(x.split('_')[-1].split('.')[0])))

    h=2160
    w=2560
    size = 672

    cpus = mp.cpu_count()
    pool = mp.Pool(cpus-cpus//2)

    start = time.time()

    for i in range(0, len(files), size):
        pool.apply_async(worker, args=(i, band_name))

    pool.close()
    pool.join()

    end = time.time()
    print(end)
    print('Time consuming:', end - start)
    print(band_name + ' is ok!')