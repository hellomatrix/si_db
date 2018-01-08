
## from raw to final results
## problm: too slow


import numpy as np
import os
import matplotlib.pyplot as plt
import cv2
import scipy.io as sio
import copy
import imutils

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

    thresh = copy.copy(cv2.medianBlur(image, 7))
    # showImg(thresh1)

    ret, thresh = cv2.threshold(thresh, 5, 255, cv2.THRESH_BINARY)
    ori_mask=thresh.copy()

    # find contours in the thresholded image
    cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if imutils.is_cv2() else cnts[1]

    # loop over the contours

    if (not cnts) == False:
        c = max(cnts, key=cv2.contourArea)
        # compute the center of the contour
        M = cv2.moments(c)
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])

        r = getExtrem(c, cX, cY)
        cv2.circle(image_mask, (cX, cY), int(0.8*r), (255, 255, 255), -1)

    return image_mask,ori_mask

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


if __name__=='__main__':

    fpath = '../data/TS_dingbiao/'
    band_name = '499nm'

    path = fpath+band_name+'/'

    files = os.listdir(path)
    files.sort(key=lambda x: (int(x.split('_')[0]), int(x.split('_')[-1].split('.')[0])))

    h=2160
    w=2560

    for i in range(0, 10000, 672):
        na_files = files[i:i+672]

        devide_mask = np.ones((h,w), dtype=int)
        f_img = np.zeros((h,w), dtype=np.float)

        for k,f in enumerate(na_files):

            img,img_u8 = raw2Img(path+f,h,w)
            small_mask,_ = findCircle(img_u8)
            # ori_mask=ori_mask+ori_m
            # showImg(ori_mask)

            # temp_img=img.copy()
            img[small_mask<1]=0
            f_img = f_img+img

            small_mask[small_mask>0]=1
            devide_mask = devide_mask+small_mask
            # showImg(img_mask)

            print('The {} image'.format(k))

        im = np.array(f_img/devide_mask,dtype=np.uint16)
        showImg(np.array(im,np.uint8))
        showImg(devide_mask)

        cv2.imwrite(fpath + band_name + '.png', im)
        cv2.imwrite(fpath + band_name + '_u8.png', np.array(im,np.uint8))

        print('ok')