import numpy as np
import os
import cv2
import time
import matplotlib.pyplot as plt
import scipy.io as sio


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

    h1 = h+2
    img_ori = np.zeros((h1, w), dtype=np.uint16)
    for i in range(h1):
        for j in range(w):
            img_ori[i,j] = hf[t]
            t = t + 1

    print('maxmum value:', (img_ori[1:-1, :].max()))
    img_uin8 = np.array(img_ori[1:-1,:].copy()/2047 * 255, dtype=np.uint8)
    print('maxmum value:', (img_ori[1:-1,:].max(),img_uin8.max()))
    # showImg(img)

    return img_ori[1:-1],img_uin8


if __name__=='__main__':


    fpath = '../data/TS_dingbiao/J_dingbiao/'
    band_name = '622-730nm'
    path_ori = fpath+band_name+'_ori_imgs/'
    path_u8 = fpath+band_name+'_u8_imgs/'

    path = fpath + band_name + '/'
    files = os.listdir(path)
    files.sort(key=lambda x: (int(x.split('_')[0]), int(x.split('_')[-1].split('.')[0])))

    size = 672

    h=2160
    w=2560
    j=7
    start=7*672

    for i in range(start, len(files), size):
        na_files = files[i:i+size]

        path_ori_j = path_ori+str(j)+'/'
        path_u8_j = path_u8+str(j)+'/'
        if os.path.exists(path_ori_j):
            pass
        else:
            os.makedirs(path_ori_j, exist_ok=False)

        if os.path.exists(path_u8_j):
            pass
        else:
            os.makedirs(path_u8_j, exist_ok=False)

        j=j+1

        start = time.process_time()
        for k,f in enumerate(na_files):
            img,img_u8 = raw2Img(path+f,h,w)
            cv2.imwrite(path_ori_j + f + '.png', img)
            # test = cv2.imread(path_ori_j + f + '.png',cv2.CV_16U)
            # print(np.max(test))
            cv2.imwrite(path_u8_j + f + '.png', img_u8)
            print('The {} image'.format(i+k),f)

        end=time.process_time()

        print('Time consuming: ',end-start)