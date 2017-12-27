import numpy as np
import os
import matplotlib.pyplot as plt
import cv2
import scipy.io as sio
import copy


def getImgsOneFolder(path,path_dis,h=2162,w=2560,a=0,b=-4):

    files = os.listdir(path)
    files.sort(key = lambda x:int(x[a:b]))

    print(path)
    os.makedirs(path_dis,exist_ok=False)

    imgs=[]
    for fileName in files:
        print(fileName)
        fname = fileName.split('.')[0]

        f = open(path+fileName,'rb')
        hf = np.fromfile(f,dtype=np.uint16)
        img = raw2img(hf, h, w)
        print('maxmum value:',img.max())
        imgs.append(copy.copy(img))

        # showImg(img)
        saveImg(img,fname,path_dis)


def getImgsOneFolderSplit(path,path_dis,h=2162,w=2560):

    files = os.listdir(path)
    files.sort(key = lambda x:(int(x.split('_')[0]),int(x.split('_')[-1].split('.')[0])))
    print(path)

    os.makedirs(path_dis,exist_ok=True)

    imgs=[]
    for fileName in files:
        print(fileName)
        fname = fileName.split('.')[0]

        f = open(path+fileName,'rb')
        hf = np.fromfile(f,dtype=np.uint16)
        img = raw2img(hf, h, w)
        print('maxmum value:',img.max())
        imgs.append(copy.copy(img))

        showImg(img)
        # saveImg(img,fname,path_dis)

    # sio.savemat(path_dis+'imgs.mat', {'imgs': imgs})
    return 0


def showImg(img):

    # img = (img - img.min()) / (img.max() - img.min()) * 255
    img = img/65535 * 255

    print(img.max())
    plt.figure()
    plt.imshow(img, cmap=plt.cm.get_cmap('gray'))
    plt.title('')
    plt.show()

def saveImg(img,fname,path_dis):

    img = img / 65535 * 255
    print(img.max())
    cv2.imwrite(path_dis+fname+'.png',img)

def raw2img(hf,h,w):

    t = 0
    img_temp = np.zeros((h, w), dtype=np.uint16)
    for i in range(h):
        for j in range(w):
            img_temp[i,j] = hf[t]
            t = t + 1

    return img_temp

if __name__=='__main__':


    fpath = '../data/TS_dingbiao/'
    band_name = 'J_499nm'
    path_dis = '../data/TS_dingbiao/'+band_name+'_imgs/'
    getImgsOneFolderSplit(fpath+band_name+'/',path_dis)


    # fpath = '../data/TS_dingbiao/'
    # band_name = '499nm'
    # path_dis = '../data/TS_dingbiao/'+band_name+'_imgs/'
    # getImgsOneFolder(fpath+band_name+'/',path_dis, h=2162, w=2560, a=0, b=-10)
