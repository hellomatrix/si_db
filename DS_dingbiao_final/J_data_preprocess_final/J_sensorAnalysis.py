# -*- coding: UTF-8 -*-

import numpy as np
import os
import cv2
import matplotlib.pyplot as plt
import time

def showImg(img):

    print(img.max())
    plt.figure()
    plt.imshow(img,cmap=plt.cm.get_cmap('gray'))
    plt.title('')
    plt.show()

def get_median(data):
     data.sort()
     half = len(data) // 2
     return (data[half] + data[~half]) / 2


def correction_noise(img,mask):

    h,w = np.where(mask==1)

    for yi,xi in zip(h,w):
        a,b = get_range(yi,5)
        temp = np.sum(img[a:b, xi] * (1-mask[a:b, xi]))/(len(np.where(mask[a:b, xi]!=1)[0]))
        img[yi,xi] = temp
        print(yi,xi)

    return img

def get_range(yi,gap):

    a=yi-gap
    b=yi+gap
    if yi-gap<0:
        a=0
        b=yi+gap+abs(yi-gap)
    if yi+gap>2560:
        a=yi-gap-abs(yi-gap)
        b=0
    return a,b

# get the intersection of noises
def noise_intersec(folder):

    val_mean = []

    files0 = os.listdir(folder)
    files0.sort(key=lambda x: (int(x.split('_')[0]), int(x.split('_')[-1].split('.')[0])))
    flag = 1
    ins=[]
    for j in range(27,len(files0),28):

        img0 = cv2.imread(folder + files0[j], cv2.CV_16U)
        img0 = np.array(img0,np.int32)
        img_temp = img0.copy()

        median = get_median(np.reshape(img_temp,(-1,)))

        hist,_ = np.histogram(img0,204,[0,2047])

        mean = np.average(img0)
        Th3 = 3*mean
        Th2 = 2*mean
        Th1 = 1*mean
        t_s= 140

        print('Median:{};3 means:{},{},{} ; smooth area:{}'.format(median,Th1,Th2,Th3,t_s))

        mn = Th3

        if j>27 and j < (len(files0)//28*28):
            img_intersection = np.zeros_like(img0)

            img0_temp = img0.copy()

            if flag==1:

                img_intersection[(img0_temp >= mn) * (img_before == img0_temp)]=1
                img_before_temp = np.copy(img0_temp*img_intersection)
                flag=0
            else:

                img_intersection[(img0_temp >= mn) * (img_before_temp == img0_temp)]=1
                img_before_temp = np.copy(img0_temp*img_intersection)

            ins.append(len(np.where(img_intersection!=0)[0]))
            val_mean.append(np.average(img_before[img_before>Th3]))
            print(len(np.where(img_intersection!=0)[0]),len(np.where(img0_temp >= mn)[0]),len(np.where((img_before-img0_temp)==0)[0]),len(np.where(img_before_temp >=mn)[0]))

            path='../data/TS_dingbiao/0.png'
            img_show = cv2.imread(path, cv2.CV_16U)
            showImg(np.array(img_show/2047*255,dtype=np.uint8))

            img_show[img0_temp >= mn] = 0
            showImg(np.array(img_show/2047*255,dtype=np.uint8))

            mask= np.zeros_like(img0_temp)
            mask[img0_temp >= mn]=1

            img_corr = correction_noise(img_show, mask)
            showImg(np.array(img_corr/2047*255,dtype=np.uint8))

            hist, _ = np.histogram(img_corr, 204, [0, 2047])

            plt.figure()
            plt.plot(hist)
            plt.show()

        img_before = img0.copy()


def noiseInfo(folder):

    max = []
    t_ss = []
    mean1 = []
    mean2 = []
    mean3 = []
    medians = []
    val_mean = []

    files0 = os.listdir(folder)
    files0.sort(key=lambda x: (int(x.split('_')[0]), int(x.split('_')[-1].split('.')[0])))
    ins=[]
    for j in range(27,len(files0),28):

        img0 = cv2.imread(folder + files0[j], cv2.CV_16U)
        img0 = np.array(img0,np.int32)
        img_temp = img0.copy()

        median = get_median(np.reshape(img_temp,(-1,)))

        hist,_ = np.histogram(img0,204,[0,2047])

        total = img0.shape[0]*img0.shape[1]

        mean = np.average(img0)
        Th3 = 3*mean
        Th2 = 2*mean
        Th1 = 1*mean

        max.append(len(np.where(img0==2047)[0]))

        t_s= 140
        t_ss.append(len(np.where(img0>t_s)[0])/total)
        mean1.append(len(np.where(img0>Th1)[0])/total)
        mean2.append(len(np.where(img0>Th2)[0])/total)
        mean3.append(len(np.where(img0>Th3)[0])/total)
        medians.append(len(np.where(img0>median)[0])/total)

        print('Median:{};3 means:{},{},{} ; smooth area:{}'.format(median,Th1,Th2,Th3,t_s))

    plt.figure()
    plt.plot(hist)

    plt.figure()
    plt.plot(range(len(ins)),ins,label='intersection')
    plt.legend()

    plt.figure()
    plt.plot(range(len(val_mean)),val_mean,label='val_mean')
    plt.legend()

    plt.figure()
    plt.plot(range(len(max)),max,label='max')
    plt.legend()

    plt.figure()
    plt.plot(range(len(medians)),medians,label='medians')
    plt.legend()

    plt.figure()
    plt.plot(range(len(t_ss)),t_ss,label='t_ss')
    plt.legend()

    plt.figure()
    plt.plot(range(len(mean1)),mean1,label='1MEAN')
    plt.plot(range(len(mean2)),mean2,label='2MEAN')
    plt.plot(range(len(mean3)),mean3,label='3MEAN')
    plt.plot(range(len(t_ss)),t_ss,label='t_ss')
    plt.legend()

    plt.show()

    return max,medians,t_s,mean1,mean2,mean3,Th3


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

def add(x, y) :
    return x + y

def average_noise(path = '../data/TS_dingbiao/J_dingbiao/'):

    bands=['400-439nm','442nm','445-463nm','466-496nm','499-619','622-730nm']
    # bands=['442nm','445-463nm']
    # bands=['400-439nm','442nm','445-463nm','466-496nm']
    noise_folder=[]
    median_folder=[]
    noise_band=[]
    median_band=[]

    big_100 = np.zeros((100,2160,2560),dtype=np.uint16)
    small_100 = np.ones((100,2160,2560),dtype=np.uint16)*2047

    N_Ex2 = np.zeros((2160,2560),dtype=np.float64)
    E_x_2 = np.zeros((2160,2560),dtype=np.float64)
    HIST = np.zeros(2047,dtype=np.float64)
    N=0

    for b in bands:

        path_ori = path + b + '_ori_imgs/'
        folders = traversalDir_FirstDir(path_ori)
        folders.sort()
        print(path_ori,folders)

        start = time.time()
        for f in folders:
        # for f in range(0,2):
        #     f_path_ori = path_ori+folders[f]+'/'
            f_path_ori = path_ori+f+'/'
            files0 = os.listdir(f_path_ori)
            files0.sort(key=lambda x: (int(x.split('_')[0]), int(x.split('_')[-1].split('.')[0])))
            print(f_path_ori,files0)

            noise_imgs_temp = []
            for j in range(27, len(files0), 28):
                img0 = 1.0*cv2.imread(f_path_ori + files0[j], cv2.CV_16U)
                noise_imgs_temp.append(img0)

                N_Ex2 = N_Ex2+np.square(img0)
                E_x_2 = E_x_2+img0

                np.maximum(big_100[0,:,:],np.uint16(img0),big_100[0,:,:])
                big_100.sort(axis=0)

                # np.minimum(small_100[-1,:,:],np.uint16(img0),small_100[-1,:,:])
                # small_100.sort(axis=0)

                HIST= HIST+1.0*(np.histogram(img0, bins=2047, density=True)[0])

                N=N+1
                print('img:{},time:{}'.format(N,(time.time()-start)))

            # noise_folder.append(np.average(noise_imgs_temp, 0))

        # noise_band
        avg = 1.0*E_x_2/N
        sigma = np.sqrt(1.0*N_Ex2/N - np.square(avg))
        total = avg.shape[0]*avg.shape[1]
        HIST_avg = HIST/N

        acc_HIST = []
        for m in range(len(HIST_avg)):
            acc_HIST.append(np.sum(HIST_avg[:m]))
        acc_HIST = np.array(acc_HIST)
        median = np.where(np.abs(acc_HIST- 0.5) == np.min(np.abs(acc_HIST - 0.5)))[0][0]

        # print(sigma-covs)
    del noise_folder

    avg_sigma=avg+3*sigma

    mask = np.zeros_like(img0,dtype=np.uint8)
    mask_all = np.zeros_like(img0,dtype=np.uint8)
    mask[avg>4*median]=1
    print(np.sum(mask)/total)

    mask = np.zeros_like(img0,dtype=np.uint8)
    for ii in range(big_100.shape[0]):
        mask_temp = np.zeros_like(img0,dtype=np.uint8)
        mask_temp[big_100[ii,:,:]>4*median]=1
        mask_all = mask_all+mask_temp

    print(np.sum(mask_all!=0) / total)
    mask[mask_all>N//1000]=1
    print(np.sum(mask) / total)
    mask[avg > 4 * median] = 1
    print(np.sum(mask) / total)

    print(time.time()-start)

    plt.figure()
    plt.imshow(mask)
    plt.legend()

    print()

if __name__=='__main__':

    # fpath = '../data/TS_dingbiao/J_dingbiao/'
    # band_name = '400-439nm'
    # # band_name = '442nm'
    # # band_name = '445-463nm'
    # band_name = '622-730nm'
    # path_ori = fpath+band_name+'_ori_imgs/'
    #
    # bp_list = []
    # for i in range(0,10):
    #     path0=path_ori+str(i)+'/'
    #     # path1=path_ori+str(i+1)+'/'
    #
    #     files0 = os.listdir(path0)
    #     files0.sort(key=lambda x: (int(x.split('_')[0]), int(x.split('_')[-1].split('.')[0])))
    #
    #     # files1 = os.listdir(path1)
    #     # files1.sort(key=lambda x: (int(x.split('_')[0]), int(x.split('_')[-1].split('.')[0])))
    #
    #     # max, medians, t_s, mean1, mean2, mean3, TH = noiseInfo(path0)
    #
    #     noise_intersec(path0)


    ##=================================================== all noise================================

    average_noise()