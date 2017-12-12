
import xlwt
from tempfile import TemporaryFile

import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
import scipy.io as sio
import curve_fitting as cf
import copy
import  light_distribution as ld

def getExposureTime(start,end,time,zero,path=None):

    files = os.listdir(path)
    files.sort(key=lambda x: int(x[:-4]))

    img = cv2.imread(path + files[0], cv2.IMREAD_GRAYSCALE)
    # h,w = img.shape[0],img.shape[1]
    mask = ld.getFilterMask(img)

    time_exposure=[]
    for i in range(start-zero,end-zero+1):

        f = files[i]
        print(f)

        temp = cv2.imread(path+f,cv2.IMREAD_GRAYSCALE)

        G1 = np.average(temp[mask == 'G1'])
        B = np.average(temp[mask == 'B'])
        R = np.average(temp[mask == 'R'])
        G2 = np.average(temp[mask == 'G2'])

        top = np.max([G1,B,R,G2])
        # top=R
        # top=B

        time_exposure.append(1000*100*time/top)

    plt.figure()
    plt.plot(range(start,end+1,1),time_exposure)
    plt.title(('exposure time(ms) : '+str(time)))

    return time_exposure

if __name__=='__main__':

    # path = '../data/WithoutQDandWithDiffuser/1/'
    # getExposureTime(380,450,500,390,path)
    # getExposureTime(451,600,300,380,path)
    # getExposureTime(601,800,100,380,path)
    # getExposureTime(801,900,50,380,path)
    # getExposureTime(901,1000,100,380,path)

    path = '../data/test/2nd_baoguang/'

    exp_time=[]

    start = 390
    end = 1000

    exp_time.append(getExposureTime(start,455,500,start,path))
    exp_time.append(getExposureTime(456,600,250,start,path))
    exp_time.append(getExposureTime(601,752,100,start,path))
    exp_time.append(getExposureTime(753,770,80,start,path))
    exp_time.append(getExposureTime(771,800,150,start,path))
    exp_time.append(getExposureTime(801,815,65,start,path))
    exp_time.append(getExposureTime(816,820,40,start,path))
    exp_time.append(getExposureTime(821,830,20,start,path))
    exp_time.append(getExposureTime(831,840,50,start,path))
    exp_time.append(getExposureTime(841,850,100,start,path))
    exp_time.append(getExposureTime(851,870,120,start,path))
    exp_time.append(getExposureTime(871,895,45,start,path))
    exp_time.append(getExposureTime(896,920,150,start,path))
    exp_time.append(getExposureTime(921,955,200,start,path))
    exp_time.append(getExposureTime(956,990,300,start,path))
    exp_time.append(getExposureTime(991,end,500,start,path))

    book = xlwt.Workbook()
    sheet1 = book.add_sheet('sheet1')

    temp=[]
    for k in range(len(exp_time)):
        temp = temp + exp_time[k]
        for i,value in enumerate(exp_time[k]):
            sheet1.write(i,k,value)

    # book.save('exposure_time.xls')
    # book.save(TemporaryFile())

    a=np.array(temp.copy())
    a[a>500000] = 500000
    a = a.astype(int)

    bands = np.array(list(range(start,end+1,1)))

    # interval = 0
    # gap = 1+interval
    # bands1 = np.array(list(range(start,end+1-interval,gap)))
    # bands2 = np.array(list(range(start+interval,end+1,gap)))
    # print(bands1,bands2)

    info = np.vstack((bands,bands,a))
    info = np.transpose(info.astype(int))

    np.savetxt("exposure_time.txt", info,fmt='%d')

    plt.show()