import xlwt
from tempfile import TemporaryFile

import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
import scipy.io as sio
import curve_fitting as cf
import copy
import light_distribution as ld


def makeConf():

    for band in range(400,1000,10):
        for time in range(10,300,10):




def getETv1():

    path = '../data/test/2nd_baoguang/'

    exp_time = []

    start = 390
    end = 1000

    exp_time.append(getExposureTimeOld(start, 455, 500, start, path))
    exp_time.append(getExposureTimeOld(456, 600, 250, start, path))
    exp_time.append(getExposureTimeOld(601, 752, 100, start, path))
    exp_time.append(getExposureTimeOld(753, 770, 80, start, path))
    exp_time.append(getExposureTimeOld(771, 800, 150, start, path))
    exp_time.append(getExposureTimeOld(801, 815, 65, start, path))
    exp_time.append(getExposureTimeOld(816, 820, 40, start, path))
    exp_time.append(getExposureTimeOld(821, 830, 20, start, path))
    exp_time.append(getExposureTimeOld(831, 840, 50, start, path))
    exp_time.append(getExposureTimeOld(841, 850, 100, start, path))
    exp_time.append(getExposureTimeOld(851, 870, 120, start, path))
    exp_time.append(getExposureTimeOld(871, 895, 45, start, path))
    exp_time.append(getExposureTimeOld(896, 920, 150, start, path))
    exp_time.append(getExposureTimeOld(921, 955, 200, start, path))
    exp_time.append(getExposureTimeOld(956, 990, 300, start, path))
    exp_time.append(getExposureTimeOld(991, end, 500, start, path))

    # book = xlwt.Workbook()
    # sheet1 = book.add_sheet('sheet1')
    #
    # temp = []
    # for k in range(len(exp_time)):
    #     temp = temp + exp_time[k]
    #     for i, value in enumerate(exp_time[k]):
    #         sheet1.write(i, k, value)
    # book.save('exposure_time.xls')
    # book.save(TemporaryFile())

    # a = np.array(temp.copy())
    # a[a > 500000] = 500000
    # a = a.astype(int)
    #
    # bands = np.array(list(range(start, end + 1, 1)))

    # interval = 0
    # gap = 1+interval
    # bands1 = np.array(list(range(start,end+1-interval,gap)))
    # bands2 = np.array(list(range(start+interval,end+1,gap)))
    # print(bands1,bands2)

    # info = np.vstack((bands, bands, a))
    # info = np.transpose(info.astype(int))
    #
    # np.savetxt("exposure_time.txt", info, fmt='%d')


    temp_err = []
    temp_et = []
    temp_val = []
    for i in range(len(exp_time)):

        temp_err=temp_err+exp_time[i][1]
        temp_et=temp_et+exp_time[i][0]
        temp_val=temp_val+exp_time[i][2]

    plotLines(start,end,temp_err)
    plotLines(start,end,temp_et)
    plotLines(start,end,temp_val)


    print('ok')

def getExposureTimeOld(start,end,time,zero,path=None):

    files = os.listdir(path)
    files.sort(key=lambda x: int(x[:-4]))

    img = cv2.imread(path + files[0], cv2.IMREAD_GRAYSCALE)
    # h,w = img.shape[0],img.shape[1]
    mask = ld.getFilterMask(img)
    order = rgbOrder()

    list = ['G', 'R', 'B','G']

    time_exposure = []
    val=[]
    band=[]

    err = []
    for i in range(start - zero, end - zero + 1):
        f = files[i]
        print(f)

        temp = cv2.imread(path + f, cv2.IMREAD_GRAYSCALE)

        G1 = np.average(temp[mask == 'G1'])
        R = np.average(temp[mask == 'R'])
        B = np.average(temp[mask == 'B'])
        G2 = np.average(temp[mask == 'G2'])

        top = np.max([G1, R, B, G2])
        val.append(top)
        band.append(list[[G1, R, B,G2].index(top)])

        time_exposure.append(1000 * 100 * time / top)

        if order[i] == band[i+zero-start]:
            err.append(1)
            # print('1')
        else:
            err.append(0)
            # print('0')

    # plotLines(start,end,time_exposure,title='')

    return time_exposure,err,val

def plotLines(start,end,data,title=''):

    plt.figure()
    plt.plot(range(start, end + 1, 1), data)
    plt.title(title)
    plt.show()

    return 0


def rgbOrder(start= 380,end= 1000,rrang=None):

    # start = 380
    # end = 1001
    start=start
    end=end+1
    mid1 = 499
    mid2 = 582
    B = list(np.tile('B', mid1-start))
    G1 = list(np.tile('G',mid2-mid1))
    R = list(np.tile('R', end-mid2))

    return B+G1+R

def getExposureTime(start,end,time,intensities):

    expect = 100
    time_exposure=[]
    for i in range(start-start,end+1-start):

        time_exposure.append(expect*time[i]//intensities[i])

    a=np.array(time_exposure.copy())
    a[a>500000] = 500000
    a = a.astype(int)

    plotLines(start,end,a)

    return a

def getAvgIntensities(start,end,zero,path=None):

    files = os.listdir(path)
    files.sort(key=lambda x: int(x[:-4]))

    img = cv2.imread(path + files[0], cv2.IMREAD_GRAYSCALE)
    mask = ld.getFilterMask(img)

    Intensity=[]
    val=[]
    band=[]
    order = rgbOrder()
    list = ['G', 'R', 'B','G']
    err=[]
    for i in range(start-zero,end-zero+1):

        f = files[i]
        print(f)

        temp = cv2.imread(path+f,cv2.IMREAD_GRAYSCALE)

        G1 = np.average(temp[mask == 'G1'])
        B = np.average(temp[mask == 'B'])
        R = np.average(temp[mask == 'R'])
        G2 = np.average(temp[mask == 'G2'])

        top = np.max([G1, R, B, G2])
        val.append(top)
        band.append(list[[G1, R, B,G2].index(top)])

        if order[i] == band[i+zero-start]:
            err.append(1)
            # print('1')
        else:
            err.append(0)
            # print('0')

        Intensity.append(top)

    plotLines(start, end, Intensity)
    plotLines(start, end, val)
    plotLines(start, end, err)

    return Intensity

def getTimeConf(fileName):

    time = np.loadtxt(fileName)
    return time

def saveTxt(start,end,data,fileName):

    bands = np.array(list(range(start,end+1,1)))

    info = np.vstack((bands,bands,data))
    info = np.transpose(info.astype(int))

    np.savetxt(fileName+'.txt',info,fmt='%d')

if __name__=='__main__':

    # path = '../data/WithoutQDandWithDiffuser/1/'
    # t1, e1 ,val1= getExposureTimeOld(380,450,500,380,path)
    # t2, e2 ,val2=getExposureTimeOld(451,600,300,380,path)
    # t3, e3 ,val3=getExposureTimeOld(601,800,100,380,path)
    # t4, e4 ,val4=getExposureTimeOld(801,900,50,380,path)
    # t5, e5 ,val5=getExposureTimeOld(901,1000,100,380,path)
    #
    # eall= e1+e2+e3+e4+e5
    # plotLines(380,1000,eall)
    # print(eall[470-380:530-380],eall[560-380:600-380],eall[850-380:880-380])
    #
    # tall= t1+t2+t3+t4+t5
    # plotLines(380,1000,tall)
    #
    # vall= val1+val2+val3+val4+val5
    # plotLines(380,1000,vall)


    # 2nd
    getETv1()


    # path = '../data/test/2nd_baoguang/'
    # start = 390
    # end = 1000
    # exp_time = get_time_V10(path, start, end)


    #
    # path = '../data/test/2nd_baoguang/'
    # start = 390
    # end = 1000
    # intensities = getAvgIntensities(start,end,start,path)
    # times = getTimeConf("exposure_time.txt")
    # et = getExposureTime(start,end,times[:,2],intensities)


    #
    #
    #
    # book = xlwt.Workbook()
    # sheet1 = book.add_sheet('sheet1')
    #
    # temp=[]
    # for k in range(len(exp_time)):
    #     temp = temp + exp_time[k]
    #     for i,value in enumerate(exp_time[k]):
    #         sheet1.write(i,k,value)
    #
    # # book.save('exposure_time.xls')
    # # book.save(TemporaryFile())
    #
    # a=np.array(temp.copy())
    # a[a>500000] = 500000
    # a = a.astype(int)
    #
    # bands = np.array(list(range(start,end+1,1)))
    #
    # # interval = 0
    # # gap = 1+interval
    # # bands1 = np.array(list(range(start,end+1-interval,gap)))
    # # bands2 = np.array(list(range(start+interval,end+1,gap)))
    # # print(bands1,bands2)
    #
    # info = np.vstack((bands,bands,a))
    # info = np.transpose(info.astype(int))
    #
    # np.savetxt("exposure_time.txt", info,fmt='%d')
    #
    # plt.show()

    # path = '../data/test/20171212/'
    # start = 390
    # end = 1000
    # intensities = getAvgIntensities(start,end,start,path)
    # times = getTimeConf("exposure_time.txt")
    # et = getExposureTime(start,end,times[:,2],intensities)
    # name = 'exposure_time_20171212'
    # saveTxt(start, end, et, name)


print('pause')


