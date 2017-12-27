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

#
# def makeConf():

def plotLines(start,end,data,title=''):

    plt.figure()
    plt.plot(range(start, end + 1, 1), data)
    plt.title(title)
    plt.show()

    return 0


def rgbOrder(start= 390,end= 1000,rrang=None):

    start=start
    end=end+1
    mid1 = 499
    mid2 = 582
    B = list(np.tile('B', mid1-start))
    G1 = list(np.tile('G',mid2-mid1))
    R = list(np.tile('R', end-mid2))

    return B+G1+R


def getExposureTime(start,end,time,intensities,expect=None):

    # plotLines(start, end, intensities,title='The intensity of last imaging')

    time_exposure=[]
    for i in range(start-start,end+1-start):

        time_exposure.append(expect*time[i]/intensities[i])

    a=np.array(time_exposure.copy())
    testExpTime(start, end, a, expect, intensities, time)

    a[a>500000] = 500000
    a = a.astype(int)
    testExpTime(start,end,a,expect,intensities,time)

    # plotLines(start, end, time,title='The last exposure time')
    # plotLines(start,end,a,title='The new exposure time')

    return a

def testExpTime(start,end,a,expect,intensities,time):

    plotLines(start,end,expect*time/a,'The reconstruction of last intensity')
    plotLines(start,end,intensities*a/time,'The reconstruction of expect intensity')


def getAvgIntensities(start,end,zero,path=None,b=-4):

    files = os.listdir(path)
    files.sort(key=lambda x: int(x[:b]))

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

        Intensity.append(top)

    plotLines(start, end, Intensity,'The intensity of average imgs')
    plotLines(start, end, band,'The band of different lambda')

    return Intensity

def getAvgIntensities_area(start,end,zero,a1,a2,b1,b2,path=None):

    files = os.listdir(path)
    files.sort(key=lambda x: int(x[:-4]))

    img = cv2.imread(path + files[0], cv2.IMREAD_GRAYSCALE)
    mask = ld.getFilterMask(img)

    Intensity=[]
    val=[]
    band=[]
    list = ['G', 'R', 'B','G']
    for i in range(start-zero,end-zero+1):

        f = files[i]
        print(f)

        im = cv2.imread(path+f,cv2.IMREAD_GRAYSCALE)
        img_temp = im[b1:b2,a1:a2]
        mask_temp = mask[b1:b2,a1:a2]

        G1 = np.average(img_temp[mask_temp == 'G1'])
        B = np.average(img_temp[mask_temp == 'B'])
        R = np.average(img_temp[mask_temp == 'R'])
        G2 = np.average(img_temp[mask_temp == 'G2'])

        top = np.max([G1, R, B, G2])
        val.append(top)
        band.append(list[[G1, R, B,G2].index(top)])

        Intensity.append(top)

    plotLines(start, end, Intensity)

    return Intensity


def getTimeConf(fileName):

    time = np.loadtxt(fileName)
    return time


def saveTxt(start,end,data,fileName):

    bands = np.array(list(range(start,end+1,1)))

    info = np.vstack((bands,bands,data))
    info = np.transpose(info.astype(int))

    np.savetxt(fileName+'.txt',info,fmt='%d')


def saveExcel(exp_time,name='exposure_time.xls'):

    book = xlwt.Workbook()
    sheet1 = book.add_sheet('sheet1')

    temp=[]
    for k in range(len(exp_time)):
        temp = temp + exp_time[k]
        for i,value in enumerate(exp_time[k]):
            sheet1.write(i,k,value)

    book.save(name)
    book.save(TemporaryFile())

def testDif():
    start = 390
    end = 1000

    path2 = '../data/test/20171212withoutqd/'
    times2 = getTimeConf(path2+"dn170_exposure_time_after_20171212.txt")
    times1 = getTimeConf(path2+"dn170_exposure_time_after_20171212_sea.txt")

    print(np.average(times1[:,2]-times2[:,2]))


def testImgWithoutQD(path,b=-4,imgs=None):

    start = 390
    end = 1000
    cols, rows, qd_square = ld.getArea(path = '../data/DingBiao/data_20171205/',name='coords_calibration.mat')
    if imgs==None:
        imgs = ld.getImgsOneFolder(path,b=b)
    avgImg= np.average(imgs,0)
    mask = ld.getFilterMask(avgImg)
    avg_cols = ld.getAvgGRBG(avgImg, cols, mask)
    avg_rows = ld.getAvgGRBG(avgImg, rows, mask)
    mix_val = ld.getMixGRBG(col=avg_cols[0],row=avg_rows[0])
    col_bands_all, col_vals_all=ld.showGRBG(imgs,cols,mask)
    rows_bands_all,rows_vals_all=ld.showGRBG(imgs,rows,mask)
    intensities = getAvgIntensities(start, end, start, path,b=b)
    plt.figure()
    plt.plot(range(start,end+1),col_bands_all,'-',label="bands follow col")
    plt.plot(range(start,end+1),rows_bands_all,'-',label="bands follow row")
    plt.legend()
    plt.figure()
    plt.plot(range(start,end+1),col_vals_all,'-',label="values follow col")
    plt.plot(range(start,end+1),rows_vals_all,'-',label="values follow rows")
    plt.plot(range(start,end+1),intensities,'-',label="avg value of all images")
    plt.legend()
    plt.show()


def cal_expTime(path1,path2,name1,name2,expect):

    start = 390
    end = 1000
    intensities = getAvgIntensities(start,end,start,path1)
    times = getTimeConf(path2+name1)
    et = getExposureTime(start,end,times[:,2],intensities,expect)
    name = path2+str(expect)+'_exposure_time_after_'+name2
    saveTxt(start, end, et, name)


if __name__=='__main__':





    # # test light distribution of DN_100
    path1 = '../data/test/20171212withoutqd/20171212/'
    testImgWithoutQD(path1,b=-4)


    # # test light distribution of DN_170
    # path1 = '../data/DingBiao/N.0_201712191159dn170/'
    # testImgWithoutQD(path1,b=-4)


    #### test light distribution of DN_170
    path1 = '../data/DingBiao/N.0_201712191333dn160/'
    testImgWithoutQD(path1,b=-4)

    # # test light distribution of DN_170
    # path1 = '../data/DingBiao/N.0_201712191605dn160/'
    # testImgWithoutQD(path1,b=-4)

    # # # test light distribution of DN_170 with qd
    # sensor ='N.1_DN170_20171218_CALI_DATA'
    # path = '../data/DingBiao/'+sensor+'/DATA/'
    # path1 = '../data/DingBiao/N.1_DN170_20171218_CALI_DATA/DATA/'
    # imgs, f_names = ld.getAvgData(path, sensor)
    # testImgWithoutQD(path,b=-6,imgs=imgs)


    # # # cal exposure time config
    # path1 = '../data/DingBiao/N.0_201712191159dn170/'
    # path2 = '../data/DingBiao/'
    # expect = 170
    # name1= 'dn170_exposure_time_after_20171212.txt'
    # name2 = '201712191159dn170'
    # cal_expTime(path1,path2,name1,name2,expect)

    #
    # # # cal exposure time config
    # path1 = '../data/DingBiao/N.0_201712191333dn160/'
    # path2 = '../data/DingBiao/'
    # expect = 160
    # name1= '160_exposure_time_after_201712191159.txt'
    # name2 = '201712191333dn160'
    # cal_expTime(path1,path2,name1,name2,expect)




    # # # cal exposure time config
    # path1 = '../data/test/20171212withoutqd/20171212/'
    # path2 = '../data/test/20171212withoutqd/'
    # expect = 170
    # start = 390
    # end = 1000
    # intensities = getAvgIntensities(start,end,start,path1)
    # times = getTimeConf(path2+"exposure_time.txt")
    # expect = 170
    # et = getExposureTime(start,end,times[:,2],intensities,expect)
    # name = path2+'dn170_exposure_time_after_20171212'
    # saveTxt(start, end, et, name)


    # # test
    # path1 = '../data/test/20171212withoutqd/20171212/'
    # path2 = '../data/test/20171212withoutqd/'
    # start = 390
    # end = 1000



    # getAvgIntensities_area(start,end,start,a1=1,a2=20,path=path1)
    # getAvgIntensities_area(start,end,start,a1=1000,a2=1020,path=path1)
    # getAvgIntensities_area(start,end,start,a1=1,a2=1020,path=path1)


    # ## test light distribution
    # avgImage =np.average(ld.getImgsOneFolder(path1),0)
    # img = cf.img2Tuple(avgImage)
    # cf.fit_3d(img)


    # # # check the time
    # sensor ='N.1_DN170_20171218_CALI_DATA'
    # path = '../data/DingBiao/'+sensor+'/DATA/'
    # imgs,f_names = ld.getAvgData(path,sensor)
    #
    # getAvgIntensities_area(start,end,start,a1=1217,a2=1229,b1=898,b2=968,path=path1)

print('pause')


