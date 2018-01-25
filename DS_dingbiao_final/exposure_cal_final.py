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

COLOR_1 = (255, 0, 0)

def getTheBiggestBand(G1RBG2):

    top = np.max(G1RBG2)
    band=G1RBG2.index(top)
    if band == 3:
        band=0

    return band,top

def getAvgGRBG(img,area,mask,show=0):

    square = area
    show_img = img.astype(np.uint8).copy()

    intensity = np.zeros((square.shape[0],square.shape[1],4),dtype=float)

    bands=[]
    vals=[]
    for i in range(square.shape[0]):
        for j in range(square.shape[1]):

            x1 = square[i, j, 0]
            x2 = square[i, j, 2]

            y1 = square[i, j, 1]
            y2 = square[i, j, 3]

            temp = img[ y1:y2, x1:x2 ]
            mask_temp = mask[ y1:y2 , x1:x2 ]

            # print([y1,y2, x1,x2])
            cv2.rectangle(show_img, (x1, y1), (x2, y2), COLOR_1, 2)

            G1=np.average(temp[mask_temp == 'G1'])
            R=np.average(temp[mask_temp == 'R'])
            B=np.average(temp[mask_temp == 'B'])
            G2=np.average(temp[mask_temp == 'G2'])

            G1RBG2 = [G1,R,B,G2]
            intensity[i,j,:] = G1RBG2
            band,val = getTheBiggestBand(G1RBG2)
            bands.append(band)
            vals.append(val)

    # if show==1:
    #     # showGrayImg(show_img,'')
    #
    #     cf.fit_3d(cf.img2Tuple(intensity[:,:,0]), 'getAvgGRBG:band G1')
    #     cf.fit_3d(cf.img2Tuple(intensity[:,:,1]), 'getAvgGRBG:band R')
    #     cf.fit_3d(cf.img2Tuple(intensity[:,:,2]), 'getAvgGRBG:band B')
    #     cf.fit_3d(cf.img2Tuple(intensity[:,:,3]), 'getAvgGRBG:band G2')

    return np.average(vals)

def getAvgIntensities_area(area,start,end,zero,path=None,b=-4):

    files = os.listdir(path)
    files.sort(key=lambda x: int(x[:b]))

    img = cv2.imread(path + files[0], cv2.IMREAD_GRAYSCALE)
    mask = ld.getFilterMask(img)

    intensities=[]
    for i in range(start-zero,end-zero+1):

        f = files[i]
        print(f)
        temp = cv2.imread(path+f,cv2.IMREAD_GRAYSCALE)
        intensities.append(getAvgGRBG(temp,area,mask,show=0))

    return intensities

def getAvgIntensities(start,end,zero,path=None,b=-4):

    files = os.listdir(path)
    files.sort(key=lambda x: int(x[:b]))

    img = cv2.imread(path + files[0], cv2.IMREAD_GRAYSCALE)
    mask = ld.getFilterMask(img)

    Intensity=[]
    val=[]
    band=[]
    #order = rgbOrder()
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

    #plotLines(start, end, Intensity,'The intensity of average imgs')
    #plotLines(start, end, band,'The band of different lambda')

    return Intensity


def getTimeConf(fileName):

    time = np.loadtxt(fileName)
    return time


def saveTxt(start,end,data,fileName):

    bands = np.array(list(range(start,end+1,1)))

    info = np.vstack((bands,bands,data))
    info = np.transpose(info.astype(int))

    np.savetxt(fileName+'.txt',info,fmt='%d')

def testExpTime(start,end,zero,a,b,expect,intensities,time1):

    time = time1[start-zero:end+1-zero]

    plotLines(start, end, intensities,title='The intensity of last imaging')
    plotLines(start, end, intensities*b//time,'The expecting intensity ')
    plotLines(start, end, intensities*a//time,'The expecting intensity after ceiling')

    plotLines(start, end, time, title='The last exposure time')
    plotLines(start, end, a, title='The new exposure time ')
    plotLines(start, end, b, title='The new exposure time after ceiling')

def getExposureTime(start,end,zero,time,intensities,expect=None):

    time_exposure=[]
    for i in range(start-zero,end+1-zero):
        time_exposure.append(expect*time[i]/intensities[i])

    a=np.array(time_exposure.copy())
    b=a.copy()
    # testExpTime(start, end,zero, a, expect, intensities, time)

    a = (a+1).astype(int)
    a[a>500000] = 500000

    testExpTime(start,end,zero,a,b,expect,intensities,time)

    return a

def cal_expTime(start,end,zero,path1,path2,name1,name2,expect):

    intensities = getAvgIntensities(start,end,start,path1)
    times = getTimeConf(path2+name1)
    et = getExposureTime(start,end,zero,times[:,2],intensities,expect)
    #name = path2+str(expect)+'_exposure_time_after_'+name2
    name = path2+name2
    saveTxt(start, end, et, name)


def cal_expTime_area(area,start,end,zero,path1,path2,name1,name2,expect,b=-4):

    intensities = getAvgIntensities_area(area,start,end,start,path1,b)
    times = getTimeConf(path2+name1)
    et = getExposureTime(start,end,zero,times[:,2],intensities,expect)
    name = path2+name2
    saveTxt(start, end, et, name)

def getArea(path,name):

    mat = sio.loadmat(path+name)
    cols = mat['col']
    rows = mat['row']
    qd_square = mat['QD']

    return cols,rows,qd_square


def plotLines(start,end,data,title=''):

    plt.figure()
    plt.plot(range(start, end + 1, 1), data)
    plt.title(title)
    plt.show()

    return 0

if __name__=='__main__':

    # # # cal exposure time config
    # imgs_path = '../data/DingBiao/N.0_201712191159dn170/' # The folder of last imaging
    # expo_path = '../data/DingBiao/'                       # The folder of last exposure time
    # expect = 160                          # The expecting DN value
    # old_expo_name= '160_exposure_time_after_201712191159.txt'  # The last exposure time config file
    # new_expo_name = 'test_final' # The name of new exposure time config file
    # start_band = 390 # start band
    # end_band = 1000 # end band
    # cal_expTime(start_band,end_band,start_band,imgs_path,expo_path,old_expo_name,new_expo_name,expect)


    #
    # # # cal exposure time config
    # imgs_path = '../data/DingBiao/N.1_1(390-500)/' # The folder of last imaging
    # expo_path = '../data/DingBiao/'                # The folder of last exposure time
    # expect = 160                          # The expecting DN value
    # old_expo_name= '160_exposure_time_after_201712191333dn160.txt'  # The last exposure time config file
    # new_expo_name = 'test_final_390_500' # The name of new exposure time config file
    # start_band = 390 # start band
    # end_band = 500 # end band
    # cols, rows, qd_square = getArea(path='../data/DingBiao/data_20171205/', name='coords_calibration.mat')
    # cal_expTime_area(cols,start_band,end_band,start_band,imgs_path,expo_path,old_expo_name,new_expo_name,expect)


    # # # cal exposure time config 201712171023
    # imgs_path = '../data/DingBiao/N.1_390-500_0/' # The folder of last imaging
    # expo_path = '../data/DingBiao/'                # The folder of last exposure time
    # expect = 80                          # The expecting DN value
    # old_expo_name= 'test_final_390_500.txt'  # The last exposure time config file
    # new_expo_name = 'test_final_390_500_201712171023' # The name of new exposure time config file
    # start_band = 390 # start band
    # end_band = 500 # end band
    # cols, rows, qd_square = getArea(path='../data/DingBiao/data_20171205/', name='coords_calibration.mat')
    # cal_expTime_area(cols,start_band,end_band,start_band,imgs_path,
    #                  expo_path,old_expo_name,new_expo_name,expect,b=-6)


    # # # cal exposure time config 201712171023
    # imgs_path = '../data/DingBiao/N1_EXP390_500/' # The folder of last imaging
    # expo_path = '../data/DingBiao/'                # The folder of last exposure time
    # expect = 40                          # The expecting DN value
    # old_expo_name= 'test_final_390_500_201712171023.txt'  # The last exposure time config file
    # new_expo_name = 'test_final_390_500_201712171104' # The name of new exposure time config file
    # start_band = 390 # start band
    # end_band = 500 # end band
    # cols, rows, qd_square = getArea(path='../data/DingBiao/data_20171205/', name='coords_calibration.mat')
    # cal_expTime_area(cols,start_band,end_band,start_band,imgs_path,
    #                  expo_path,old_expo_name,new_expo_name,expect,b=-6)



    # # # cal exposure time config 201712171023
    # imgs_path = '../data/DingBiao/N2_EXP390_500/' # The folder of last imaging
    # expo_path = '../data/DingBiao/'                # The folder of last exposure time
    # expect = 160                          # The expecting DN value
    # old_expo_name= 'test_final_390_500_201712171104.txt'  # The last exposure time config file
    # new_expo_name = 'test_final_390_500_201712171142' # The name of new exposure time config file
    # start_band = 390 # start band
    # end_band = 500 # end band
    # cols, rows, qd_square = getArea(path='../data/DingBiao/data_20171205/', name='coords_calibration.mat')
    # cal_expTime_area(cols,start_band,end_band,start_band,imgs_path,
    #                  expo_path,old_expo_name,new_expo_name,expect,b=-6)


    # # cal exposure time config 201712171023
    imgs_path = '../data/DingBiao/N1_EXP390_500_3/' # The folder of last imaging
    expo_path = '../data/DingBiao/'                # The folder of last exposure time
    expect = 160                          # The expecting DN value
    old_expo_name= 'test_final_390_500_201712171142.txt'  # The last exposure time config file
    new_expo_name = 'test_final_390_500_201712171215' # The name of new exposure time config file
    start_band = 390 # start band
    end_band = 500 # end band
    cols, rows, qd_square = getArea(path='../data/DingBiao/data_20171205/', name='coords_calibration.mat')
    cal_expTime_area(cols,start_band,end_band,start_band,imgs_path,
                     expo_path,old_expo_name,new_expo_name,expect,b=-6)


    print('ok!')