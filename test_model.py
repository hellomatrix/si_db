import light_distribution as ld
import curve_fitting as cf
import numpy as np


def getTestImgs():

    testImgs=[]
    h=1024
    w=1280
    temp = np.zeros((h,w))

    # uniform value
    testImgs.append(np.ones((h,w)))

    # up follow row
    for i in range(h):
        temp[i,:]=i
    testImgs.append((temp-np.min(range(h)))/(np.max(range(h)) - np.min(range(h)))*255)

    # down follow col
    for i in range(w):
        temp[:,i]=w-i-1
    testImgs.append((temp-np.min(range(w)))/(np.max(range(w)) - np.min(range(w)))*255)

    return testImgs



if __name__=='__main__':

    # imgs = getTestImgs()
    # ld.getAreaIntensity(imgs)
    #
    # imgs = getTestImgs()
    # img = cf.img2Tuple(imgs[1])
    # cf.fit_3d(img,'')

    # path='../data/DingBiao/data_20171205/200ms_DATA/'
    # imgs = ld.getAllImgs(path)
    #
    # avg_img= np.average(imgs,0)
    # ld.showImg(avg_img)
    #
    # ld.getAreaIntensity([avg_img])

    path = '../data/DingBiao/data_20171205/Calibration/800nm/300ms/'
    avgImg = ld.getAvgImg(path)
    allimgs= np.uint8(np.average(ld.getAllImgs(path),0))

    print(avgImg['avgImage']-allimgs)

