
import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
import scipy.io as sio
import curve_fitting as cf
import copy

COLOR_1 = (255, 0, 0)


def getAvgImg(path):

    files = os.listdir(path)

    imgs=[]
    for fileName in files:
        temp = cv2.imread(path+fileName,cv2.IMREAD_GRAYSCALE)
        imgs.append(temp)

    avg_img = np.array(np.average(imgs, 0), dtype=np.uint8)

    return {'avgImage':[avg_img]}


def getImgList(path=None,interv=None):

    files = os.listdir(path)
    files.sort(key = lambda x:int(x[:-4]))

    imgs=[]
    fileNames=[]
    for i in range(0,len(files),interv):

        f = files[i]
        fileNames.append(f)
        print(f)

        temp = cv2.imread(path+f,cv2.IMREAD_GRAYSCALE)
        imgs.append(temp)

    return {'imgNames':fileNames,'images':imgs}


def getAreaIntensity(images=None,path=None,name=None,mask=None):

    COLOR_1 = (255,0,0)
    COLOR_2 = (0,0,255)

    path = '../data/DingBiao/data_20171205/'
    name='coords_calibration.mat'
    mat = sio.loadmat(path+name)

    for image in images:

        array_img = np.array(image)
        show_img1 = copy.deepcopy(array_img)
        show_img2 = copy.deepcopy(array_img)

        # col
        cols = mat['col']
        avg_cols= []
        for i in range(cols.shape[0]):
            for j in range(cols.shape[1]):
                cv2.rectangle(show_img1, (cols[i,j,0], cols[i,j,1]), (cols[i,j,2], cols[i,j,3]),COLOR_1,2)
                # print(cols[i, j, 0], cols[i, j, 1],cols[i, j, 2], cols[i, j, 3])
                temp = np.average(np.average(array_img[cols[i,j,1]:cols[i,j,3],cols[i,j,0]:cols[i,j,2]],0),0)
                avg_cols.append([i,j,temp])
                # print(avg_cols)

        title='Light distrib follow cols'
        cf.fit_3d(np.array(avg_cols),title)

        plt.figure()
        plt.imshow(show_img1)
        plt.title(title)
        plt.show()

        # row
        rows = mat['row']
        avg_rows = []
        for i in range(rows.shape[0]):
            for j in range(rows.shape[1]):
                cv2.rectangle(show_img2, (rows[i,j,0], rows[i,j,1]), (rows[i,j,2], rows[i,j,3]),COLOR_1,2)
                avg_rows.append([i,j,np.average(np.average(array_img[rows[i,j,1]:rows[i,j,3],rows[i,j,0]:rows[i,j,2]],0),0)])
                # print(avg_rows)

        title='Light distrib follow rows'
        cf.fit_3d(np.array(avg_rows),title)

        plt.figure()
        plt.imshow(show_img2)
        plt.title(title)
        plt.show()

    return avg_cols,avg_rows


def getArea():

    path = '../data/DingBiao/data_20171205/'
    name='coords_calibration.mat'
    mat = sio.loadmat(path+name)

    cols = mat['col']
    rows = mat['row']
    qd_square = mat['QD']

    return cols,rows,qd_square


def getMonochromatorValue(start,end,path=None,name=None):

    path = '../data/DingBiao/data_20171205/'
    name='monochromator.mat'
    mat = sio.loadmat(path+name)
    temp = mat['monochromator']

    monochromator = temp[start-int(temp[0,0]):end+1-int(temp[0,0])]

    return monochromator


def getAvgGRBG(img,area):

    mask = getFilterMask(img)
    square = area
    show_img = copy.deepcopy(img)

    intensity = np.zeros((square.shape[0],square.shape[1],4),dtype=float)

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

            intensity[i,j,:]=[G1,R,B,G2]
    # print('[G1,R,B,G2]',intensity[i,j,:])

    # title = 'Intensity '
    # showImg(show_img,title)
    #
    # cf.fit_3d(cf.img2tuple(intensity[:,:,0]), 'band G1')

    return intensity


def showImg(img,title=''):

    plt.figure()
    plt.imshow(img,cmap=plt.cm.get_cmap('gray'))
    plt.title(title)
    plt.show()

## get the transmission matrix
def getReflect(img,monochromator):

    cols,rows,qd_square = getArea()

    qd_intensity = getAvgGRBG(img,qd_square)
    cols_intensity = getAvgGRBG(img,cols)
    rows_intensity = getAvgGRBG(img,rows)

    without_qd_intensity = np.zeros((qd_intensity.shape[0],qd_intensity.shape[1],4),dtype=float)
    for i in range(qd_intensity.shape[0]):
        for j in range(qd_intensity.shape[1]):
            without_qd_intensity[i,j,:] = np.average((cols_intensity[i,j,:],cols_intensity[i,j+1,:],
                                   rows_intensity[i,j,:],rows_intensity[i+1,j,:]),0)

    C = cf.fit_3d(cf.img2Tuple(without_qd_intensity[:,:,0]), 'without_qd_intensity')

    paras = np.zeros((qd_intensity.shape[0],qd_intensity.shape[1]),dtype=float)
    for i in range(qd_intensity.shape[0]):
        for j in range(qd_intensity.shape[1]):

             Z = C[0] * i + C[1] * j + C[2]
             paras[i,j]=Z/C[2]

    cf.fit_3d(cf.img2Tuple(paras),'Paras test')

    Reflect = np.zeros((qd_intensity.shape[0],qd_intensity.shape[1],4),dtype=float)
    for i in range(qd_intensity.shape[0]):
        for j in range(qd_intensity.shape[1]):
            Reflect[i,j,:]=qd_intensity[i,j,:]/(monochromator*paras[i,j])

    return Reflect

def getAllImgs(path=None): # get all files in folder and children folder

    imgs = []
    files =os.listdir(path)
    for f in files:
        f_path=os.path.join(path,f)

        if os.path.isdir(f_path):
            imgs = imgs+getAllImgs(f_path)
            print(len(imgs))
        else:
            imgs.append(cv2.imread(f_path,cv2.IMREAD_GRAYSCALE))
            print(f_path)

    return imgs

def getT_new():

    imgs = getAllImgs()





def getT(start,end,zero,path):

    files = os.listdir(path)
    files.sort(key=lambda x: int(x[:-4]))

    mono_values = getMonochromatorValue(start,end)

    reflect_list = []
    for i in range(start-zero,end-zero+1):
        f = files[i]
        temp = cv2.imread(path+f,cv2.IMREAD_GRAYSCALE)
        print(f,mono_values[i])

        reflect_list.append(getReflect(temp,mono_values[i,1]))

    return reflect_list


def getALLT():

    data1 = getT(390,450,390,path='../data/DingBiao/data_20171205/200ms_DATA/390-450-200ms/')
    data2 = getT(451,500,451,path='../data/DingBiao/data_20171205/200ms_DATA/451-500-200ms/')
    data3 = getT(501,550,501,path='../data/DingBiao/data_20171205/200ms_DATA/501-550-200ms/')
    data4 = getT(551,600,551,path='../data/DingBiao/data_20171205/200ms_DATA/551-600-200ms/')
    data5 = getT(601,700,601,path='../data/DingBiao/data_20171205/200ms_DATA/601-700-200ms/')
    data6 = getT(701,800,701,path='../data/DingBiao/data_20171205/200ms_DATA/701-800-200ms/')
    data7 = getT(801,900,801,path='../data/DingBiao/data_20171205/200ms_DATA/801-900-200ms/')
    data8 = getT(901,1000,901,path='../data/DingBiao/data_20171205/200ms_DATA/901-1000-200ms/')

    all=np.array((data1+data2+data3+data4+data5+data6+data7+data8))

    sio.savemat('all.mat', {'all': all})

    allT = []
    for k in range(4):
        for i in range(10):
            for j in range(10):
                allT.append(all[:,i,j,k])

    sio.savemat('allT.mat', {'allT': allT})

    plt.figure()
    for i in range(len(allT)):
        plt.plot(range(allT[0]),allT[i])
    plt.show()

    return allT

def getFilterMask(img):

    # make mask
    h, w = img.shape[0], img.shape[1]
    mask = [['G1', 'R'], ['B', 'G2']]
    mask = np.tile(mask, (int(h / 2 + 1), int(w / 2 + 1)))
    mask_final = mask[0:h, 0:w]
    # print(mask_final[0:5, 0:5])

    return mask_final

def plotLine(img,title=''):

    img=np.array(img)

    # ploting col by col
    plt.figure()
    txt = []
    for i in range(np.max(img[:,1]).astype(int)):
        plt.plot(range(len(img[img[:,1]==0][:,2])),img[img[:,1]==i][:,2])
        txt.append('col_{}'.format(i))

    plt.legend(tuple(txt),loc='upper right')
    plt.title(title)
    # plt.show()

    # ploting row by row
    plt.figure()
    txt = []
    for i in range(np.max(img[:,0]).astype(int)):
        plt.plot(range(len(img[img[:,0]==0][:,2])),img[img[:,0]==i][:,2])
        txt.append('row_{}'.format(i))

    plt.legend(tuple(txt),loc='upper right')
    plt.title(title)
    plt.show()

if __name__=='__main__':

    # The 2nd data with qd
    # path = '../data/DingBiao/data_20171205/Calibration/800nm/300ms/'
    # avgImg = getAvgImg(path)
    # avg_cols, avg_rows = getAreaIntensity(avgImg['avgImage'])
    # plotLine(avg_cols)
    # plotLine(avg_rows)

    # # test the image by sampling
    # avgImg = getAvgImg(path)
    # img = cf.img2tuple(avgImg['avgImage'][0])
    # cf.fit_3d(img)

    ## test one by one
    # imgs = getImgList(path,3)
    # getAreaIntensity(imgs['images'])

    # # The 1nd data without qd
    # path = '../data/DingBiao/100DS_first_dingbiao/WithoutQD/390-500/'
    # # path = '../data/DingBiao/100DS_first_dingbiao/WithoutQD/742-900/'
    # imgs = getImgList(path,10)
    # getAreaIntensity(imgs['images'])

    # The 1nd data with qd
    path = '../data/DingBiao/100DS_first_dingbiao/WithQD/390-500/'
    imgs = getImgList(path,10)
    getAreaIntensity(imgs['images'])



    ####----------------------------------------------transmission matrix-----------------------------

    # path = '../data/DingBiao/data_20171205/Calibration/500nm/320ms/'
    # avgImg = getAvgImg(path)
    # # getAvgGBRG(avgImg['avgImage'][0])

    T = getALLT()
