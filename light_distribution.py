
import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
import scipy.io as sio
import curve_fitting as cf
import copy
from matplotlib.backends.backend_pdf import PdfPages
import xlrd

COLOR_1 = (255, 0, 0)




def getImgsOneFolder(path,a=0,b=-4):

    files = os.listdir(path)
    files.sort(key = lambda x:int(x[a:b]))

    print(path)

    imgs=[]
    for fileName in files:
        f = cv2.imread(path+fileName,cv2.IMREAD_GRAYSCALE)
        imgs.append(f)

        print(fileName)

    return imgs


def getImgsOneFolderSplit(path):

    files = os.listdir(path)
    files.sort(key = lambda x:(int(x.split('_')[0]),int(x.split('_')[-1].split('.')[0])))

    print(path)

    imgs=[]
    f_names=[]
    for fileName in files:
        f = cv2.imread(path+fileName,cv2.IMREAD_GRAYSCALE)

        imgs.append(f)
        f_names.append(fileName)

        print(fileName)

    return f_names,imgs


def getAvgImg(imgs):

     print(len(imgs))
     return np.average(imgs,0)


def showImgArea(img,area):

    show_img = copy.deepcopy(img)
    for i in range(area.shape[0]):
        for j in range(area.shape[1]):
            cv2.rectangle(show_img, (area[i,j,0], area[i,j,1]), (area[i,j,2], area[i,j,3]),COLOR_1,2)
            print(area[i, j, 0], area[i, j, 1],area[i, j, 2], area[i, j, 3])


    plt.figure()
    plt.imshow(show_img,cmap=plt.cm.get_cmap('gray'))
    plt.title('')
    plt.show()

def getArea(path,name):

    mat = sio.loadmat(path+name)
    cols = mat['col']
    rows = mat['row']
    qd_square = mat['QD']

    return cols,rows,qd_square


def getMixGRBG(col,row,show=0):

    uper = min(col.shape[0],col.shape[1])

    mixGRBG = np.zeros((uper,uper,4))

    for i in range(uper):
        for j in range(uper):
            mixGRBG[i,j,:] = np.average((col[i,j,:],col[i,j+1,:],row[i,j,:],row[i+1,j,:]),0)

    if show == 1:
        cf.fit_3d(cf.img2Tuple(mixGRBG[:,:,0]), 'getMixGRBG:band G1')
        cf.fit_3d(cf.img2Tuple(mixGRBG[:,:,1]), 'getMixGRBG:band R')
        cf.fit_3d(cf.img2Tuple(mixGRBG[:,:,2]), 'getMixGRBG:band B')
        cf.fit_3d(cf.img2Tuple(mixGRBG[:,:,3]), 'getMixGRBG:band G2')

    return mixGRBG


def getMonochromatorValue(start,end,path=None,name=None):

    path = '../data/DingBiao/data_20171205/'
    name='monochromator.mat'
    mat = sio.loadmat(path+name)
    temp = mat['monochromator']

    monochromator = temp[start-int(temp[0,0]):end+1-int(temp[0,0])]

    return monochromator


def getTheBiggestBand(G1RBG2):

    top = np.max(G1RBG2)
    band=G1RBG2.index(top)
    if band == 3:
        band=0

    return band,top


# def checkBand(GRBG):
#
#     if (np.sum(GRBG - GRBG[0]) == 0):
#         # print('This band is good!')
#         k = cf.fit_3d(cf.img2Tuple(intensities[:, :, bands[0]]),title=str(bands[0]))
#
#     else:
#         print('This band is bad! band:',str(lamb))
#
#     return k


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

    # print('[G1,R,B,G2]',intensity[i,j,:])

    if show==1:
        showGrayImg(show_img,'')

        cf.fit_3d(cf.img2Tuple(intensity[:,:,0]), 'getAvgGRBG:band G1')
        cf.fit_3d(cf.img2Tuple(intensity[:,:,1]), 'getAvgGRBG:band R')
        cf.fit_3d(cf.img2Tuple(intensity[:,:,2]), 'getAvgGRBG:band B')
        cf.fit_3d(cf.img2Tuple(intensity[:,:,3]), 'getAvgGRBG:band G2')

    return intensity,bands,vals


def showGrayImg(img,title=''):

    show_img = copy.deepcopy(img)
    plt.figure()
    plt.imshow(show_img,cmap=plt.cm.get_cmap('gray'))
    plt.title(title)
    plt.show()

def getAllImgs(path=None): # get all files in folder and children folder

    imgs = []
    files =os.listdir(path)
    f_test = os.path.join(path, files[0])
    if os.path.isdir(f_test):
        files.sort(key=lambda x: int(x[:-11]))
    else:
        files.sort(key=lambda x: int(x[:-4]))

    for f in files:
        f_path=os.path.join(path,f)

        if os.path.isdir(f_path):
            imgs = imgs+getAllImgs(f_path)
            print(len(imgs))
        else:
            imgs.append(cv2.imread(f_path,cv2.IMREAD_GRAYSCALE))
            print(f_path)

    return imgs

#
# def getK(intensities,bands,lamb=None):
#
#     if(np.sum(bands-bands[0])==0):
#         # print('This band is good!')
#         k = cf.fit_3d(cf.img2Tuple(intensities[:, :, bands[0]]),title=str(bands[0]))
#
#     else:
#         print('This band is bad! band:',str(lamb))
#
#     return k

def getK(lightZone,band):

    k = cf.fit_3d(cf.img2Tuple(lightZone[:, :, band]),title=str(band))

    return k


def getOneBandT(exp_time,fore_ground,K,back_ground=None):

    T = np.zeros((fore_ground.shape[0],fore_ground.shape[1],4),dtype=float)
    for i in range(fore_ground.shape[0]):
        for j in range(fore_ground.shape[1]):

            # Z = K[0] * i + K[1] * j + K[2]
            # kk = Z / K[2]


            # best-fit quadratic curve
            Z=K
            kk = Z[i,j]/Z[0, 0]

            T[i, j, :] = fore_ground[i, j, :] / (back_ground * kk*exp_time)

    return T


def getT(monos,exp_time,test_no,pp,imgs,cols, rows, qd_square,mask,K=None):


    #mono_values = getMonochromatorValue(390, 1000)
    mono_values = np.array(monos).T


    T_all = []
    for i,im in zip(range(len(imgs)),imgs):

        avg_cols = getAvgGRBG(im, cols, mask)
        avg_rows = getAvgGRBG(im, rows, mask)
        mix_val = getMixGRBG(col=avg_cols[0], row=avg_rows[0],show=1)

        band= avg_rows[1][0]
        K = getK(mix_val,band)

        fore_ground = getAvgGRBG(im, qd_square, mask)

        T_all.append(getOneBandT(exp_time[i],fore_ground[0],K,back_ground=mono_values[i,1]))

    T_all = np.array(T_all)

    T_final=[]

    for i in range(T_all.shape[1]):
        for j in range(T_all.shape[2]):
            savPdf(mono_values[:,1],pp,T_all[:,i,j,:], list(range(390,1001)),i+j,label ='')
            for k in range(T_all.shape[3]):
                T_final.append(T_all[:,i,j,k])

    sio.savemat(test_no+'.mat', {'T': T_final})
    plt.figure()
    for m in range(len(T_final)):
        plt.plot(range(390,1001), T_final[m])

    plt.legend('', loc='upper right')
    plt.title('')
    plt.show()


def getFilterMask(img):

    # make mask
    h, w = img.shape[0], img.shape[1]
    mask = [['G1', 'R'], ['B', 'G2']]
    mask = np.tile(mask, (int(h / 2 + 1), int(w / 2 + 1)))
    mask_final = mask[0:h, 0:w]
    # print(mask_final[0:5, 0:5])

    return mask_final


def showGRBG(imgs,area,mask):

    bands_all=[]
    vals_all=[]
    for im in imgs:
        _,bands,vals = getAvgGRBG(im, area, mask)
        bands_all.append(np.average(bands,0))
        vals_all.append(np.average(vals,0))


    plotLine(list(range(390,1001,1)), bands_all, title='')
    plotLine(list(range(390,1001,1)), vals_all, title='')

    return bands_all,vals_all


def plotLine(range,data,title=''):

    plt.figure()
    plt.plot(range,data)
    plt.legend('',loc='upper right')
    plt.title(title)
    plt.show()

def saveMat(fnames,imgs,sensor):

    sio.savemat(sensor+'.mat',{'fnames':fnames,'imgs':imgs})

def getData(path,sensor):

    name = sensor+'.mat'
    if os.path.exists(name):

        imgs = sio.loadmat(name)['imgs']
        fnames = sio.loadmat(name)['fnames']
        print(fnames)

    else:

        fnames, imgs = getImgsOneFolderSplit(path)
        saveMat(fnames, imgs,sensor)

    return imgs,fnames

def readExcel(path):
    data = xlrd.open_workbook(path,encoding_override='utf-8')
    tabel = data.sheets()[0]

    monos = [tabel.col_values(0), tabel.col_values(1)]

    return monos



def getAvgData(path,sensor):

    name = sensor+'.mat'
    if os.path.exists(name):

        imgs_post = sio.loadmat(name)['imgs']
        fnames_post = sio.loadmat(name)['fnames']
        print(fnames_post)

    else:

        fnames, imgs = getImgsOneFolderSplit(path)

        imgs_post = []
        fnames_post = []
        for i in range(0, len(imgs), 3):
            imgs_post.append(np.average(imgs[i:i + 3], 0))
            fnames_post.append(fnames[i])

        imgs_post = np.array(imgs_post,dtype=np.uint8)
        saveMat(fnames_post, imgs_post,sensor)

    return imgs_post,fnames_post


def savPdf(light,pp, y, x,qd_indx,label =''):

    # fmt = "L1_%d_n%d_me%d_mf%d.pdf"
    # nr = noise_ratio * 100
    # me = measure_error * 100
    # mf = measure_off * 100
    # pp = PdfPages(fmt % (spec, nr, me, mf))
    cor = ['g','r','b','y']
    lab=['G1','R','B','G2']

    for i in range(y.shape[1]):
        plt.plot(x, y[:,i], color=cor[i],label=lab[i])
    plt.plot(x, light/np.max(light), color='k', label='light')
    plt.xlabel('Wavelength [nm]')
    plt.ylabel('Transmission')
    plt.title('QD:'+str(qd_indx))
    plt.legend()
        # plt.text(600, 0.9, label)
    pp.savefig()
    plt.clf()


def getTimeConf(fileName):

    time = np.loadtxt(fileName)
    return time


if __name__=='__main__':

    ####----------------------------------------------V2----------------------------------------------


    # # test1

    # path = '../data/DingBiao/N.1_dingbiao/390-1000nm/'
    # imgs =getImgsOneFolder(path)
    # avg_img = getAvgImg(imgs)
    # cols, rows, qd_square = getArea(path = '../data/DingBiao/data_20171205/',name='coords_calibration.mat')
    # # showImgArea(avg_img, cols)
    #
    # mask = getFilterMask(avg_img)
    # avg_cols = getAvgGRBG(avg_img, cols, mask)
    # avg_rows = getAvgGRBG(avg_img, rows, mask)
    #
    # mix_val = getMixGRBG(col=avg_cols,row=avg_rows)

    # # test2
    # sensor ='N.1_DN170_20171218_CALI_DATA'
    # path = '../data/DingBiao/'+sensor+'/DATA/'
    # imgs,f_names = getAvgData(path,sensor)
    #
    # avg_img = getAvgImg(imgs)
    # cols, rows, qd_square = getArea(path = '../data/DingBiao/data_20171205/',name='coords_calibration.mat')
    # # showImgArea(avg_img, cols)
    #
    # mask = getFilterMask(avg_img)
    # showGRBG(imgs, cols, mask)
    #
    # avg_cols = getAvgGRBG(avg_img, cols, mask)
    # avg_rows = getAvgGRBG(avg_img, rows, mask)
    #
    # mix_val = getMixGRBG(col=avg_cols[0],row=avg_rows[0])


#####---------------------------------------v3----------------------------------------------

    # # test3
    # test_no ='N.1_DN160_20171219_CALI_DATA'
    # path = '../data/DingBiao/'+test_no+'/DATA/'
    # imgs = getImgsOneFolder(path)
    # cols, rows, qd_square = getArea(path = '../data/DingBiao/data_20171205/',name='coords_calibration.mat')
    # mask = getFilterMask(imgs[0])
    # exp_time_conf=getTimeConf('160_exposure_time_after_201712191333dn160.txt')
    # #exp_times=exp_time_conf[:,2]/np.max(exp_time_conf[:,2])
    # exp_times=exp_time_conf[:,2]
    # fmt = test_no+'.pdf'
    # pp = PdfPages(fmt)
    # getT(exp_times,test_no,pp,imgs,cols, rows, qd_square,mask)
    # pp.close()

    # # # test4
    # monos = readExcel('../data/DingBiao/N1/20171220_1.xlsx')
    # test_no ='N.1_DN160_20171219_CALI_DATA_3times'
    # path1 = '../data/DingBiao/N1/N.1_DN160_20171219_CALI_DATA/'
    # path2 = '../data/DingBiao/N1/N.1_DN160_20171219_CALI_DATA_2/'
    # path3 = '../data/DingBiao/N1/N.1_DN160_20171219_CALI_DATA_3/'
    # imgs1 = np.array(getImgsOneFolder(path1),dtype=np.uint16)
    # imgs2 = np.array(getImgsOneFolder(path2),dtype=np.uint16)
    # imgs3 = np.array(getImgsOneFolder(path3),dtype=np.uint16)
    # imgs=[]
    # for i in range(len(imgs1)):
    #     imgs.append((imgs1[i]+imgs2[i]+imgs3[i])/3)
    # imgs=np.array(imgs,dtype=np.uint8)
    # cols, rows, qd_square = getArea(path = '../data/DingBiao/data_20171205/',name='coords_calibration.mat')
    # mask = getFilterMask(imgs[0])
    # exp_time_conf=getTimeConf('160_exposure_time_after_201712191333dn160.txt')
    # exp_times=exp_time_conf[:,2]/np.max(exp_time_conf[:,2])
    # fmt = test_no+'.pdf'
    # pp = PdfPages(fmt)
    # getT(monos,exp_times,test_no,pp,imgs,cols, rows, qd_square,mask)
    # pp.close()

    # # test5
    monos = readExcel('../data/DingBiao/N1/20171220_1.xlsx')
    test_no ='N.2_DN160_20171220_CALI_DATA_3times'
    path1 = '../data/DingBiao/N2/N.2_DN160_20171220_CALI_DATA_1/DATA/'
    path2 = '../data/DingBiao/N2/N.2_DN160_20171220_CALI_DATA_2/DATA/'
    path3 = '../data/DingBiao/N2/N.2_DN160_20171220_CALI_DATA_3/DATA/'
    imgs1 = np.array(getImgsOneFolder(path1),dtype=np.uint16)
    imgs2 = np.array(getImgsOneFolder(path2),dtype=np.uint16)
    imgs3 = np.array(getImgsOneFolder(path3),dtype=np.uint16)
    imgs=[]
    for i in range(len(imgs1)):
        imgs.append((imgs1[i]+imgs2[i]+imgs3[i])/3)
    imgs=np.array(imgs,dtype=np.uint8)
    cols, rows, qd_square = getArea(path = '../data/DingBiao/data_20171205/',name='coords_calibration.mat')
    mask = getFilterMask(imgs[0])
    exp_time_conf=getTimeConf('160_exposure_time_after_201712191333dn160.txt')
    exp_times=exp_time_conf[:,2]/np.max(exp_time_conf[:,2])
    fmt = test_no+'.pdf'
    pp = PdfPages(fmt)
    getT(monos,exp_times,test_no,pp,imgs,cols, rows, qd_square,mask)
    pp.close()