import light_distribution as ld
import numpy as np
import matplotlib.pyplot as plt
import copy
from matplotlib.backends.backend_pdf import PdfPages
import scipy.io as sio
import os


# linear fitting
def best_fit(X, Y,m):

    xbar = sum(X)/len(X)
    ybar = sum(Y)/len(Y)
    n = len(X) # or len(Y)

    numer = sum([xi*yi for xi,yi in zip(X, Y)]) - n * xbar * ybar
    denum = sum([xi**2 for xi in X]) - n * xbar**2

    b = numer / denum
    a = ybar - b * xbar

    print('best fit line:\ny = {:.2f} + {:.2f}x'.format(a, b))

    if(m%6==0):
        plt.figure()
        plt.scatter(X, Y)
        yfit = [a + b * xi for xi in X]
        plt.plot(X, yfit)

    return a, b


def savPdf(pp, y, x, label = ''):

    # fmt = "L1_%d_n%d_me%d_mf%d.pdf"
    # nr = noise_ratio * 100
    # me = measure_error * 100
    # mf = measure_off * 100
    # pp = PdfPages(fmt % (spec, nr, me, mf))

    plt.plot(x, y, color='r')
    plt.xlabel('Wavelength [nm]')
    plt.ylabel('Intensity')
    plt.title('relative error: %.2f' % (err * 100) + '%')
    plt.text(600, 0.9, label)
    pp.savefig()
    plt.clf()

def saveMat(fnames,imgs):

    sio.savemat('linearTest.mat',{'fnames':fnames,'imgs':imgs})


def plotrgb(wave,data):

    plt.figure()
    plt.plot(wave,data)
    plt.show()

def partFit(data,l,u):

    data = np.array(data)
    y = copy.deepcopy(data[:,l:u])
    aa=[]
    kk=[]
    xx=list(range(l,u))
    for i in range(y.shape[0]):

        yy = 100/y[i,0]*y[i,:]

        b,k = best_fit(xx,yy,m=i)
        aa.append(b)
        kk.append(k)

    print('SD:',(np.std(kk)))
    print('Error:',(np.max(kk)-np.min(kk))/np.mean(kk))

    plt.figure()
    plt.plot(range(0,y.shape[0]),aa)
    plt.figure()
    plt.plot(range(0,y.shape[0]),kk)

def findGoodRange(data,mask,zero = 400,start = 690,end = 750,uper=200,lower=50):

    # imgs = copy.deepcopy(data)
    imgs = data

    rgb=[]
    index_uper_lower=[]
    arr_imgs = np.array(imgs)
    avg_channel_all=[]
    for m in range(start,end+10,10):
        d=(m-zero)//10

        lowest = 0
        upest = 30

        y1=500
        y2=510
        x1=500
        x2=510

        anylise_zone = arr_imgs[d * 30:d * 30 + 30, y1:y2, x1:x2]

        mask_temp = mask[y1:y2, x1:x2]
        avg_channel = []

        for i in range(len(anylise_zone)):
            im = anylise_zone[i]
            avg_channel.append([np.average(im[mask_temp=='G1']),np.average(im[mask_temp=='R']),
                               np.average(im[mask_temp == 'B']),np.average(im[mask_temp=='G2'])])

        temp = np.average(avg_channel,0)
        temp = temp.tolist()

        top = temp.index(np.max(temp))
        rgb.append(top)

        band_idx = np.array(avg_channel)[:,top]
        avg_channel_all.append(band_idx)

        index_uper_lower.append([np.where(np.array(band_idx)>50)[0][0],np.where(np.array(band_idx)<200)[0][-1]])

        print('band, lower and uper',(m,np.where(np.array(band_idx)>50)[0][0],np.where(np.array(band_idx)<200)[0][-1]))

    l=np.max(np.array(index_uper_lower)[:,0])
    u=np.min(np.array(index_uper_lower)[:,1])
    print('lowest and upest', (l,u))

    plotrgb(list(range(start,end+10,10)),rgb)

    partFit(avg_channel_all, l, u)

    plt.show()

    return l,u,avg_channel_all


def getData(path):

    if os.path.exists('linearTest.mat'):

        imgs = sio.loadmat('linearTest.mat')['imgs']
        fnames = sio.loadmat('linearTest.mat')['fnames']
        print(fnames)

    else:

        fnames, imgs = ld.getImgsOneFolderSplit(path)
        saveMat(fnames, imgs)

    return imgs,fnames

# def linerTest(mask,start = 400,data=None):
#
#     imgs = data
#
#     aa=[]
#     kk=[]
#     fac1= []
#     avg_channel = []
#     for m in range(start,1000,10):
#         d=(m-390)//10
#         arr_imgs = np.array(imgs)
#
#
#         y1=500
#         y2=510
#         x1=500
#         x2=510
#         anylise_zone = arr_imgs[d * 30:d * 30 + 30, y1:y2, x1:x2]
#         mask_temp = mask[y1:y2, x1:x2]
#
#
#         for i in range(len(anylise_zone)):
#             im = anylise_zone[i]
#             avg_channel.append([np.average(im[mask_temp=='G1']),np.average(im[mask_temp=='R']),
#                                np.average(im[mask_temp == 'B']),np.average(im[mask_temp=='G2'])])
#
#         index=30
#
#         fac1.append(avg_channel[7,1])
#         a=avg_channel[d * 30:d * 30 + 30]
#
#
#         #
#         #
#         # # b = (100/b[7])*b
#         #
#         # x = list(range(len(b)))
#         # y = bbb
#         #
#         # c,k = best_fit(x,y,m)
#         # aa.append(c)
#         # kk.append(k)
#
#     plt.figure()
#     plt.plot(range(start,1000,10),aa)
#     plt.figure()
#     plt.plot(range(start,1000,10),kk)
#
#     plt.figure()
#     plt.plot(range(len(fac1)),fac1)
#
#     plt.show()


if __name__=='__main__':

    path = '../data/DingBiao/201712131040/'
    # linerTest(path)

    imgs,fnames=getData(path)
    mask=ld.getFilterMask(imgs[0])
    findGoodRange(imgs, mask)