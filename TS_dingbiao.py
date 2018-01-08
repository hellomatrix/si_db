import numpy as np
import os
import matplotlib.pyplot as plt
import cv2
import scipy.io as sio
import copy

COLOR_1 = (0,255, 0)

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


def getImgsOneFolderSplit(path,lower=0,uper=100):

    files = os.listdir(path)
    files.sort(key = lambda x:(int(x.split('_')[0]),int(x.split('_')[-1].split('.')[0])))

    print(path)

    imgs=[]
    f_names=[]
    for i,fileName in enumerate(files):
        if i==uper:
            break
        elif i>lower:
            f = cv2.imread(path+fileName,cv2.IMREAD_GRAYSCALE)

            imgs.append(f)
            f_names.append(fileName)

            print(fileName)



    return f_names,imgs

def getRaw2Imgs_OneFolder(path,path_dis,h=2162,w=2560,a=0,b=-4):

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


def getRaw2Imgs_OneFolderSplit(path,path_dis,h=2162,w=2560):

    files = os.listdir(path)
    files.sort(key = lambda x:(int(x.split('_')[0]),int(x.split('_')[-1].split('.')[0])))
    print(path)

    if os.path.exists(path_dis):
        return getImgsOneFolderSplit(path_dis)
    else:
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

        showImg(img)
        # saveImg(img,fname,path_dis)

    # sio.savemat(path_dis+'imgs.mat', {'imgs': imgs})
    return imgs


def showImg(img):

    # img = img/65535 * 255

    print(img.max())
    plt.figure()
    plt.imshow(img)
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

def dif_s1from_s2(set1,set2):

    if not list(set2):
        suplementary = set1
    elif not list(set1):
        suplementary = tuple([])
        print('this img is empty')
    else:
        a = np.reshape(np.array(set1).T,(len(set1[0]),-1))
        b = np.reshape(np.array(set2).T,(len(set2[0]),-1))

        c = [val for val in a if val not in b]
        suplementary = tuple(np.array(c).T)

    return suplementary

def dif(rois):

    max_y = len(rois)
    max_x = 0
    for i in range(max_y):
        if len(rois[i])>max_x:
            max_x = copy.copy(len(rois[i]))

    rois_final = []
    sets_final = []
    for i in range(max_y+2):
        rois_final.append([])
        sets_final.append([])
        for j in range(max_x+2):
            rois_final[i].append([])
            sets_final[i].append([])

    for i in range(max_y):
        for j in range(len(rois[i])):
            rois_final[i+1][j+1]=rois[i][j]

    for i in range(1,max_y+1):
        for j in range(1,len(rois[i])+1):
            for m in range(i-1,i+2):
                for n in range(j-1,j+2):

                    if m*n!=1:
                        rois_final[i][j] = dif_s1from_s2(rois_final[i][j],rois_final[m][n])
                        print((i, j, m, n))

            sets_final[i][j] = copy.copy(rois_final[i][j])
            print(i,j,rois_final[i][j])

    return sets_final

def getValue(sets,imgs):

    img_final = np.zeros_like(imgs[0],dtype=np.uint16)
    k=0
    for i in range(len(sets)):
        for j in range(sets[i]):
            if (not sets[i][j])==False:
                img_final = img_final+imgs[k][sets[i][j]]
                k+=1




# def dif8(rois,x_end,y_end):
#
#     y_end = len(rois)
#
#     for m in range(len(rois)):
#         for n in range(len(rois[m])):
#         # temp = roi[m][n]
#
#         if m-1>=0:
#             h_start = m-1
#         else:
#             h_start = m
#
#         if m + 1 >= y_end:
#             h_end = m + 1
#         else:
#             h_end = temp[0]
#
#         if temp[1] - 1 >= 0:
#             w_start = temp[1] - 1
#         else:
#             w_start = temp[1]
#
#         if temp[1] + 1 >= x_end:
#             w_end = temp[1] + 1
#         else:
#             w_end = temp[1]
#
#         for i in range(h_start,h_end):
#             for j in range(w_start,w_end):
#
#                 # dif_s1from_s2(temp,roi[i,j])


if __name__=='__main__':


    fpath = '../data/TS_dingbiao/'
    band_name = 'J_499nm'
    path_dis = '../data/TS_dingbiao/'+band_name+'_imgs/'
    # imgs = getRaw2Imgs_OneFolderSplit(fpath+band_name+'/',path_dis)


    _,imgs = getImgsOneFolderSplit(path_dis,0,50)

    roi = []
    i,j= -1,0
    max_x=0
    for k in range(len(imgs)):

         avgimg = imgs[k]
         # showImg(avgimg/ 65535 * 255)
         thresh1 = cv2.equalizeHist(avgimg[1:-1,:])
         ret, thresh1 = cv2.threshold(thresh1, 240, 255, cv2.THRESH_BINARY)
         thresh1 = cv2.medianBlur(thresh1, 7)
         ret, thresh1 = cv2.threshold(thresh1, 250, 255, cv2.THRESH_BINARY)



         # showImg(thresh1)

         # # # opt
         # # kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
         # kernel = np.uint8(np.ones((5, 1)))
         # opened = cv2.morphologyEx(thresh1, cv2.MORPH_OPEN, kernel,iterations = 3)
         # closed = cv2.morphologyEx(thresh1, cv2.MORPH_CLOSE, kernel)
         # showImg(thresh1)

         set = np.where(thresh1 > 0)
         # if k==25:
         #     print()

         if not list(set[1]):
             x_max=0
         else:
            x_max = np.max(set[1])

         if x_max >= 2550:
             i+=1
             j=0
             roi.append([])
             if i>0 and len(roi[i-1])<5:
                 i-=1
                 roi.pop()

         print(i,x_max,k)

         roi[i].append(set)

    dif(roi)

    sio.savemat('roi.mat',{'rois':roi})

    #find difference between sets


         # img,contours, hierarchy = cv2.findContours(thresh1, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
         # roi.append(np.where(thresh1>0))

         # x_low = np.min(contours[3][:1,0,0])
         # x_up = np.max(contours[3][:,0,0])
         # y_low = np.min(contours[3][:,0,1])
         # y_up = np.max(contours[3][:,0,1])

         # img = cv2.cvtColor(img,cv2.COLOR_GRAY2BGR)
         # cv2.drawContours(img, contours, -1, (0, 0, 255), 3)
         # showImg(img)

         # cv2.imshow("img", img)
         # cv2.waitKey(0)


    print('ok')


    # fpath = '../data/TS_dingbiao/'
    # band_name = '499nm'
    # path_dis = '../data/TS_dingbiao/'+band_name+'_imgs/'
    # getImgsOneFolder(fpath+band_name+'/',path_dis, h=2162, w=2560, a=0, b=-10)
