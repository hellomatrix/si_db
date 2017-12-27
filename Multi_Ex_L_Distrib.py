import light_distribution as ld
import numpy as np
import matplotlib.pyplot as plt
import copy


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


# path = '../data/DingBiao/201712131040/'
# imgs =ld.getImgsOneFolderSplit(path)

# # # test the average image of all
# avg_img = ld.getAvgImg(imgs)
# cols, rows, qd_square = ld.getArea(path = '../data/DingBiao/data_20171205/',name='coords_calibration.mat')
# mask = ld.getFilterMask(avg_img)
# avg_cols = ld.getAvgGRBG(avg_img, cols, mask)
# avg_rows = ld.getAvgGRBG(avg_img, rows, mask)
# mix_val = ld.getMixGRBG(col=avg_cols,row=avg_rows)


# # # test avg by band
# path = '../data/DingBiao/201712131040/'
# imgs =ld.getImgsOneFolderSplit(path)
# for i in range(600,len(imgs),30):
#     avg_img = ld.getAvgImg(imgs[i:i+30])
#     cols, rows, qd_square = ld.getArea(path = '../data/DingBiao/data_20171205/',name='coords_calibration.mat')
#     mask = ld.getFilterMask(avg_img)
#     avg_cols = ld.getAvgGRBG(avg_img, cols, mask)
#     avg_rows = ld.getAvgGRBG(avg_img, rows, mask)
#     mix_val = ld.getMixGRBG(col=avg_cols,row=avg_rows)


# # # test avg by band
path = '../data/DingBiao/201712131040/'
imgs = ld.getImgsOneFolderSplit(path)
aa=[]
kk=[]
start = 390

fac1= []
for m in range(start,1000,10):
    d=(m-390)//10
    arr_imgs = np.array(imgs)
    a = np.average(np.average(arr_imgs[d*30:d*30 + 30,:,:],1),1)

    index=30
    b = copy.deepcopy(a[8:30])
    bbb = copy.deepcopy((100/b[0])*b)

    if max(a) > 200:
        index_list = np.where(a > 200)
        index = index_list[0][0]
        print('too much exp:',(index,a[index]))
        print('band:',d)
        b = copy.deepcopy(a[0:index])
        bbb = copy.deepcopy((100/b[8])*b)

    fac1.append(a[7])

    # b = (100/b[7])*b

    x = list(range(len(b)))
    y = bbb

    c,k = best_fit(x,y,m)
    aa.append(c)
    kk.append(k)

plt.figure()
plt.plot(range(start,1000,10),aa)
plt.figure()
plt.plot(range(start,1000,10),kk)

plt.figure()
plt.plot(range(len(fac1)),fac1)

plt.show()



# for j in range(start,1000,10):
#     d=(j-390)//10
#     # u=len(imgs)//30
#     # for i in range(d,u):
#     arr_imgs = np.array(imgs)
#
#     a = np.average(np.average(arr_imgs[d*30:d*30 + 30,600:605,600:605],1),1)
#
#     x = list(range(0,30))
#     y = a
#     # y=(a-np.min(a))/(np.max(a)-np.min(a)) # normalization
#
#     a,k = best_fit(x,y)
#
#     aa.append(a)
#     kk.append(k)
#
# plt.figure()
# plt.plot(range(start,1000,10),aa)
# plt.figure()
# plt.plot(range(start,1000,10),kk)
#
# plt.show()


# # # test on dataset 2
# path = '../data/DingBiao/N.2_20171223/CaliData/'
# imgs =ld.getImgsOneFolderSplit(path)
# avg_imgs=[]
# for i in range(0,len(imgs),3):
#     avg_imgs.append(ld.getAvgImg(imgs[i:i+3]))
#
# for j in range(0,len(imgs)):
#     avg_img = avg_imgs[j]
#     cols, rows, qd_square = ld.getArea(path = '../data/DingBiao/data_20171205/',name='coords_calibration.mat')
#     mask = ld.getFilterMask(avg_img)
#     avg_cols = ld.getAvgGRBG(avg_img, cols, mask)
#     avg_rows = ld.getAvgGRBG(avg_img, rows, mask)
#     mix_val = ld.getMixGRBG(col=avg_cols,row=avg_rows)