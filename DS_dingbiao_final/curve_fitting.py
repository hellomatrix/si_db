#!/usr/bin/evn python

import numpy as np
import scipy.linalg
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt


def leastsq(x,y):

    x=np.array(x)
    y=np.array(y)
    A=np.vstack([x,np.ones(len(x))]).T
    m,c = np.linalg.lstsq(A,y)[0]

    plt.plot(x,y,'o',label='Original data',markersize=10)
    plt.plot(x,m*x+c,'r',label='Fitted line')
    plt.legend()
    plt.show()

    # cal error


# # test the image by sampling
def img2Tuple(img):

    img=np.array(img)

    imgTuple=[]
    if(img.shape[0]>20):
        m = 0
        for i in range(0,img.shape[0],img.shape[0]//10):
            n = 0
            for j in range(0,img.shape[1],img.shape[1]//10):
                imgTuple.append(np.array([m, n, img[i+1, j]]))
                n+=1
            m += 1
    else:
        for i in range(img.shape[0]):
            for j in range(img.shape[1]):
                imgTuple.append(np.array([i,j,img[i,j]]))

    return np.array(imgTuple)


def fit_3d(data,title=''):

    x_min=np.min(data[:,0])
    x_max=np.max(data[:,0])

    y_min=np.min(data[:,1])
    y_max=np.max(data[:,1])

    # regular grid covering the domain of the data
    X, Y = np.meshgrid(np.arange(x_min, x_max+1, 1), np.arange(y_min, y_max+1, 1))
    XX = X.flatten()
    YY = Y.flatten()

    order = 2  # 1: linear, 2: quadratic
    if order == 1:
        # best-fit linear plane
        A = np.c_[data[:, 0], data[:, 1], np.ones(data.shape[0])]
        C, _, _, _ = scipy.linalg.lstsq(A, data[:, 2])  # coefficients

        # evaluate it on grid
        Z = C[0] * X + C[1] * Y + C[2]

        # or expressed using matrix/vector product
        # Z = np.dot(np.c_[XX, YY, np.ones(XX.shape)], C).reshape(X.shape)

    elif order == 2:
        # best-fit quadratic curve
        A = np.c_[np.ones(data.shape[0]), data[:, :2], np.prod(data[:, :2], axis=1), data[:, :2] ** 2]
        C, _, _, _ = scipy.linalg.lstsq(A, data[:, 2])

        # evaluate it on a grid
        Z = np.dot(np.c_[np.ones(XX.shape), XX, YY, XX * YY, XX ** 2, YY ** 2], C).reshape(X.shape)

    # # plot points and fitted surface #
    # show_surf(data,X, Y, Z, title)

    return Z

def show_surf(data,X,Y,Z,title):

    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.plot_surface(X, Y, Z, rstride=1, cstride=1, alpha=0.2)
    ax.scatter(data[:, 0], data[:, 1], data[:, 2], c='r', s=50)
    plt.xlabel('H')
    plt.ylabel('W')
    ax.set_zlabel('Z')
    ax.axis('equal')
    ax.axis('tight')
    plt.title(title)
    plt.show()


if __name__=='__main__':

    # # some 3-dim points
    # mean = np.array([0.0, 0.0, 0.0])
    # cov = np.array([[1.0, -0.5, 0.8], [-0.5, 1.1, 0.0], [0.8, 0.0, 1.0]])
    # data = np.random.multivariate_normal(mean, cov, 50)
    #
    # fit_3d(data)


    # # test2
    x=[0,1,2,3]
    y=[-1,0.2,0.9,2.1]
    leastsq(x,y)