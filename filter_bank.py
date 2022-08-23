import cv2
import numpy as np
import math
from matplotlib import pyplot as plt    
img = cv2.imread('lenna_test.png')
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) 
def gaussian1D(dim,sigma):
    val = math.floor(dim/2)
    exponent = pow(np.arange(-val,val+1,1)/(sigma),2)
    exponent = -exponent/2
    comp = np.exp(exponent)
    comp = comp/sigma
    comp = comp/np.sqrt(2*np.pi)
    return np.asmatrix(np.exp(comp))

def gaussian2D(dim,sigma):
    ret = gaussian1D(dim,sigma)
    ret = np.transpose(ret)*ret
    return ret
        
def firstOrderGaussianMat(dim,sigma):
    g1D = gaussian1D(dim,sigma)
    val = math.floor(dim/2)
    val = np.multiply(val,np.arange(-val,val+1,1))
    val = - val/pow(sigma,2)
    return np.asmatrix(val) 

sobel_filter_x = np.array([[0,0,0,0,0],[0,-1,0,1,0],[0,-2,0,2,0],[0,-1,0,1,0],[0,0,0,0,0]])
sobel_filter_y = np.transpose(sobel_filter_x)
sobel_filter_x = cv2.normalize(src=sobel_filter_x, dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)
sobel_filter_y = cv2.normalize(src=sobel_filter_y, dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)
sobel_gaussian_x = cv2.filter2D(sobel_filter_x,ddepth=-1,kernel=gaussian2D(3,1))
sobel_gaussian_y = cv2.filter2D(sobel_filter_y,ddepth=-1,kernel=gaussian2D(3,1))

nOrientations = 16
orientations = []
for theta in np.linspace(0,2*np.pi,nOrientations):
    val = np.cos(theta)*sobel_gaussian_x + np.sin(theta)*sobel_gaussian_y
    val = cv2.normalize(src=val, dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)
    orientations.append(val)


cv2.imshow('test',cv2.resize(sobel_filter_x,(960,50)))
cv2.waitKey(5000)
cv2.destroyAllWindows()
