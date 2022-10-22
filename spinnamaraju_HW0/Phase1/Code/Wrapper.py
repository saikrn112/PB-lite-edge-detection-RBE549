#!/usr/bin/env python
# coding: utf-8

# In[1]:


import cv2
import numpy as np
import math
from matplotlib import pyplot as plt
from matplotlib.pyplot import figure
from sklearn.cluster import KMeans
from scipy import signal


# In[2]:


def normalize(image_mat):
    return cv2.normalize(image_mat,dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)

def gaussian1D(dim,sigma,derivative_order,x):
    numerator = pow(x,2)
    variance = pow(sigma,2)
    exponent = numerator/variance
    exponent = -0.5*exponent
    comp = np.exp(exponent)
    comp = comp/(sigma*np.sqrt(2*np.pi))
    scale_factor = 1/np.sum(comp)
    comp = comp*scale_factor
    if derivative_order == 1:
        comp = np.multiply(comp,np.asmatrix(x))
        comp = -comp/pow(sigma,2)
    elif derivative_order == 2:
        comp = np.multiply(comp,(numerator - variance)/(pow(variance,2)))
    comp = np.transpose(np.asmatrix(comp))
    return comp

def plot_gaussian1D():
    kernel_size = 15
    val = math.floor(kernel_size/2)
    x = np.arange(-val,val+1,1)
    sigma = 2
    orders = [0,1,2]
    plt.figure(figsize=(8,8),dpi=200)
    fig,axs = plt.subplots(1,len(orders))
    for i,order in enumerate(orders):
        val = gaussian1D(kernel_size,sigma,order,x)
        val = normalize(val)
        axs[i].set_yticklabels([])
        axs[i].set_xticklabels([])
        axs[i].set_xticks([])
        axs[i].set_yticks([])
        axs[i].imshow(cv2.cvtColor(val,cv2.COLOR_BGR2RGB))
        
# plot_gaussian1D()


# In[3]:


def gaussian2D(dim,sigma_x,sigma_y,x_order,y_order,theta):
    val = math.floor(dim/2)
    x = np.arange(-val,val+1,1)
    y = x
    x2d,y2d = np.meshgrid(x,x)
    coords = np.column_stack((y2d.ravel(),x2d.ravel()))
    rotation_mat = np.matrix([[np.cos(theta),-np.sin(theta)],[np.sin(theta),np.cos(theta)]])
    points = rotation_mat.dot(coords.transpose())
    ret_x = gaussian1D(dim,sigma_x,x_order,np.asarray(points[0]))
    ret_y = gaussian1D(dim,sigma_y,y_order,np.asarray(points[1]))
    ret = np.multiply(ret_x,ret_y)
    ret = np.reshape(ret,[len(x),len(x)])
    return ret


def plot_symmetricalGaussian2D():
    kernel_size = 3
    sigma = 2
    theta = np.pi*(0/180)
    orders = [[0,0],[1,0],[0,1]]
    plt.figure(figsize=(8,8),dpi=200)
    fig,axs = plt.subplots(1,len(orders))
    for i,order in enumerate(orders):
        val = gaussian2D(kernel_size,sigma,sigma,order[0],order[1],theta)
        val = normalize(val)
        axs[i].set_yticklabels([])
        axs[i].set_xticklabels([])
        axs[i].set_xticks([])
        axs[i].set_yticks([])
        axs[i].imshow(cv2.cvtColor(val,cv2.COLOR_BGR2RGB))
        
def plot_asymmetricalGaussian2D():
    kernel_size = 30
    sigma_x = 2
    sigma_y = 6
    theta = np.pi*(0/180)
    orders = [[0,0],[1,0],[0,1],[2,0],[0,2]]
    plt.figure(figsize=(8,8),dpi=200)
    fig,axs = plt.subplots(1,len(orders))
    for i,order in enumerate(orders):
        val = gaussian2D(kernel_size,sigma_x,sigma_y,order[0],order[1],theta)
        val = normalize(val)
        axs[i].set_yticklabels([])
        axs[i].set_xticklabels([])
        axs[i].set_xticks([])
        axs[i].set_yticks([])
        _ = axs[i].imshow(cv2.cvtColor(val,cv2.COLOR_BGR2RGB))
    plt.show()

def plot_asymmetricalRotatedGaussian2D():
    kernel_size = 20
    sigma_x = 1
    sigma_y = 3
    thetas = np.array([0,30,60,90,120,150,180,210,240])*np.pi/180
    thetas = np.array([0,60,90])*np.pi/180
    plt.figure(figsize=(8,8),dpi=200)
    fig,axs = plt.subplots(1,len(thetas))
    for i,theta in enumerate(thetas):
        val = gaussian2D(kernel_size,sigma_x,sigma_y,0,0,theta)
        val = normalize(val)
        axs[i].set_yticklabels([])
        axs[i].set_xticklabels([])
        axs[i].set_xticks([])
        axs[i].set_yticks([])
        axs[i].imshow(cv2.cvtColor(val,cv2.COLOR_BGR2RGB))
    plt.show();
# plot_symmetricalGaussian2D();
# plot_asymmetricalGaussian2D();
# plot_asymmetricalRotatedGaussian2D();


# In[4]:


def OrientedDerivativeOfGaussian(kernel_size,sigma,orient):
    sobel_filter_x = gaussian2D(kernel_size,sigma,sigma,1,0,0)
    sobel_filter_y = gaussian2D(kernel_size,sigma,sigma,0,1,0)
    sobel_filter_x = cv2.filter2D(sobel_filter_x,ddepth=-1,kernel=gaussian2D(3,1,1,0,0,0))
    sobel_filter_y = cv2.filter2D(sobel_filter_y,ddepth=-1,kernel=gaussian2D(3,1,1,0,0,0))
#     sobel_filter_x = signal.convolve2d(sobel_filter_x,gaussian2D(3,1,1,0,0,0))
#     sobel_filter_y = signal.convolve2d(sobel_filter_y,gaussian2D(3,1,1,0,0,0))
    ret = np.cos(orient)*sobel_filter_x + np.sin(orient)*sobel_filter_y
    return ret

def test_DoG_filterbank():
    scales = [5,10]
    sigma = 2
    orients = np.array([0,30,60,90,120,150,180,210,240,270,300,330])*np.pi/180
    imagesv = []
    fig,axs = plt.subplots(len(scales),len(orients))
    for i,scale in enumerate(scales):
        if len(scales) > 1:
            axi = axs[i]
        else:
            axi = axs
        for j,orient in enumerate(orients):
            if len(orients) > 1:
                axj = axi[j]
            else:
                axj = axj
            val = OrientedDerivativeOfGaussian(scale,sigma,orient)
            axj.imshow(cv2.cvtColor(normalize(val),cv2.COLOR_BGR2RGB))
            axj.axis('off')
            axj.set_xticks([])
            axj.set_yticks([])

    fig.tight_layout()
#     fig.suptitle('Oriented Derivative of Gaussians',fontsize=16,y=0.8)
    plt.savefig('DoG.png',dpi=1200)
test_DoG_filterbank()
            
# test_DoG_filterbank()


# In[5]:


def convolve_odog():
    # dog_filter = gaussian2D(3,2,2,0,0,0)*10
    dog_filter = OrientedDerivativeOfGaussian(3,1,0)*100000
    filtered_img= cv2.filter2D(img,ddepth=-1,kernel=dog_filter)
    print(dog_filter)
    plt.imshow(cv2.cvtColor(filtered_img,cv2.COLOR_BGR2RGB))
# convolve_odog()


# In[6]:


def LM_filterbank(kernel_size,scales):

    orients = np.array([0,30,60,90,120,150])*np.pi/180
    filter_banks = []


    filter_banks = []
    for scale in scales[0:3]:
        scale_filters = []
        #LMS filter_banks first order 6*3 = 18
        for orient in orients:
            scale_filters.append(gaussian2D(kernel_size,scale,3*scale,1,0,orient))
        #LMS filter_banks second order 6*3 = 18
        for orient in orients:
            scale_filters.append(gaussian2D(kernel_size,scale,3*scale,2,0,orient))
        filter_banks.append(scale_filters)

    last_banks = []
    
    #laplacian of gaussians 8
    loG_scales = []
    loG_scales.append(scales)
    loG_scales.append(2*scales)
    loG_scales = [num for elem in loG_scales for num in elem]

    for scale in loG_scales:
        log_x = gaussian2D(kernel_size,scale,scale,2,0,0)
        log_y = gaussian2D(kernel_size,scale,scale,0,2,0)
        last_banks.append(log_x+log_y)
    
    #LMS filter_banks gaussians 4
    for scale in scales:
        last_banks.append(gaussian2D(kernel_size,scale,scale,0,0,0))
    
    filter_banks.append(last_banks)
    return filter_banks

def test_LM_filterbank():
    kernel_size = 49
    scales = np.array([1,np.sqrt(2),2,2*np.sqrt(2)])
    LMS_filter_banks = LM_filterbank(kernel_size,scales)
    plt.figure(figsize=(12,12),dpi=200)
    fig,axs = plt.subplots(np.shape(LMS_filter_banks)[0]*2,np.shape(LMS_filter_banks)[1])
    for i,filters in enumerate(LMS_filter_banks):
        for j,filter_g in enumerate(filters):
            val = normalize(filter_g)
            axs[i,j].imshow(cv2.cvtColor(val,cv2.COLOR_BGR2RGB))
            axs[i,j].set_xticks([])
            axs[i,j].set_yticks([])

    
    scales = np.array([np.sqrt(2),2,2*np.sqrt(2),4])
    LML_filter_banks = LM_filterbank(kernel_size,scales)
#     fig,axs = plt.subplots(np.shape(LML_filter_banks)[0],np.shape(LMS_filter_banks)[1])
    for i,filters in enumerate(LML_filter_banks):
        for j,filter_g in enumerate(filters):
            val = normalize(filter_g)
            axs[i+np.shape(LMS_filter_banks)[0],j].imshow(cv2.cvtColor(val,cv2.COLOR_BGR2RGB))
            axs[i+np.shape(LMS_filter_banks)[0],j].set_xticks([])
            axs[i+np.shape(LMS_filter_banks)[0],j].set_yticks([])

#     fig.suptitle('Leung-Malik filters',fontsize=16,y=1.001)
    fig.tight_layout()
    plt.savefig('LM.png',dpi=1200)

test_LM_filterbank()


# In[7]:


def gabor_filter(dim,lamda,theta,psi,sigma,gamma):
    val = math.floor(dim/2)
    x = np.arange(-val,val+1,1)
    y = x
    x2d,y2d = np.meshgrid(x,x)
    coords = np.column_stack((y2d.ravel(),x2d.ravel()))
    rotation_mat = np.matrix([[np.cos(theta),np.sin(theta)],[-np.sin(theta),np.cos(theta)]])
    points = rotation_mat.dot(coords.transpose())
    gaussian_part = np.exp(-(np.power(points[0],2)+np.power(gamma,2)*np.power(points[1],2))/(2*np.power(sigma,2)))
    sinusodial_part_real = np.cos(2*np.pi*points[0]/lamda+psi)
    sinusodial_part_complex = np.sin(2*np.pi*points[0]/lamda+psi)
    comp = []
    comp = np.multiply(gaussian_part,sinusodial_part_real)
    comp = np.reshape(comp,[len(x),len(x)])
    return comp

def test_gabor_filter():
    val = gabor_filter(100,14,np.pi*(0/180),np.pi*(0/180),11,0.9)
    val = normalize(val)
    plt.imshow(cv2.cvtColor(val,cv2.COLOR_BGR2RGB))
test_gabor_filter()


# In[8]:


def convolve_img_gabor():
    filters = gabor_filter(100,14,np.pi*(10/180),np.pi*(20/180),11,0.9)/10
    img1= cv2.filter2D(img,ddepth=-1,kernel=filters[47])
    print(np.shape(img))
    plt.imshow(cv2.cvtColor(img1,cv2.COLOR_BGR2RGB))
# convolve_img_gabor()


# In[9]:


def create_and_plot_gabor_kernels(plot=False):
    
    gabor_kernel_size = 31

    gabor_sigma = [6,8,12,20]
    gabor_theta = np.pi/180*np.array([0,30,60,90,120,150,170])
    gabor_psi = np.pi/180*np.array([0])
    gabor_lambda = [10]
    gabor_gamma = [1]
    gabor_filters = []
    for sigma in gabor_sigma:
        for theta in gabor_theta:
            for psi in gabor_psi:
                for glambda in gabor_lambda:
                    for gamma in gabor_gamma:
                        g_filter = gabor_filter(gabor_kernel_size,glambda,theta,psi,sigma,gamma)
                        gabor_filters.append(g_filter)

    if plot:
        plt.figure(figsize=(8,8),dpi=200)
        fig, axs = plt.subplots(len(gabor_sigma),len(gabor_theta))
        axs = axs.ravel()
        for i,gfilter in enumerate(gabor_filters):
            axs[i].set_xticks([])
            axs[i].set_yticks([])
            axs[i].imshow(cv2.cvtColor(normalize(gfilter),cv2.COLOR_BGR2RGB))
        fig.tight_layout()
        plt.savefig('Gabor.png')
    return gabor_filters

_ = create_and_plot_gabor_kernels(True)


# In[10]:


def group_filters():
    dog_scaling_factor = 100000
    dog_scales = [5,10]
    dog_sigma = 2
    dog_orients = np.array([0,30,60,90,120,150,180,210,240,270,300,330])*np.pi/180
    dog_filters = []
    for scale in dog_scales:
        for orient in dog_orients:
            dog = OrientedDerivativeOfGaussian(scale,dog_sigma,orient)*10000
            dog_filters.append(dog)
    print("nDoG:",np.shape(dog_filters))

    LM_scaling_factor = 2000
    LM_kernel_size = 49
    LMS_scales = np.array([np.sqrt(2),2,2*np.sqrt(2),4])
    LMS_filters = LM_filterbank(LM_kernel_size,LMS_scales)
    LMS_filters = np.reshape(np.array(LMS_filters),[48,49,49])*LM_scaling_factor
    
    LML_scales = np.array([np.sqrt(2),2,2*np.sqrt(2),4])
    LML_filters = LM_filterbank(LM_kernel_size,LML_scales)
    LML_filters = np.reshape(np.array(LML_filters),[48,49,49])*LM_scaling_factor
    
    LM_filters = LMS_filters.tolist() + LML_filters.tolist()
    print("nLM:",np.shape(LM_filters))
    
    gabor_filters = create_and_plot_gabor_kernels()

    print("nGabor:",np.shape(gabor_filters))
    filter_bank = dog_filters
    filter_bank += LM_filters
    filter_bank += gabor_filters
    return filter_bank


# In[11]:


def half_disk_mask_pair(dim,theta):
    val = math.floor(dim/2)
    x = np.arange(-val,val+1,1)
    y = x
    x2d,y2d = np.meshgrid(x,x)
    coords = np.column_stack((y2d.ravel(),x2d.ravel()))
    coords = np.transpose(coords)
    angles = np.arctan2(coords[1],coords[0])
    magnis = np.power(np.power(coords[1],2) + np.power(coords[0],2),0.5)
    image1 = np.full(len(x)*len(x),0)
    image2 = np.full(len(x)*len(x),0)
    
    for i,angle in enumerate(angles):
        if angle >=0 and (angle <= theta or angle == np.pi - theta) and magnis[i] <= len(x)/2:
            image1[i] = 255
        if angle < 0 and angle >= (theta - np.pi) and magnis[i] <= len(x)/2:
            image1[i] = 255
            continue
    for i,angle in enumerate(angles):
        if angle >=0 and (angle > theta and (not angle == np.pi - theta)) and magnis[i] <= len(x)/2:
            image2[i] = 255
        if angle < 0 and (angle < (theta - np.pi)) and magnis[i] <= len(x)/2:
            image2[i] = 255
            continue
    
    image1 = np.reshape(image1,[len(x),len(x)])
    image2 = np.reshape(image2,[len(x),len(x)])
    return [image1,image2]
    
def plot_half_disk_masks():
    scales = [10,20,30]
    orients = np.array([0,30,75,90,115,130,145])*np.pi/180
    plt.figure(figsize=(8,8),dpi=200)
    fig,axs = plt.subplots(len(scales),2*len(orients))
    for i,scale in enumerate(scales):
        if len(scales) > 1:
            ax = axs[i]
        else:
            ax = axs
        for j,orient in enumerate(orients):
            mask = half_disk_mask_pair(scale,orient)
            mask[0] = normalize(mask[0])
            ax[2*j].set_yticklabels([])
            ax[2*j].set_xticklabels([])
            ax[2*j].set_xticks([])
            ax[2*j].set_yticks([])
            ax[2*j].imshow(cv2.cvtColor(mask[0],cv2.COLOR_BGR2RGB))
            mask[1] = normalize(mask[1])
            ax[2*j+1].set_yticklabels([])
            ax[2*j+1].set_xticklabels([])
            ax[2*j+1].set_xticks([])
            ax[2*j+1].set_yticks([])
            ax[2*j+1].imshow(cv2.cvtColor(mask[1],cv2.COLOR_BGR2RGB))
    plt.savefig('HDMasks.png',dpi=1200)
            
plot_half_disk_masks()


# In[12]:


# Brightness and color map
def get_cluster_map(img,clusters,n_features):
    img_shape = np.shape(img)
    img_2d = img.reshape(-1,n_features)
    km = KMeans(n_clusters = clusters, init='random',n_init=10,max_iter=300,tol=1e-04, random_state=0)
    clustered_image = km.fit_predict(img_2d)
    clustered_image = np.reshape(clustered_image,(img_shape[0],img_shape[1]))
    return clustered_image
def plot_brightness_color_clustered_map(img,img_color):
    clustered_brightness = get_cluster_map(img,16,1)
    clustered_color = get_cluster_map(img_color,16,3)
    plt.subplot(121),plt.imshow(cv2.cvtColor(normalize(clustered_brightness),cv2.COLOR_BGR2RGB))
    plt.title('clustered_brightness')
    plt.xticks([]),plt.yticks([])
    plt.subplot(122),plt.imshow(cv2.cvtColor(normalize(clustered_color),cv2.COLOR_BGR2RGB))
    plt.title('clustered_color')
    plt.xticks([]),plt.yticks([])

# plot_brightness_color_clustered_map(img,img_color)    


# In[13]:


# gradients
def update_chi_sqr_dist(chi_image,g_i,h_i):
    distance = np.power(g_i - h_i,2)
    np.seterr(invalid='ignore')
    distance = np.divide(distance,(g_i+h_i))
    np.seterr(invalid=None)
    distance = distance/2
    chi_image += distance

def interpolate_nan_and_inf(image):
    r_img = image.reshape(-1,1)
    mask = np.logical_or(np.isnan(r_img),np.isinf(r_img))
    r_img[mask] = np.interp(np.flatnonzero(mask),np.flatnonzero(~mask), r_img[~mask])
    image = r_img.reshape(image.shape)
    return image

def get_gradient(img,left_half,right_half):
    chi_image = np.zeros(np.shape(img))
    img0 = img.flatten()
    tmp = np.zeros(np.shape(img0))
    for i in range(255):
        tmp[img0 == i] = 1
        reshaped_tmp = tmp.reshape(img.shape)
        g_i = cv2.filter2D(reshaped_tmp,ddepth=-1,kernel=left_half)
        h_i = cv2.filter2D(reshaped_tmp,ddepth=-1,kernel=right_half)
        update_chi_sqr_dist(chi_image,g_i,h_i)
    chi_image = np.power(chi_image,0.5)
    return chi_image

def get_image_gradients(img):
    scales = [7,11,21]
    orients = np.array([0,30,75,90,115,130,145])*np.pi/180
    images = []
    for scale in scales:
        for orient in orients:
            mask = half_disk_mask_pair(scale,orient)
            gradient_image = get_gradient(img,mask[0],mask[1])
            gradient_image = interpolate_nan_and_inf(gradient_image)
            images.append(gradient_image)
    return images


# In[14]:


# filters = group_filters()
# filtered_images = []
# for customFilter in filters:
#     filtered_images.append(cv2.filter2D(img,ddepth=-1,kernel=np.array(customFilter)))

# print(np.shape(filtered_images))


# In[15]:


# clustered_brightness = get_cluster_map(img,16,1)
# clustered_color = get_cluster_map(img_color,16,3)
# clustered_texture = get_cluster_map(np.transpose(filtered_images,axes=[1,2,0]),64,len(filters))


# In[16]:


# fig,axs = plt.subplots(1,3)
# axs[0].imshow(clustered_brightness)
# axs[1].imshow(clustered_color)
# axs[2].imshow(clustered_texture)


# In[17]:


# brightness_gradients = get_image_gradients(clustered_brightness)
# color_gradients = get_image_gradients(clustered_color)
# texture_gradients = get_image_gradients(clustered_texture)


# In[18]:


# brightness_gradients = np.mean(brightness_gradients, axis=0)
# color_gradients = np.mean(color_gradients, axis=0)
# texture_gradients = np.mean(texture_gradients, axis=0)


# In[19]:


# print(np.shape(brightness_gradients))
# fig,axs = plt.subplots(1,3)
# axs[0].imshow(brightness_gradients,cmap='gist_gray')
# axs[0].set_xticks([]), axs[0].set_yticks([])
# axs[1].imshow(color_gradients,cmap='gist_gray')
# axs[1].set_xticks([]), axs[1].set_yticks([])
# axs[2].imshow(texture_gradients,cmap='gist_gray')
# axs[2].set_xticks([]), axs[2].set_yticks([])
# plt.savefig('gradients_'+image_name+'.png')


# In[20]:


# canny_baseline = cv2.imread('../spinnamaraju_HW0/Phase1/BSDS500/CannyBaseline/'+image_name+'.png',cv2.IMREAD_GRAYSCALE)
# sobel_baseline = cv2.imread('../spinnamaraju_HW0/Phase1/BSDS500/SobelBaseline/'+image_name+'.png',cv2.IMREAD_GRAYSCALE)
# plt.subplot(121),plt.imshow(normalize(canny_baseline),cmap='gist_gray'),plt.title('canny_baseline')
# plt.xticks([]), plt.yticks([])
# plt.subplot(122),plt.imshow(sobel_baseline,cmap='gist_gray'),
# plt.xticks([]), plt.yticks([])


# In[21]:


#hadamard product
def get_pbEdges(texture_gradients,brightness_gradients,color_gradients,canny_baseline,sobel_baseline,w1,w2):
    sum_of_gradients = np.sum(texture_gradients,axis=0)
    sum_of_gradients += np.sum(brightness_gradients,axis=0)
    sum_of_gradients += np.sum(color_gradients,axis=0)
    gradients_avg = sum_of_gradients/3
    baseline_avg = w1*canny_baseline + w2*sobel_baseline
    pbEdges = np.multiply(gradients_avg,baseline_avg)
    return pbEdges
# print(np.shape(texture_gradients))
# print(np.shape(brightness_gradients))
# print(np.shape(color_gradients))

# pbEdges = get_pbEdges(texture_gradients,brightness_gradients,color_gradients,canny_baseline,sobel_baseline,0.5,0.5)
# plt.imshow(cv2.cvtColor(normalize(pbEdges),cv2.COLOR_BGR2RGB))


# In[22]:


def generate_pb_edges():
    image_edges = []
    for i in range(1,11):
        base_path = '../BSDS500/'
        image_name = str(i)
        img_color = cv2.imread(base_path + '/Images/'+ image_name + '.jpg')
        img = cv2.cvtColor(img_color, cv2.COLOR_BGR2GRAY)
        img_shape = np.shape(img)
        
        # filter bank generation
        filters = group_filters()
        filtered_images = []
        for customFilter in filters:
            filtered_images.append(cv2.filter2D(img,ddepth=-1,kernel=np.array(customFilter)))
        
        # generating clusters
        clustered_brightness = get_cluster_map(img,16,1)
        clustered_color = get_cluster_map(img_color,16,3)
        clustered_texture = get_cluster_map(np.transpose(filtered_images,axes=[1,2,0]),80,len(filters))
        ## plot individual texton maps
        plt.imshow(cv2.cvtColor(normalize(clustered_texture),cv2.COLOR_BGR2RGB))
        plt.xticks([])
        plt.yticks([])
        plt.savefig('TextonMap_'+image_name+'.png',dpi=1200)
        plt.close()
        ## plot all the clustered images (brightness,color,texture)
        fig,axs = plt.subplots(1,3)
        axs[0].imshow(clustered_brightness),axs[0].set_xticks([]),axs[0].set_yticks([])
        axs[1].imshow(clustered_color),axs[1].set_xticks([]),axs[1].set_yticks([])
        axs[2].imshow(clustered_texture),axs[2].set_xticks([]),axs[2].set_yticks([])
        plt.savefig('Clustered_'+image_name+'.png',dpi=1200)
        plt.close()
        # gradient computation
        brightness_gradients = get_image_gradients(clustered_brightness)
        color_gradients = get_image_gradients(clustered_color)
        texture_gradients = get_image_gradients(clustered_texture)
        
        brightness_gradients = np.mean(brightness_gradients, axis=0)
        color_gradients = np.mean(color_gradients, axis=0)
        texture_gradients = np.mean(texture_gradients, axis=0)
        ## plot individual gradients

        plt.imshow(cv2.cvtColor(normalize(brightness_gradients),cv2.COLOR_BGR2RGB))
        plt.xticks([])
        plt.yticks([])
        plt.savefig('Bg_'+image_name+'.png',dpi=1200)
        plt.close()
        
        plt.imshow(cv2.cvtColor(normalize(color_gradients),cv2.COLOR_BGR2RGB))
        plt.xticks([])
        plt.yticks([])
        plt.savefig('Cg_'+image_name+'.png',dpi=1200)
        plt.close()
        
        plt.imshow(cv2.cvtColor(normalize(texture_gradients),cv2.COLOR_BGR2RGB))
        plt.xticks([])
        plt.yticks([])
        plt.savefig('Tg_'+image_name+'.png',dpi=1200)
        plt.close()
        ## plot all the gradients
        fig,axs = plt.subplots(1,3)
        axs[0].imshow(cv2.cvtColor(normalize(brightness_gradients),cv2.COLOR_BGR2RGB))
        axs[0].set_xticks([]), axs[0].set_yticks([])
        axs[1].imshow(cv2.cvtColor(normalize(brightness_gradients),cv2.COLOR_BGR2RGB))
        axs[1].set_xticks([]), axs[1].set_yticks([])
        axs[2].imshow(cv2.cvtColor(normalize(brightness_gradients),cv2.COLOR_BGR2RGB))
        axs[2].set_xticks([]), axs[2].set_yticks([])
        plt.savefig('Gradients_'+image_name+'.png',dpi=1200)
        plt.close()
        
        # get canny and sobel baselines
        canny_baseline = cv2.imread(base_path + '/CannyBaseline/'+image_name+'.png',cv2.IMREAD_GRAYSCALE)
        sobel_baseline = cv2.imread(base_path + '/SobelBaseline/'+image_name+'.png',cv2.IMREAD_GRAYSCALE)
        plt.subplot(121),plt.imshow(normalize(canny_baseline),cmap='gist_gray'),plt.title('canny_baseline')
        plt.xticks([]), plt.yticks([])
        plt.subplot(122),plt.imshow(sobel_baseline,cmap='gist_gray'),plt.title('sobel_baseline')
        plt.xticks([]), plt.yticks([])
        plt.close()
        
        # get pb Edges
        pbEdges = get_pbEdges(texture_gradients,brightness_gradients,color_gradients,canny_baseline,sobel_baseline,0.5,0.5)
        
        # save individual pbEdges
       
        plt.imshow(cv2.cvtColor(normalize(pbEdges),cv2.COLOR_BGR2RGB))
        plt.xticks([])
        plt.yticks([])
        plt.savefig('PbLite_'+image_name + '.png',dpi=1200)
        plt.close()
        image_edges.append(pbEdges)
    
    fig,axs = plt.subplots(2,5)
    axs = axs.ravel()
    for i,image_edge in enumerate(image_edges):
        axs[i].imshow(cv2.cvtColor(normalize(image_edge),cv2.COLOR_BGR2RGB))
        axs[i].set_xticks([])
        axs[i].set_yticks([])
    plt.savefig('pbedges.png',dpi=1200)
    return image_edges
        
def main():
	image_edges = generate_pb_edges()
	return None


# In[63]:


# import matplotlib.gridspec as gridspec
# gs = gridspec.GridSpec(4,3)
# gs.update(wspace=0.025,hspace=0.05)
# fig, axs = plt.subplots(4,3)
# fig.delaxes(axs[3,2])
# fig.delaxes(axs[3,1])
# axs = axs.ravel()
# for i,image_edge in enumerate(image_edges):
#     axs[i].imshow(cv2.cvtColor(normalize(image_edge),cv2.COLOR_BGR2RGB))
#     axs[i].set_xticks([])
#     axs[i].set_yticks([])
#     axs[i].set_aspect('equal')
# # plt.subplots_adjust(wspace=0.0001,hspace=0.1)
# axs[9].set_position([0.24,-0.05,0.55,0.343])

# plt.savefig('pbedges.png',dpi=1200)


if __name__ == '__main__':
    main()
 


