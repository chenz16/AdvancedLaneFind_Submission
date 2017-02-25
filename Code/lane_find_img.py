import numpy as np
import cv2
import glob
import matplotlib.pyplot as plt
import pickle
from camera_cal2 import camera_cal


mtx, dist = camera_cal()


images = glob.glob('../test_images/*.jpg')

def hls_select(img, thresh=(0, 255)):
    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    s_channel = hls[:,:,2]
    binary_output = np.zeros_like(s_channel)
    binary_output[(s_channel > thresh[0]) & (s_channel <= thresh[1])] = 1
    return binary_output

def mag_thresh(img, sobel_kernel=3, mag_thresh=(0, 255)):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    gradmag = np.sqrt(sobelx**2 + sobely**2)
    scale_factor = np.max(gradmag)/255
    gradmag = (gradmag/scale_factor).astype(np.uint8)
    binary_output = np.zeros_like(gradmag)
    binary_output[(gradmag >= mag_thresh[0]) & (gradmag <= mag_thresh[1])] = 1
    return binary_output

def dir_threshold(img, sobel_kernel=3, thresh=(0, np.pi/2)):
    # Grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # Calculate the x and y gradients
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    # Take the absolute value of the gradient direction,
    # apply a threshold, and create a binary image result
    absgraddir = np.arctan2(np.absolute(sobely), np.absolute(sobelx))
    binary_output =  np.zeros_like(absgraddir)
    binary_output[(absgraddir >= thresh[0]) & (absgraddir <= thresh[1])] = 1

    # Return the binary image
    return binary_output

for idx, fname in enumerate(images):
    img = cv2.imread(fname)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    undst = cv2.undistort(img, mtx, dist, None, mtx)
    img_s_sel          = hls_select(undst, thresh=(90, 255))
    img_grad_mag_sel   = mag_thresh(undst,sobel_kernel=15, mag_thresh=(10, 100) )
    img_dir_sel        = dir_threshold(undst,sobel_kernel=15, thresh=(0.7, 1.2) )

    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(20,10))
    f.tight_layout()
    ax1.imshow(undst)
    ax1.set_title('Original Image', fontsize=30)
    ax2.imshow(img_grad_mag_sel, cmap='gray')
    ax2.set_title('Undistorted Image', fontsize=30)
    write_name = '../test_images/mag_sel/'
    f.savefig(write_name + str(idx) + '.jpg')
    plt.close('all')
    #plt.pause(0.5)

    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(20,10))
    f.tight_layout()
    ax1.imshow(undst)
    ax1.set_title('Original Image', fontsize=30)
    ax2.imshow(img_s_sel, cmap='gray')
    ax2.set_title('Undistorted Image', fontsize=30)
    write_name = '../test_images/Schannel_sel/'
    f.savefig(write_name + str(idx) + '.jpg')
    plt.close('all')
    #plt.pause(0.5)



    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(20,10))
    f.tight_layout()
    ax1.imshow(undst)
    ax1.set_title('Original Image', fontsize=30)
    ax2.imshow(img_dir_sel, cmap='gray')
    ax2.set_title('Undistorted Image', fontsize=30)
    write_name = '../test_images/dir_sel/'
    f.savefig(write_name + str(idx) + '.jpg')
    plt.close('all')
