# import modules
import numpy as np
import cv2
import glob
import matplotlib.pyplot as plt
import pickle

# import self defined modules
import visualization_save as vs
import configure as cfg
from process import camera_cal, abs_sobel_thresh, mag_thresh, dir_threshold
from process import hls_select, perspective_trans, sliding_window, curvature

test_pic = cfg.Souce['test_images'] # folder storing test images
images = glob.glob(test_pic + '*.jpg')
mtx, dist = camera_cal()

'''
for idx, fname in enumerate(images):
    img = cv2.imread(fname) # read image
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # convert color from BGR to RGB
    image_color = cv2.undistort(img, mtx, dist, None, mtx) # undistorted color image


    s_binary = hls_select(image_color, thresh=(100, 255)) # select color by s channel
    image = np.dstack((s_binary, s_binary, s_binary))

    ksize = 9
    gradx = abs_sobel_thresh(image_color, orient='x', sobel_kernel=ksize, thresh=(50, 200))
    grady = abs_sobel_thresh(image_color, orient='y', sobel_kernel=ksize, thresh=(50, 200))
    mag_binary = mag_thresh(image_color, sobel_kernel=ksize, mag_thresh=(80, 255))
    dir_binary = dir_threshold(image_color, sobel_kernel=ksize, thresh=(0.7, 1.2))

    combined = np.zeros_like(dir_binary)
    combined[((gradx == 1) & (grady == 1)) | ((mag_binary == 1) & (dir_binary == 1))
                | (s_binary ==1)] = 1

    # create a masked edges image using cv2.fillPoly()
    #mask = np.zeros_like(combined)
    #ignore_mask_color = 1

    # define a four sided polygon to mask
    #imshape = image_color.shape
    #vertices = np.array([[(100,imshape[0]),(620, 400), (750, 400), (1200,imshape[0])]], dtype=np.int32)
    #cv2.fillPoly(mask, vertices, ignore_mask_color)
    #combined = cv2.bitwise_and(combined, mask)

    image_warped =  perspective_trans(combined)
    histogram = np.sum(image_warped[image_warped.shape[0]/2:,:], axis=0)
    out_img = np.dstack((image_warped*100, image_warped*255, image_warped*255))


    f, (ax1, ax2) = plt.subplots(1, 2, figsize = (24, 10))
    ax1.imshow(out_img)
    ax2.imshow(image_warped, cmap = 'gray')
    plt.show()
    '''
    '''
    leftx, lefty, rightx, righty = sliding_window(image_warped)

    # Fit a second order polynomial to each

    y_eval = np.int(out_img.shape[0]/2)-1 # curvature is evaulated @ point y_eval
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)

    left_curverad, right_curverad = curvature (leftx, lefty, rightx, righty, y_eval)
    # Now our radius of curvature is in meters
    print(left_curverad, 'm', right_curverad, 'm')


    # visualize
    left_fitx, right_fity = polyfit_visualize(out_img, left_fit, right_fit, save_indx = 0):

    vehicle_offcenter = lane_center_offset(left_fitx, right_fity,ym_per_pix = 30/720, xm_per_pix = 3.7/700):

    result = image_overlay(image_warped, left_fitx, right_fitx)


    visualize_and_save(image_color, result, save_indx = 0, base_title = 'Original Image',
        new_title = 'new image', save_folder = '../test_images/transform2/'):
    '''

    '''
    y_eval = image_trains.shape[0]-1
    left_curverad = ((1 + (2*left_fitx[0]*y_eval + left_fitx[1])**2)**1.5) / np.absolute(2*left_fitx[0])
    right_curverad = ((1 + (2*right_fitx[0]*y_eval + right_fitx[1])**2)**1.5) / np.absolute(2*right_fitx[0])
    '''




    '''
    #print(np.amax(histogram, axis =0))

    f,(ax1, ax2) = plt.subplots(1, 2, figsize=(20,10))
    f.tight_layout()
    ax1.imshow(combined, cmap = 'gray')
    ax1.set_title('Original Image', fontsize=30)
    ax2.imshow(image_warped, cmap='gray')
    ax2.set_title('Transformed Image', fontsize=30)
    write_name = '../test_images/transform2/'
    f.savefig(write_name + str(idx) + '.jpg')
    plt.close('all')
    #plt.show()

    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(20,10))
    f.tight_layout()
    ax1.imshow(image_color)
    ax1.set_title('Original Image', fontsize=30)
    ax2.imshow(combined, cmap='gray')
    ax2.set_title('Undistorted Image', fontsize=30)
    write_name = '../test_images/combined/'
    f.savefig(write_name + str(idx) + '.jpg')
    plt.close('all')

    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(20,10))
    f.tight_layout()
    ax1.imshow(image_warped, cmap = 'gray')
    ax1.set_title('Original Image', fontsize=30)
    ax2.plot(histogram)
    ax2.set_title('Undistorted Image', fontsize=30)
    write_name = '../test_images/histogram/'
    f.savefig(write_name + str(idx) + '.jpg')
    plt.close('all')

    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(20,10))
    f.tight_layout()
    ax1.imshow(image_color)
    ax1.set_title('Original Image', fontsize=30)
    ax2.imshow(mag_binary, cmap = 'gray')
    ax2.set_title('Undistorted Image', fontsize=30)
    write_name = '../test_images/mag_sel/'
    f.savefig(write_name + str(idx) + '.jpg')
    plt.close('all')

    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(20,10))
    f.tight_layout()
    ax1.imshow(image_color)
    ax1.set_title('Original Image', fontsize=30)
    ax2.imshow(dir_binary, cmap = 'gray')
    ax2.set_title('Sobex Mag Image', fontsize=30)
    write_name = '../test_images/dir_sel/'
    f.savefig(write_name + str(idx) + '.jpg')
    plt.close('all')


    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(20,10))
    f.tight_layout()
    ax1.imshow(image_color)
    ax1.set_title('Original Image', fontsize=30)
    ax2.imshow(s_binary, cmap = 'gray')
    ax2.set_title('Sobex Mag Image', fontsize=30)
    write_name = '../test_images/Schannel_sel/'
    f.savefig(write_name + str(idx) + '.jpg')
    plt.close('all')
    '''
