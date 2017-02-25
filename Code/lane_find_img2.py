import numpy as np
import cv2
import glob
import matplotlib.pyplot as plt
import pickle
from camera_cal2 import camera_cal


def abs_sobel_thresh(img, orient='x', sobel_kernel=3, thresh=(0, 255)):
    # Calculate directional gradient
    # Apply threshold
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    if orient is 'x':
        sobel = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    elif orient is 'y':
        sobel = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    else:
        print('Error orient')

    abs_sobel = np.absolute(sobel)
    scaled_sobel = np.uint8(255*abs_sobel/np.max(abs_sobel))
    grad_binary = np.zeros_like(sobel)
    grad_binary[(scaled_sobel >= thresh[0]) & (scaled_sobel <= thresh[1])] = 1

    return grad_binary

def mag_thresh(image, sobel_kernel=3, mag_thresh=(0, 255)):
    # Calculate gradient magnitude
    # Apply threshold
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    gradmag = np.sqrt(sobelx**2 + sobely**2)
    scale_factor = np.max(gradmag)/255
    gradmag = (gradmag/scale_factor).astype(np.uint8)
    mag_binary = np.zeros_like(gradmag)
    mag_binary[(gradmag >= mag_thresh[0]) & (gradmag <= mag_thresh[1])] = 1
    return mag_binary

def dir_threshold(image, sobel_kernel=3, thresh=(0, np.pi/2)):
    # Calculate gradient direction
    # Apply threshold
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # Calculate the x and y gradients
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    # Take the absolute value of the gradient direction,
    # apply a threshold, and create a binary image result
    absgraddir = np.arctan2(np.absolute(sobely), np.absolute(sobelx))
    dir_binary=  np.zeros_like(absgraddir)
    dir_binary[(absgraddir >= thresh[0]) & (absgraddir <= thresh[1])] = 1

    return dir_binary

def hls_select(img, thresh=(0, 255)):
    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    s_channel = hls[:,:,2]
    binary_output = np.zeros_like(s_channel)
    binary_output[(s_channel > thresh[0]) & (s_channel <= thresh[1])] = 1
    return binary_output

def perspective_trans(img):

    src = np.float32([[280, 675], [1040, 675], [909,590], [390, 596]])
    dst = np.float32([[280,675], [1040,675], [1040, 590], [280,590]])

    img_size = (img.shape[1], img.shape[0])
    # Given src and dst points, calculate the perspective transform matrix
    M = cv2.getPerspectiveTransform(src, dst)
    # Warp the image using OpenCV warpPerspective()
    warped = cv2.warpPerspective(img, M, img_size)
    return warped

mtx, dist = camera_cal()
images = glob.glob('../test_images/*.jpg')

for idx, fname in enumerate(images):
    img = cv2.imread(fname)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    image_color = cv2.undistort(img, mtx, dist, None, mtx)
    s_binary = hls_select(image_color, thresh=(100, 255))
    #image = image_color
    image = np.dstack((s_binary, s_binary, s_binary))
    #print(image.shape)

    ksize = 9
    gradx = abs_sobel_thresh(image, orient='x', sobel_kernel=ksize, thresh=(50, 200))
    grady = abs_sobel_thresh(image, orient='y', sobel_kernel=ksize, thresh=(50, 200))
    mag_binary = mag_thresh(image, sobel_kernel=ksize, mag_thresh=(80, 255))
    dir_binary = dir_threshold(image, sobel_kernel=ksize, thresh=(0.7, 1.2))
    #s_binary          = hls_select(image, thresh=(60, 255))
    combined = np.zeros_like(dir_binary)
    combined[((gradx == 1) & (grady == 1)) | ((mag_binary == 1) & (dir_binary == 1))
                | (s_binary ==1)] = 1
    #combined[s_binary ==1] =1
    #combined[((gradx == 1) & (grady == 1)) | ((mag_binary == 1) & (dir_binary == 1))
                #| ((s_binary ==1) & (dir_binary == 1))] = 1


    # Next we'll create a masked edges image using cv2.fillPoly()
    mask = np.zeros_like(combined)
    ignore_mask_color = 1

    # This time we are defining a four sided polygon to mask
    imshape = image.shape
    #vertices = np.array([[(0,imshape[0]),(0, 0), (imshape[1], 0), (imshape[1],imshape[0])]], dtype=np.int32)
    vertices = np.array([[(100,imshape[0]),(620, 400), (750, 400), (1200,imshape[0])]], dtype=np.int32)

    cv2.fillPoly(mask, vertices, ignore_mask_color)
    '''
    plt.imshow(mask, cmap='gray')
    plt.show()
    print(combined.shape, mask.shape)'''

    #combined = cv2.bitwise_and(combined, mask)

    image_trans =  perspective_trans(combined)
    histogram = np.sum(image_trans[image_trans.shape[0]/2:,:], axis=0)

    out_img = np.dstack((image_trans*100, image_trans*255, image_trans*255))

    '''
    f, (ax1, ax2) = plt.subplots(1, 2, figsize = (24, 10))
    ax1.imshow(out_img)
    ax2.imshow(image_trans, cmap = 'gray')
    plt.show()
    '''
    midpoint = np.int(histogram.shape[0]/2)
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint

    # Choose the number of sliding windows
    nwindows = 9
    # Set height of windows
    window_height = np.int(image_trans.shape[0]/nwindows)
    # Identify the x and y positions of all nonzero pixels in the image
    nonzero = image_trans.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])

    leftx_current = leftx_base
    rightx_current = rightx_base
    # Set the width of the windows +/- margin
    margin = 100
    # Set minimum number of pixels found to recenter window
    minpix = 50
    # Create empty lists to receive left and right lane pixel indices
    left_lane_inds = []
    right_lane_inds = []

    for window in range(nwindows):
    # Identify window boundaries in x and y (and right and left)
        win_y_low = image_trans.shape[0] - (window+1)*window_height
        win_y_high = image_trans.shape[0] - window*window_height
        win_xleft_low = leftx_current - margin
        win_xleft_high = leftx_current + margin
        win_xright_low = rightx_current - margin
        win_xright_high = rightx_current + margin
        # Draw the windows on the visualization image
        cv2.rectangle(out_img,(win_xleft_low,win_y_low),(win_xleft_high,win_y_high),(100,255,255), 2)
        cv2.rectangle(out_img,(win_xright_low,win_y_low),(win_xright_high,win_y_high),(100,255,0), 2)
        #cv2.waitKey(500)

        # Identify the nonzero pixels in x and y within the window
        good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
        good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]
        # Append these indices to the lists
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)
        # If you found > minpix pixels, recenter next window on their mean position
        if len(good_left_inds) > minpix:
            leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
        if len(good_right_inds) > minpix:
                rightx_current = np.int(np.mean(nonzerox[good_right_inds]))

    # Concatenate the arrays of indices
    #print(type(left_lane_inds))
    left_lane_inds = np.concatenate(left_lane_inds)
    #print(type(left_lane_inds))
    right_lane_inds = np.concatenate(right_lane_inds)

    # Extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds]
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]

    # Fit a second order polynomial to each

    ym_per_pix = 30/720 # meters per pixel in y dimension
    xm_per_pix = 3.7/700 # meters per pixel in x dimension
    y_eval = np.int(out_img.shape[0]/2)-1

    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)


    left_fit_cr = np.polyfit(lefty*ym_per_pix, leftx*xm_per_pix, 2)
    right_fit_cr = np.polyfit(righty*ym_per_pix, rightx*xm_per_pix, 2)

    # Calculate the new radii of curvature
    left_curverad = ((1 + (2*left_fit_cr[0]*y_eval*ym_per_pix + left_fit_cr[1])**2)**1.5) / np.absolute(2*left_fit_cr[0])
    right_curverad = ((1 + (2*right_fit_cr[0]*y_eval*ym_per_pix + right_fit_cr[1])**2)**1.5) / np.absolute(2*right_fit_cr[0])
    # Now our radius of curvature is in meters
    print(left_curverad, 'm', right_curverad, 'm')


    # visualize

    ploty = np.linspace(0, image_trans.shape[0]-1, image_trans.shape[0] )
    left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
    right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]

    out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [1, 255, 1]
    out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [100, 100, 255]

    f, ax =plt.subplots(figsize=(20,10))
    ax.imshow(out_img)
    #plt.imshow(out_img)
    #plt.show()
    ax.plot(left_fitx, ploty, color='yellow')
    ax.plot(right_fitx, ploty, color='yellow')
    plt.xlim(0, 1280)
    plt.ylim(720, 0)
    #plt.pause(1)
    write_name = '../test_images/curve_fit/'
    f.savefig(write_name + str(idx) + '.jpg')
    plt.close('all')
    plt.close('all')

    veh_center_pix = np.int(left_fitx[-1]/2 + right_fitx[-1]/2) +1
    veh_offcent_pix = veh_center_pix - np.int(image_trans.shape[1]/2)
    veh_offcenter = veh_offcent_pix* xm_per_pix
    if veh_offcenter >= 0:
        print('vehice is at left of lane center', veh_offcenter, 'm')
    else:
        print('vehice is at right of lane center', np.absolute(veh_offcenter), 'm')

    warp_zero = np.zeros_like(image_trans).astype(np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

    # Recast the x and y points into usable format for cv2.fillPoly()
    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    pts = np.hstack((pts_left, pts_right))

    # Draw the lane onto the warped blank image
    cv2.fillPoly(color_warp, np.int_([pts]), (0,255, 0))

    # Warp the blank back to original image space using inverse perspective matrix (Minv)
    #color_warp = out_img


    src = np.float32([[280, 675], [1040, 675], [909,590], [390, 596]])
    dst = np.float32([[280,675], [1040,675], [1040, 590], [280,590]])

    Minv = cv2.getPerspectiveTransform(dst, src)
    newwarp = cv2.warpPerspective(color_warp, Minv, (image_color.shape[1], image_color.shape[0]))
    # Combine the result with the original image
    result = cv2.addWeighted(image_color, 1, newwarp, 0.3, 0)
    plt.imshow(result)
    plt.show()


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
    ax2.imshow(image_trans, cmap='gray')
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
    ax1.imshow(image_trans, cmap = 'gray')
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
