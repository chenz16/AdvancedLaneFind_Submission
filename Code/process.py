'''
This module defines the core functions to process images

camera_cal: get camera calibration matrix

abs_sobel_thresh: return a binary array at each pixle
    where normalized pixle gradient  of x or y direction is within a specified range

mag_thresh: return a binary array at each pixle where normalized magnitude of pixle gradient
    is within a specified range

dir_threshold: return a binary array at each pixle where direction of pixle gradient is within
    a specified range

hls_select: return a binary array at each pixle  the s channel value is within a specified range

perspective_trans: perspective transformation providing options to convert image from camera view to
    bird eye view or vice versa

sliding_window: return non-zero pixle coordinations which is indentified from sliding window method

image_overlay: retun image overlay between the orginal color undistored image and
    the image with lane detected

lane_center_offset: define how much (m) the vehicle offsets the lane center

curvarad: return the turn radius of lanes  (curvature = 1/ turn_radius)


'''





import numpy as np
import cv2
import glob
import matplotlib.pyplot as plt
import pickle
import configure as cfg


# calibrate camera
def camera_cal():
    nx = cfg.Source['image_size'][0]
    ny = cfg.Source['image_size'][1]
    objp = np.zeros((nx*ny,3), np.float32) # 3- x,y,z coord, nx*ny = number of test points
    objp[:,:2] = np.mgrid[0:nx, 0:ny].T.reshape(-1,2) # define x, y, and keep z as zero

    # Arrays to store object points and image points from all the images.
    objpoints = [] # 3d points in real world space
    imgpoints = [] # 2d points in image plane.

    # Make a list of calibration images
    chessboard_pic = cfg.Source['chessboard']
    images = glob.glob(chessboard_pic + 'calibration*.jpg')

    # Step through the list and search for chessboard corners
    for idx, fname in enumerate(images):
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # Find the chessboard corners
        ret, corners = cv2.findChessboardCorners(gray, (nx,ny), None)
        # If found, add object points, image points
        if ret == True:
            objpoints.append(objp)
            imgpoints.append(corners)

    img_size = (img.shape[1], img.shape[0])
    # Do camera calibration given object points and image points
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img_size, None,None)
    #undst = cv2.undistort(img, mtx, dist, None, mtx)
    # return camera matrix
    return mtx, dist

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

def mag_thresh(img, sobel_kernel=3, mag_thresh=(0, 255)):
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

def dir_threshold(img, sobel_kernel=3, thresh=(0, np.pi/2)):
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
    # Convert color space from RGB to HLS
    # Apply threshold value for s channel
    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    s_channel = hls[:,:,2]
    binary_output = np.zeros_like(s_channel)
    binary_output[(s_channel > thresh[0]) & (s_channel <= thresh[1])] = 1
    return binary_output

def mask_image(img_binary):
    # create a masked image using cv2.fillPoly()
    # return ignore_mask_color for selected area
    mask = np.zeros_like(img_binary)
    ignore_mask_color = 1

    # define a four sided polygon to mask
    imshape = img.shape
    vertices = np.array([[(100,imshape[0]),(620, 400), (750, 400), (1200,imshape[0])]], dtype=np.int32)
    cv2.fillPoly(mask, vertices, ignore_mask_color)
    return mask

def perspective_trans(img, trans_direction ='camera2birdeye',  src =[[280, 675], [1040, 675], [909,590], [390, 596]] ,
                        dst  = [[280,675], [1040,675], [1040, 590], [280,590]]):
    # peform perspective transformation
    # option for trans_direction: camera2birdeye, birdeye2camera
    src = np.float32(src)
    dst = np.float32(dst)
    img_size = (img.shape[1], img.shape[0])
    # Given src and dst points, calculate the perspective transform matrix
    if trans_direction is 'camera2birdeye':
        M = cv2.getPerspectiveTransform(src, dst)
    elif trans_direction is 'birdeye2camera': # peform inverse transform
        M = cv2.getPerspectiveTransform(dst, src)
    # Warp the image using OpenCV warpPerspective()
    warped = cv2.warpPerspective(img, M, img_size)
    return warped


def sliding_window(binary_warped):
    # sliding window method to get lane points
    histogram = np.sum(binary_warped[binary_warped.shape[0]/2:,:], axis=0)
    # Create an output image to draw on and  visualize the result
    out_img = np.dstack((binary_warped, binary_warped, binary_warped))*255
    # Find the peak of the left and right halves of the histogram
    # These will be the starting point for the left and right lines
    midpoint = np.int(histogram.shape[0]/2)
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint

    # Choose the number of sliding windows
    nwindows = 9
    # Set height of windows
    window_height = np.int(binary_warped.shape[0]/nwindows)
    # Identify the x and y positions of all nonzero pixels in the image
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    # Current positions to be updated for each window
    leftx_current = leftx_base
    rightx_current = rightx_base
    # Set the width of the windows +/- margin
    margin = 100
    # Set minimum number of pixels found to recenter window
    minpix = 50
    # Create empty lists to receive left and right lane pixel indices
    left_lane_inds = []
    right_lane_inds = []

    # Step through the windows one by one
    for window in range(nwindows):
        # Identify window boundaries in x and y (and right and left)
        win_y_low = binary_warped.shape[0] - (window+1)*window_height
        win_y_high = binary_warped.shape[0] - window*window_height
        win_xleft_low = leftx_current - margin
        win_xleft_high = leftx_current + margin
        win_xright_low = rightx_current - margin
        win_xright_high = rightx_current + margin
        # Draw the windows on the visualization image
        cv2.rectangle(out_img,(win_xleft_low,win_y_low),(win_xleft_high,win_y_high),(1,255,1), 2)
        cv2.rectangle(out_img,(win_xright_low,win_y_low),(win_xright_high,win_y_high),(1,255,1), 2)
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
    left_lane_inds = np.concatenate(left_lane_inds)
    right_lane_inds = np.concatenate(right_lane_inds)

    # Extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds]
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]
    return leftx, lefty, rightx, righty, left_lane_inds, right_lane_inds, out_img

# if lane is detected in previous image, use margin search method
def margin_search(binary_warped, left_fit, right_fit):
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    margin = 100
    left_lane_inds = ((nonzerox > (left_fit[0]*(nonzeroy**2) + left_fit[1]*nonzeroy + left_fit[2] - margin)) & (nonzerox < (left_fit[0]*(nonzeroy**2) + left_fit[1]*nonzeroy + left_fit[2] + margin)))
    right_lane_inds = ((nonzerox > (right_fit[0]*(nonzeroy**2) + right_fit[1]*nonzeroy + right_fit[2] - margin)) & (nonzerox < (right_fit[0]*(nonzeroy**2) + right_fit[1]*nonzeroy + right_fit[2] + margin)))

    # Again, extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds]
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]
    return leftx, lefty, rightx, righty, left_lane_inds, right_lane_inds # #out_img

def curvarad(left_fit, right_fit, y_eval, ym_per_pix = 30/720, xm_per_pix = 3.7/700):
    # calculate lane curvature
    # ym_per_pix = 30/720 # meters per pixel in y dimension
    # xm_per_pix = 3.7/700 # meters per pixel in x dimension
    cov_vec = [xm_per_pix/ym_per_pix**2, xm_per_pix/ym_per_pix, xm_per_pix]
    left_fit_cr = np.multiply(cov_vec, left_fit)
    right_fit_cr = np.multiply(cov_vec, right_fit)
    middle_fit_cr = (left_fit_cr + right_fit_cr)/2
    # Calculate the new radii of curvature
    left_curverad = ((1 + (2*left_fit_cr[0]*y_eval*ym_per_pix + left_fit_cr[1])**2)**1.5) / np.absolute(2*left_fit_cr[0])
    right_curverad = ((1 + (2*right_fit_cr[0]*y_eval*ym_per_pix + right_fit_cr[1])**2)**1.5) / np.absolute(2*right_fit_cr[0])
    middle_curverad = ((1 + (2*middle_fit_cr[0]*y_eval*ym_per_pix + middle_fit_cr[1])**2)**1.5) / np.absolute(2*middle_fit_cr[0])
    #print([left_curverad, right_curverad, middle_curverad])

    return left_curverad,right_curverad, middle_curverad

def lane_center_offset(left_fitx, right_fitx, pix_hor, ym_per_pix = 30/720, xm_per_pix = 3.7/700):
    # calcuate lane center off based on image center point and lane center point
    # assume: camera is mounted in the center line of vehicle,
    # lane center is calcuated as the middle point of lane left line and right line
    veh_center_pix = np.int(left_fitx[-1]/2 + right_fitx[-1]/2) +1
    veh_offcent_pix = veh_center_pix - np.int(pix_hor/2)
    veh_offcenter = veh_offcent_pix* xm_per_pix
    '''
    if veh_offcenter >= 0:
        print('vehice is at left of lane center', veh_offcenter, 'm')
    else:
        print('vehice is at right of lane center', np.absolute(veh_offcenter), 'm')
    '''
    return veh_offcenter

def image_overlay(image_color, binary_warped, left_fitx, right_fitx):
    # overlay original image and detected lane area image
    warp_zero = np.zeros_like(binary_warped).astype(np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))
    ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0] )

    # Recast the x and y points into usable format for cv2.fillPoly()
    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    pts = np.hstack((pts_left, pts_right))

    # Draw the lane onto the warped blank image
    cv2.fillPoly(color_warp, np.int_([pts]), (0,255, 0))
    # Warp the blank back to original image space using inverse perspective matrix (Minv)
    newwarp = perspective_trans(color_warp, trans_direction ='birdeye2camera')
    # Combine the result with the original image
    result = cv2.addWeighted(image_color, 1, newwarp, 0.3, 0)

    return result


def txt_overlay(result, left_curverad, right_curverad, middle_curverad, vehicle_offcenter):
    # overlay txt messagine in image
    x_center = np.int(result.shape[1]/2)
    font = cv2.FONT_HERSHEY_SIMPLEX
    '''
    txt = 'Radius of left lane ' + '{0:.2f}'.format(left_curverad) + 'm'
    cv2.putText(result,txt, (50, 50), font, 1,(255,255,255),2)
    txt = 'Radius of right lane ' + '{0:.2f}'.format(right_curverad) + 'm'
    cv2.putText(result,txt,(50, 80), font, 1,(255,255,255),2)
    '''

    txt = 'Radius of lane ' + '{0:.2f}'.format(middle_curverad[-1]) + 'm'
    cv2.putText(result,txt,(50, 80), font, 1,(255,255,255),2)


    if vehicle_offcenter >=0:
        txt = 'vehicle is ' + '{0:.2f}'.format(vehicle_offcenter) + 'm' + ' right of center'
    else:
        txt = 'vehicle is ' + '{0:.2f}'.format(np.absolute(vehicle_offcenter)) + 'm' + ' left of center'

    cv2.putText(result,txt,(50, 110), font, 1,(255,255,255),2)


    return result

# define previous image featur detection property
class Line():
    def __init__(self):
        # was the line detected in the last iteration?
        self.detected = False
        # x values of the last n fits of the line
        self.recent_xfitted = []
        #average x values of the fitted line over the last n iterations
        self.bestx = None
        #polynomial coefficients averaged over the last n iterations
        self.best_fit = None
        #polynomial coefficients for the most recent fit
        self.current_fit = [np.array([False])]
        #radius of curvature of the line in some units
        self.radius_of_curvature = None # curvature at y = 0
        #distance in meters of vehicle center from the line
        self.line_base_pos = None
        #difference in fit coefficients between last and new fits
        self.diffs = np.array([0,0,0], dtype='float')
        #x values for detected line pixels
        self.allx = None
        #y values for detected line pixels
        self.ally = None

    # update image feature after detection is done
    def update(self, detected, current_fit, radius_of_curvature):
        self.detected = detected
        self.current_fit = current_fit
        self.radius_of_curvature = radius_of_curvature
        #self.best_fit = best_fit



def sanity_check(left_fit, right_fit, left_fit_last, right_fit_last, left_xfitpoint,
    right_xfitpoint, y0 = 0, yf = 719, ym_per_pix = 30/720, xm_per_pix = 3.7/700):
    # perform sanity check
    # checking criteria:curvarad change ration within a range; numer of fit points biggger than a threshold
    y_eval = np.array([y0, yf])
    left_curverad,right_curverad, middle_curverad = curvarad(left_fit, right_fit, y_eval, ym_per_pix = ym_per_pix, xm_per_pix = xm_per_pix)
    curvarad_curret = np.array([left_curverad, right_curverad]).T

    left_curverad_last,right_curverad_last, middle_curverad_last = curvarad(left_fit_last,
    right_fit_last, y_eval, ym_per_pix = 30/720, xm_per_pix = 3.7/700)
    # get reference curvarad of last images
    curvarad_lastdet = np.array([left_curverad_last, right_curverad_last]).T
    curvarad_change = np.divide(curvarad_curret, curvarad_lastdet)
    curvarad_check = (curvarad_change <10) & (curvarad_change > 0.1) & (curvarad_curret > 150)
    curvarad_check_left = curvarad_check[0,0] & curvarad_check[1,0]  #& (left_xfitpoint > 100)
    curvarad_check_right = curvarad_check[1,0] & curvarad_check[1,1] #& (right_xfitpoint > 100)

    middle_pix_last = (left_fit_last + right_fit_last)/2
    left_delta= middle_pix_last[-1] - left_fit[-1]
    right_delta= right_fit[-1] - middle_pix_last[-1]
    curvarad_check_left = curvarad_check_left & (left_delta*xm_per_pix>3.8/2*0.7)
    curvarad_check_right = curvarad_check_right & (right_delta*xm_per_pix> 3.8/2*0.7)


    #left_fitx = left_fit[0]*y_eval**2 + left_fit[1]*y_eval + left_fit[2]
    #left_fitx = right[0]*y_eval**2 + right_fit[1]*y_eval + right_fit[2]


    return curvarad_check_left, curvarad_check_right




'''
def sanity_check(leftx, lefty, rightx, righty, y0 = 0, yf = 719,ym_per_pix = 30/720, xm_per_pix = 3.7/700):

    left_fit_cr = np.polyfit(lefty*ym_per_pix, leftx*xm_per_pix, 2)
    right_fit_cr = np.polyfit(righty*ym_per_pix, rightx*xm_per_pix, 2)

    ploty = np.array([y0, yf*ym_per_pix/2, yf*ym_per_pix])
    #print(left_fit_cr, '\n', right_fit_cr)
    left_fitx = left_fit_cr[0]*ploty**2 + left_fit_cr[1]*ploty + left_fit_cr[2]
    right_fitx = right_fit_cr[0]*ploty**2 + right_fit_cr[1]*ploty + right_fit_cr[2]

    lane_width = right_fitx - left_fitx
    width_test = (lane_width < 3.7*1.2) & (lane_width > 3.7*0.9)
    cross_test = np.absolute (lane_width - lane_width[1]) < 3.7*0.1
    test       = width_test & cross_test
    print(test)

    if test[0]== True  & test[1]== True & test[2]== True:
        return True
    else:
        return False
'''


'''
def curvarad(leftx, lefty, rightx, righty, y_eval, ym_per_pix = 30/720, xm_per_pix = 3.7/700):
    # ym_per_pix = 30/720 # meters per pixel in y dimension
    # xm_per_pix = 3.7/700 # meters per pixel in x dimension
    left_fit_cr = np.polyfit(lefty*ym_per_pix, leftx*xm_per_pix, 2)
    right_fit_cr = np.polyfit(righty*ym_per_pix, rightx*xm_per_pix, 2)
    middle_fit_cr = (left_fit_cr + right_fit_cr)/2
    # Calculate the new radii of curvature
    left_curverad = ((1 + (2*left_fit_cr[0]*y_eval*ym_per_pix + left_fit_cr[1])**2)**1.5) / np.absolute(2*left_fit_cr[0])
    right_curverad = ((1 + (2*right_fit_cr[0]*y_eval*ym_per_pix + right_fit_cr[1])**2)**1.5) / np.absolute(2*right_fit_cr[0])
    middle_curverad = ((1 + (2*middle_fit_cr[0]*y_eval*ym_per_pix + middle_fit_cr[1])**2)**1.5) / np.absolute(2*middle_fit_cr[0])

    return left_curverad,right_curverad, middle_curverad
'''
