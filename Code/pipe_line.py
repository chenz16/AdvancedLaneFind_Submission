# import modules
import numpy as np
import cv2
import glob
import matplotlib.pyplot as plt
import pickle
from moviepy.editor import VideoFileClip
import pylab
import imageio


# import self defined modules
import visualization_save as vs
import configure as cfg
from process import camera_cal, abs_sobel_thresh, mag_thresh, dir_threshold, mask_image
from process import hls_select, perspective_trans, sliding_window, curvarad
from process import lane_center_offset, image_overlay, txt_overlay
from process import Line, sanity_check, margin_search, x_eval

test_pic = cfg.Source['test_images'] # folder storing test images
images = glob.glob(test_pic + '*.jpg')
mtx, dist = camera_cal()

left_line = Line() # define an object of left line of lane
right_line = Line() # define an object of right line of lane
Detected = False # assume two lines of a lane is detected

def process_image(img):

    #img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # convert color from BGR to RGB
    image_color = cv2.undistort(img, mtx, dist, None, mtx) # undistorted color image
    s_binary = hls_select(image_color, thresh=(80, 255)) # select color by s channel
    image = np.dstack((s_binary, s_binary, s_binary))

    ksize = 9 # sobel operation kernel
    # x direction (normalized) gradient within a range
    gradx = abs_sobel_thresh(image_color, orient='x', sobel_kernel=ksize, thresh=(50, 200))
    # y direction (normalized) gradient within a range
    grady = abs_sobel_thresh(image_color, orient='y', sobel_kernel=ksize, thresh=(50, 200))
    # gradient magnitude (normalized) within a range
    mag_binary = mag_thresh(image_color, sobel_kernel=ksize, mag_thresh=(100, 255))
    # (absolute) gradient direction magnitude within a range
    dir_binary = dir_threshold(image_color, sobel_kernel=ksize, thresh=(0.7, 1.2))
    # lane feature selection by combing different methods
    combined = np.zeros_like(dir_binary)
    combined[((gradx == 1) & (grady == 1)) | ((mag_binary == 1) & (dir_binary == 1))
                | (s_binary ==1)] = 1

    #mask = mask_image(combined) # optional
    # change to bird eye view by performing perspective transform
    image_warped =  perspective_trans(combined, trans_direction ='camera2birdeye')
    # get histogram of y for each grid of x
    histogram = np.sum(image_warped[image_warped.shape[0]/2:,:], axis=0)
    out_img = np.dstack((image_warped*100, image_warped*255, image_warped*255))

    #search lane points based on sliding widows
    leftx, lefty, rightx, righty, left_lane_inds, right_lane_inds, out_img = sliding_window(image_warped)

    # Fit a second order polynomial to each
    y_eval = np.array([0,out_img.shape[0]-1])
    left_fit = np.polyfit(lefty, leftx, 2) # polyfit coefficients of lane left edge
    right_fit = np.polyfit(righty, rightx, 2) # polyfit coefficients of lane right edge

    if left_line.detected or right_line.detected:
        # filter the coefficient of polynomial fit
        filter_coeff = 0.2
        left_fit = filter_coeff*left_line.current_fit + (1-filter_coeff)*left_fit
        right_fit = filter_coeff*right_line.current_fit + (1-filter_coeff)*right_fit


        # perform sanity check based on curvrad of current image and last image
        san_check_left, san_check_right = sanity_check(left_fit, right_fit,left_line.current_fit, right_line.current_fit, leftx.shape[0], rightx.shape[0],
                                                    y0 = 0, yf = 719, ym_per_pix = 30/720, xm_per_pix = 3.7/700)
        # lane width- fit_offset
        fit_offset_eval = np.absolute (x_eval(left_line.current_fit, y_eval) - x_eval(right_line.current_fit, y_eval))
        fit_offset     = np.mean(fit_offset_eval)

        # determin how to update the lane line based on sanity check results
        if (san_check_left & (not san_check_right)):
            # if right line is not detected, right line is defined as left line + lane width
            #print('right_false detected')
            right_fit = 0*right_line.current_fit + 1*(left_fit + np.array([0, 0, fit_offset]))
            #print('right_false detected')

        elif (not san_check_left) and san_check_right:
            #print('left_false detected')
            left_fit = 0*left_line.current_fit + 1*(right_fit - np.array([0, 0, fit_offset]))

        elif (not san_check_left) and (not san_check_right):
            #print('fit coeff keeps')
            right_fit = right_line.current_fit
            left_fit = left_line.current_fit
        else:
            pass

    # update curvarad based on poly fit
    left_curverad, right_curverad, middle_curverad=curvarad(left_fit, right_fit, y_eval,ym_per_pix =30/720,xm_per_pix =3.7/700)

    # update information of lane detection
    left_line.update(True, left_fit, middle_curverad[-1])
    right_line.update(True, right_fit, middle_curverad[-1])

    # y pixle value
    ploty = np.linspace(0, out_img.shape[0]-1, out_img.shape[0] )

    # x pixle value of left curve and right curve
    left_fitx = x_eval(left_fit, ploty)
    right_fitx = x_eval(right_fit, ploty)

    pix_hor = image_warped.shape[1]

    # vehicle off center
    vehicle_offcenter = lane_center_offset(left_fitx, right_fitx, pix_hor, ym_per_pix = 30/720, xm_per_pix = 3.7/700)

    # image overlay
    result = image_overlay(image_color, image_warped, left_fitx, right_fitx) # image_warped

    # text overlay
    result = txt_overlay(result, left_curverad, right_curverad,middle_curverad, vehicle_offcenter)

    return result


output = '../project_video_DetLane.mp4'
clip = VideoFileClip('../project_video.mp4')

LaneDet = clip.fl_image(process_image)
LaneDet.write_videofile(output, audio=False)
