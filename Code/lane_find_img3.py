# import modules
import numpy as np
import cv2
import glob
import matplotlib.pyplot as plt
import pickle

# import self defined modules
import visualization_save as vs
import configure as cfg
from process import camera_cal, abs_sobel_thresh, mag_thresh, dir_threshold, mask_image
from process import hls_select, perspective_trans, sliding_window, curvarad
from process import lane_center_offset, image_overlay, txt_overlay
from process import Line, sanity_check, margin_search

test_pic = cfg.Source['test_images'] # folder storing test images
images = glob.glob(test_pic + '*.jpg')
mtx, dist = camera_cal() # get camera matrix

left_line = Line() # define an object of left line of lane
right_line = Line() # define an object of right line of lane


for idx, fname in enumerate(images):
    img = cv2.imread(fname) # read image
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # convert color from BGR to RGB
    image_color = cv2.undistort(img, mtx, dist, None, mtx) # undistorted color image

    s_binary = hls_select(image_color, thresh=(100, 255)) # select color by s channel
    image = np.dstack((s_binary, s_binary, s_binary))

    ksize = 9 # sobel operation kernel
    # x direction (normalized) gradient within a range
    gradx = abs_sobel_thresh(image_color, orient='x', sobel_kernel=ksize, thresh=(50, 200))
    # y direction (normalized) gradient within a range
    grady = abs_sobel_thresh(image_color, orient='y', sobel_kernel=ksize, thresh=(50, 200))
    # gradient magnitude (normalized) within a range
    mag_binary = mag_thresh(image_color, sobel_kernel=ksize, mag_thresh=(80, 255))
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

    # plot base image and different types of processed images in one figure and save it
    vs.visualize_and_save(img, image_color, save_indx = idx, base_title = 'Raw Image',
    new_title = 'Undistored Image', save_folder = cfg.Target['output_images'] + 'Undistored_Image/')

    vs.visualize_and_save(image_color, combined, save_indx = idx, base_title = 'Undistored Image',
    new_title = 'Feature-Selected Image', save_folder = cfg.Target['output_images'] + 'Feature_Selected_Image/')

    vs.visualize_and_save(combined, image_warped, save_indx = idx, base_title = 'Feature-Selected Image',
    new_title = 'Bird Eye View', save_folder = cfg.Target['output_images'] + 'bird_eye_view/')

    # search lane points based on sliding widows
    leftx, lefty, rightx, righty, left_lane_inds, right_lane_inds, out_img = sliding_window(image_warped)
    '''
    if left_line.detected == False and right_line.detected == False:
        # search lane points based on sliding widows
        leftx, lefty, rightx, righty, left_lane_inds, right_lane_inds, out_img = sliding_window(image_warped)
    else:
        left_fit = left_line.current_fit
        right_fit = right_line.current_fit
        # search lane points based on sliding widows
        leftx, lefty, rightx, righty, left_lane_inds, right_lane_inds = margin_search(image_warped, left_fit, right_fit)
    '''
    # Fit a second order polynomial to each
    y_eval = np.array([0,out_img.shape[0]-1])
    left_fit = np.polyfit(lefty, leftx, 2) # polyfit coefficients of lane left edge
    right_fit = np.polyfit(righty, rightx, 2) # polyfit coefficients of lane right edge

    # perform sanity check
    Detected = True # assume two lines of a lane is detected
    if left_line.detected == True or right_line.detected == True:

        left_curv_lastdet = [left_line.radius_of_curvature_0, left_line.radius_of_curvature_yf]
        right_curv_lastdet = [right_line.radius_of_curvature_0, right_line.radius_of_curvature_yf]
        # get reference curvarad of last images
        curvarad_lastdet = np.array([left_curv_lastdet, right_curv_lastdet]).T

        # perform sanity check based on curvrad of current image and last image
        curvarad_check_left, curvarad_check_right = sanity_check(left_fit, right_fit,curvarad_lastdet,
                                                    y0 = 0, yf = 719, ym_per_pix = 30/720, xm_per_pix = 3.7/700)

        # determin how to update the lane line based on sanity check result
        if curvarad_check_left==True and curvarad_check_left==False:
        # if right line is not detected, gadually adpat to left line + lane width
            right_fit = 0.95*right.line.current_fit + 0.05*(left_fit + right.line.current_fit[-1] -left.line.current_fit[-1])

        elif curvarad_check_left == False and curvarad_check_left == True:
            left_fit = 0.95*left.line.current_fit + 0.05*(right_fit + left.line.current_fit[-1] -right.line.current_fit[-1])

        elif curvarad_check_left == False and curvarad_check_left == False:
            right_fit = right_line.current_fit
            left_fit = left_line.current_fit
            Detected = False # lane detection failed
        else:
            pass

    # update curvarad based on poly fit
    left_curverad, right_curverad, middle_curverad=curvarad(left_fit, right_fit, y_eval,ym_per_pix =30/720,xm_per_pix =3.7/700)

    # update information of lane detection
    left_line.update(Detected, left_fit, left_curverad[0], left_curverad[-1])
    right_line.update(Detected, right_fit, right_curverad[0], right_curverad[-1])

    # Now our radius of curvature is in meters
    # visualize

    left_fitx, right_fitx = vs.polyfit_visualize(out_img, image_warped, left_fit, right_fit,
                            left_lane_inds, right_lane_inds, save_indx = idx,
                            save_folder = cfg.Target['output_images'] + 'Poly_Fit/')

    pix_hor = image_warped.shape[1]
    vehicle_offcenter = lane_center_offset(left_fitx, right_fitx, pix_hor, ym_per_pix = 30/720, xm_per_pix = 3.7/700)

    result = image_overlay(image_color, image_warped, left_fitx, right_fitx) # image_warped
    result = txt_overlay(result, left_curverad, right_curverad, middle_curverad, vehicle_offcenter)

    vs.visualize_and_save(image_color, result, save_indx = idx, base_title = 'Undistored Image',
                        new_title = 'Detected Lane', save_folder = cfg.Target['output_images'] + 'Show_Lane_In_Image/' )
