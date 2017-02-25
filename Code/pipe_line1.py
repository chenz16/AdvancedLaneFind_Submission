# import modules
import numpy as np
import cv2
import glob
import matplotlib.pyplot as plt
import pickle
from moviepy.editor import VideoFileClip

# import self defined modules
import visualization_save as vs
import configure as cfg
from process import camera_cal, abs_sobel_thresh, mag_thresh, dir_threshold, mask_image
from process import hls_select, perspective_trans, sliding_window, curvarad
from process import lane_center_offset, image_overlay, txt_overlay

test_pic = cfg.Source['test_images'] # folder storing test images
images = glob.glob(test_pic + '*.jpg')
mtx, dist = camera_cal()

def process_image(img):

    #img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # convert color from BGR to RGB
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

    image_warped =  perspective_trans(combined, trans_direction ='camera2birdeye')
    histogram = np.sum(image_warped[image_warped.shape[0]/2:,:], axis=0)
    out_img = np.dstack((image_warped*100, image_warped*255, image_warped*255))

    leftx, lefty, rightx, righty, left_lane_inds, right_lane_inds, out_img = sliding_window(image_warped)

    # Fit a second order polynomial to each

    y_eval = np.int(out_img.shape[0])-1 # curvature is evaulated @ point y_eval
    left_fit = np.polyfit(lefty, leftx, 2) # polyfit coefficients of lane left edge
    right_fit = np.polyfit(righty, rightx, 2) # polyfit coefficient of lane right edge

    # curve radius: lane left curve, lane right curve and middle curve
    left_curverad, right_curverad, middle_curverad = curvarad (leftx, lefty, rightx, righty, y_eval,
                                    ym_per_pix = 30/720, xm_per_pix = 3.7/700)



    '''
    left_fitx, right_fitx = vs.polyfit_visualize(out_img, image_warped, left_fit, right_fit,
                            left_lane_inds, right_lane_inds, save_indx = idx,
                            save_folder = cfg.Target['output_images'] + 'Poly_Fit/')
    '''
    # y pixle value
    ploty = np.linspace(0, out_img.shape[0]-1, out_img.shape[0] )

    # x pixle value of left curve and right curve
    left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
    right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]

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


'''
output = '../harder_challenge_DetLane.mp4'
clip = VideoFileClip('../harder_challenge_video.mp4')
LaneDet = clip.fl_image(process_image)
LaneDet.write_videofile(output, audio=False)
'''
