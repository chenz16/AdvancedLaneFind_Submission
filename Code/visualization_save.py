'''
this module is for data visulization and saving
camera_cal_visu: display and save the chess board raw images and images after undistortion
camera_cal_save: save camera calibration matrix
visualize_and_save: save two images in one figure as a comparison before and after change
polyfit_visualize: visualize detected lanes
'''


import numpy as np
import cv2
import glob
import matplotlib.pyplot as plt
import pickle
import configure as cfg


def camera_cal_visu(mtx, dist):

    nx = cfg.Source['image_size'][0]
    ny = cfg.Source['image_size'][1]
    # folder to store calibration image
    chessboard_pic = cfg.Souce['chessboard']
    images = glob.glob(chessboard_pic + 'calibration*.jpg')

    # Step through the list and search for chessboard corners
    for idx, fname in enumerate(images):
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Find the chessboard corners
        ret, corners = cv2.findChessboardCorners(gray, (nx,ny), None)

        #visualize all the calibration image
        if ret == True:
            cv2.drawChessboardCorners(img, (nx,ny), corners, ret)
            write_name = cfg.Target['chessboard_draw'] + 'corners_found'+ str(idx)+'.jpg'
            cv2.imwrite(write_name, img)
            cv2.imshow('img', img)
            cv2.waitKey(100)

    cv2.destroyAllWindows()


    # Test undistortion on a sample image
    img = cv2.imread(cfg.Source['undist_sampe'] + 'calibration5.jpg')
    img_size = (img.shape[1], img.shape[0])
    # Do camera calibration given object points and image points
    undst = cv2.undistort(img, mtx, dist, None, mtx)

    f,(ax1, ax2) = plt.subplots(1, 2, figsize=(20,10))
    f.tight_layout()
    ax1.imshow(img, cmap = 'gray')
    ax1.set_title('Original Image', fontsize=30)
    ax2.imshow(undst)
    ax2.set_title('Undistorted Image', fontsize=30)
    plt.pause(1)

    # save figure to target place
    write_name = '../test_images/transform2/'
    f.savefig(write_name + str(idx) + '.jpg')
    plt.close('all')

def camera_cal_save(mtx, dist):
    dist_pickle = {}
    dist_pickle["mtx"] = mtx
    dist_pickle["dist"] = dist
    pickle.dump( dist_pickle, open( "../output_images/camera_cal_output/wide_dist_pickle.p", "wb" ) )
    # Save the camera calibration result for later use (we won't worry about rvecs / tvecs)


def visualize_and_save(image_base, image_new, save_indx = 0, base_title = 'Original Image',
    new_title = 'new image', save_folder = '../output_images/'):

    f,(ax1, ax2) = plt.subplots(1, 2, figsize=(20,10))
    f.tight_layout()
    image_base_shape     = image_base.shape
    image_new_shape      = image_new.shape

    if len(image_base_shape) == 2:
        ax1.imshow(image_base, cmap = 'gray')
    else:
        #assert(len(image_base_shape)==3, 'error input')
        ax1.imshow(image_base)

    if len(image_new_shape) == 2:
        ax2.imshow(image_new, cmap = 'gray')
    else:
        #assert(len(image_base_shape)==3, 'error input')
        ax2.imshow(image_new)

    ax1.set_title(base_title, fontsize=30)
    ax2.set_title(new_title, fontsize=30)
    f.savefig(save_folder + str(save_indx) + '.jpg')
    #plt.show()
    #plt.pause(0.5)
    plt.close('all')
    #plt.show()


def polyfit_visualize(out_img, binary_warped, left_fit, right_fit,
                        left_lane_inds, right_lane_inds,
                         save_indx = 0, save_folder = '../output_images/Poly_Fit/'):

    ploty = np.linspace(0, out_img.shape[0]-1, out_img.shape[0] )
    left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
    right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]

    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])

    out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [1, 255, 100]
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
    f.savefig(save_folder + str(save_indx) + '.jpg')
    plt.close('all')
    return left_fitx, right_fitx


def margin_search_visualize(binary_warped, left_fit, right_fit,
                        left_lane_inds, right_lane_inds,
                         save_indx = 0, save_folder = '../output_images/Poly_Fit/'):

    ploty = np.linspace(0, out_img.shape[0]-1, out_img.shape[0])
    # Create an image to draw on and an image to show the selection window
    out_img = np.dstack((binary_warped, binary_warped, binary_warped))*255
    window_img = np.zeros_like(out_img)
    # Color in left and right line pixels
    out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
    out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]

    # Generate a polygon to illustrate the search window area
    # And recast the x and y points into usable format for cv2.fillPoly()
    left_line_window1 = np.array([np.transpose(np.vstack([left_fitx-margin, ploty]))])
    left_line_window2 = np.array([np.flipud(np.transpose(np.vstack([left_fitx+margin, ploty])))])
    left_line_pts = np.hstack((left_line_window1, left_line_window2))
    right_line_window1 = np.array([np.transpose(np.vstack([right_fitx-margin, ploty]))])
    right_line_window2 = np.array([np.flipud(np.transpose(np.vstack([right_fitx+margin, ploty])))])
    right_line_pts = np.hstack((right_line_window1, right_line_window2))

    # Draw the lane onto the warped blank image
    cv2.fillPoly(window_img, np.int_([left_line_pts]), (0,255, 0))
    cv2.fillPoly(window_img, np.int_([right_line_pts]), (0,255, 0))
    result = cv2.addWeighted(out_img, 1, window_img, 0.3, 0)
    plt.imshow(result)
    plt.plot(left_fitx, ploty, color='yellow')
    plt.plot(right_fitx, ploty, color='yellow')
    plt.xlim(0, 1280)
    plt.ylim(720, 0)
    #plt.pause(1)
    f.savefig(save_folder + str(save_indx) + '.jpg')
    plt.close('all')
    return left_fitx, right_fitx
