import numpy as np
import cv2
import glob
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import pickle

# import self defined modules
import visualization_save as vs
import configure as cfg
from process import camera_cal, abs_sobel_thresh, mag_thresh, dir_threshold
from process import hls_select, perspective_trans, sliding_window, curvature


test_pic = cfg.Source['test_images'] # folder storing test images
images = glob.glob(test_pic + '*.jpg')

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
print(images)

# Step through the list and search for chessboard corners
for idx, fname in enumerate(images):
    print(nx, ny)
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Find the chessboard corners
    ret, corners = cv2.findChessboardCorners(gray, (nx,ny), None)
    # If found, add object points, image points
    if ret == True:
        objpoints.append(objp)
        imgpoints.append(corners)

        cv2.drawChessboardCorners(img, (nx,ny), corners, ret)
        #write_name = '../camera_cal/Drawed/corners_found'+str(idx)+'.jpg'
        #cv2.imwrite(write_name, img)
        #cv2.imshow('img', img)
        #cv2.waitKey(1000)
        plt.imshow(img)
        plt.show()

img_size = (img.shape[1], img.shape[0])
print(img_size)
# Do camera calibration given object points and image points
#ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img_size, None,None)
