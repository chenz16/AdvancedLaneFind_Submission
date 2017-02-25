import numpy as np
import cv2
import glob
import matplotlib.pyplot as plt
import pickle


def camera_cal():
    nx = 9
    ny = 6

    objp = np.zeros((nx*ny,3), np.float32) # 3- x,y,z coord, 6*8 = number of test points
    objp[:,:2] = np.mgrid[0:nx, 0:ny].T.reshape(-1,2) # define x, y, and keep z as zero

    # Arrays to store object points and image points from all the images.
    objpoints = [] # 3d points in real world space
    imgpoints = [] # 2d points in image plane.

    # Make a list of calibration images
    images = glob.glob('../camera_cal/calibration*.jpg')

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

            # Draw and display the corners
            '''
            cv2.drawChessboardCorners(img, (nx,ny), corners, ret)
            write_name = '../camera_cal/Drawed/corners_found'+str(idx)+'.jpg'
            cv2.imwrite(write_name, img)
            cv2.imshow('img', img)
            cv2.waitKey(100)

    cv2.destroyAllWindows()
    # Test undistortion on an image
    img = cv2.imread('../camera_cal/calibration5.jpg')'''
    img_size = (img.shape[1], img.shape[0])
    # Do camera calibration given object points and image points
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img_size, None,None)
    undst = cv2.undistort(img, mtx, dist, None, mtx)


    # Save the camera calibration result for later use (we won't worry about rvecs / tvecs)
    dist_pickle = {}
    dist_pickle["mtx"] = mtx
    dist_pickle["dist"] = dist
    pickle.dump( dist_pickle, open( "../output_images/camera_cal_output/wide_dist_pickle.p", "wb" ) )
    #dst = cv2.cvtColor(dst, cv2.COLOR_BGR2RGB)
    # Visualize undistortion
    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(20,10))
    f.tight_layout()
    ax1.imshow(img)
    ax1.set_title('Original Image', fontsize=30)
    ax2.imshow(undst)
    ax2.set_title('Undistorted Image', fontsize=30)
    write_name = '../output_images/camera_cal_output/image_dist_undist'
    f.savefig(write_name)
    plt.close('all')
    return mtx, dist
