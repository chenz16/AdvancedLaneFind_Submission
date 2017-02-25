'''
this module is for configuration
this module defines:
image sources for camera calibration
image sources for algorithm testing
vedio sources
target folder to store all processed image/videos
'''


Source= {}
Source['chessboard'] = '../camera_cal/' # folder to store chess board images
Source['undist_sampe'] = '../camera_cal/' # undistort sample image in this folder to show how camera cal works
Source['image_size'] = (9, 6) # (nx, ny) # chess board nx, ny
Source['test_images'] = '../test_images/' # test images for lane finding

Target = {}
Target['chessboard_draw'] = '../camera_cal/Drawed)' # place to store the undistored chess board images
Target['camera_matrx'] ='../out_images/camera_cal_output/' # place to save cal calibration
Target['output_images'] = '../output_images/' # place to store all processed images
