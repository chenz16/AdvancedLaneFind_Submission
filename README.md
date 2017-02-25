
# AdvancedLaneFind_Submission

## 1. Submission packages
### Code folder:

Lane_find_img.py: script to process sample images for lane finding

pipeline.py -  pipe line to process video for lane finding

process.py -  Defines the core functions to process images for lane finding

visualze_save.py - visualize and save image

configuration.py - define basic input and output source/address of image process

writeup_report - Explain what is included for the submission and how it is done. 

### output_images folder

bird_eye_view: bird eye view (top to down) view of sample images 

camera_cal_output: camera calibration matrix and sample undistorted image

Feature_Selected_Image: images after features selected. The features include: s channel in HLS color space, gradient of image x, y direction, magnitude and direction of gradient 

Poly_Fit: plot of polynomial fit of those points which are identified as lane points 

Show_Lane_In_Image: overlay identified lane in original image (camera view) 

Undistorted_Image: undistorted image for image samples

### project_video_DetLane.mp4

project submission video which overlays identified lane with the original video



## 2. Go through rubric score

###Camera Calibration

####1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

The code for this step is contained in the function of camera_cal() in the ../Code/process.py  

I start by preparing "object points", which will be the (x, y, z) coordinates of the chessboard corners in the camera_cal folder. Assume each chessboard is in a flat plane therefore z is set as 0. Then the object points are the same for each calibration image.  Thus, `objp` is just a replicated array of coordinates, and `objpoints` will be appended with a copy of it every time I successfully detect all chessboard corners in a test image.  `imgpoints` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection.  

I then used the output `objpoints` and `imgpoints` to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function.  I applied this distortion correction to the test image using the `cv2.undistort()` function and obtained this result: 

![Alt text](/output_images/camera_cal_output/image_dist_undist.png)

###Pipeline (single images)

####1. Provide an example of a distortion-corrected image.
After obtaining the camera matrix from previous step, I then applied undistortion function  cv2.undistort to undistort the raw image. Here is an example of before and after image undistortion operation: 
![alt text](/output_images/Undistored_Image/0.jpg)
For more samples, please find through ![Here] (/output_images/Undistored_Image/)

####2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.

I used a combination of color and gradient thresholds to generate a binary image.

For color selection (see function 'hls_select' in ![process.py] (/Code/process.py) , I used the s channel of HLS color space by specifying threshold value thresh=(80, 255)  

For gradient selection, I used the x, y direction gradient (see function "abs_sobel_thresh" in ![process.py] (/Code/process.py)  ), magnitude of gradient (see "mag_thresh" in ![process.py] (/Code/process.py)  ), and direction of gradient (see function dir_threshold in ![process.py] (/Code/process.py) ).  Their threshold are shown as follows: 

    gradx = abs_sobel_thresh(image_color, orient='x', sobel_kernel=ksize, thresh=(50, 200))
    grady = abs_sobel_thresh(image_color, orient='y', sobel_kernel=ksize, thresh=(50, 200))
    mag_binary = mag_thresh(image_color, sobel_kernel=ksize, mag_thresh=(100, 255))
    dir_binary = dir_threshold(image_color, sobel_kernel=ksize, thresh=(0.7, 1.2))

Here's an example of my output for this step.  ![Alt image] (/output_images/Feature_Selected_Image/0.jpg)

####3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

The code for my perspective transform includes a function called `perspective_trans` defined in ![process.py] (/Code/process.py).  The `perspective_trans` function takes as inputs an image (`img`), as well as source (`src`) and destination (`dst`) points.  I chose the hardcode the source and destination points by my visual check of the straight line shown in the test images:

This resulted in the following source and destination points:

| Source        | Destination   | 
|:-------------:|:-------------:| 
| 280, 675      | 280, 675      | 
| 1040, 675     | 1040, 675     |
| 909, 590      | 1040, 590      |
| 390, 596      | 280, 590      |
I verified that my perspective transform was working as expected by drawing the `src` and `dst` points onto a test image and its warped counterpart to verify that the lines appear parallel in the warped image.
![Alt img](/out_images/transform/0.jpg)


####4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

I used the method called 'sliding_window' which was introduced in the course material to identify the lane-line pixel. The function is included in ![process.py] (/Code/process.py)). Afer the lane points are available, they are used to fit a second order of polynomial:

    left_fit = np.polyfit(lefty, leftx, 2) # polyfit coefficients of lane left edge
    right_fit = np.polyfit(righty, rightx, 2) # polyfit coefficients of lane right edge
    
An sample image of lane points indentification and polynominal fit is shown:
![alt img] (output_images/Poly_Fit/0.jpg)


####5 Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

The radius of curvature of the lane is calcuated through function 'curvarad' in ![process.py] (/Code/process.py).
the position of the vehicle with respect to center is calcuated through function 'lane_center_offset' in ![process.py] (/Code/process.py). The assumption here is the camera is mounted in the center fron vehicle. The lane center is identified as the middle of left and right line of a lane. 

####6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

I implemented this step in lines # through # in my code in yet_another_file.py in the function map_lane(). Here is an example of my result on a test image:

![alt img] (/output_images/Show_Lane_In_Image/0.jpg). For more image, please go to ![Here](/output_images/Show_Lane_In_Image/)


####6 Pipeline (video)
Here's a link to my video result

###Discussion

Briefly discuss any problems / issues you faced in your implementation of this project. Where will your pipeline likely fail? What could you do to make it more robust?

The pipeline works well for most of images in the vedio except a few of them. There are several causes for the error detection
    1. background noise: a. uneven shadow area where the tree shadow to the ground is uneven and changes fast b. other noises which may be identified as lane points but actually not
    2. insufficient or missing lane mark points which causes the polynomial fit incorrect
    3. curvature change: this particularly is an issue for the challenge video where the curvature of lane changes dynamically

To isolate error detections, i added sanity check function(sanity_check in ![alt txt] (/Code/process.py)) to check if the polynomial fit makes sense. Several criteria were considered:
    1. radius of curvature: compare the radius of recently indentified lane lines with the one in previous step. If it out of range, the sanity check return false, meaning the lane line detection is not effective/valid. 
    2. Absolute radius of curvature: this is a little tricky. In general situation, we do not know what's the range of radius of curvature. For this submission, we pick a value which provides a good detection result with a visual check. 
    3. compare the shift between the new identified lane lines with the lane center identified previously. The shift should be within a range. If not, the new identified lane is treated as invalid/ineffective.  

Sanity check function returns the True or False of left and right line detection effectiveness. 

In pipe_line.py, I defines handling methods for different types of line detection errors. If both lines are faulted, keep previous lines; if one of lines are faulted, use another line plus a lane width shift.  Here is the code extracted from pipe_line.py:

        if (san_check_left & (not san_check_right)):
            right_fit = 0*right_line.current_fit + 1*(left_fit + np.array([0, 0, fit_offset]))

        elif (not san_check_left) and san_check_right:
            left_fit = 0*left_line.current_fit + 1*(right_fit - np.array([0, 0, fit_offset]))

        elif (not san_check_left) and (not san_check_right):
            right_fit = right_line.current_fit
            left_fit = left_line.current_fit
        else:
            pass

