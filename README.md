# AdvancedLaneFind_Submission

## 1. Submission packages
### Code folder:

Lane_find_img.py: script to process sample images for lane finding

pipeline.py -  pipe line to process vedio for lane finding

process.py -  Defines the core functions to process images for lane finding

visualze_save.py - visualize and save image

configuration.py - define basic input and output source/address of image process

writeup_report - Explain what is included for the submission and how it is done. 

### output_images folder

bird_eye_view: bird eye view (top to down) view of sample images 

camera_cal_output: camera calibratioin matrix and sample undistorted image

Feature_Selected_Image: images after features selected. The features include: s channel in HLS color space, gradient of image x, y direction, magnitude and direction of gradient 

Poly_Fit: plot of polynomial fit of those points which are identified as lane points 

Show_Lane_In_Image: overlay indentified lane in origal image (camera view) 

Undistored_Image: undistored image for image samples

### porject_video_DetLane.mp4

Overlay identified lane in project video



## 2. Go through rubic score

###Camera Calibration

####1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

The code for this step is contained in the function of camera_cal() in the ../Code/process.py  

I start by preparing "object points", which will be the (x, y, z) coordinates of the chessboard corners in the camera_cal folder. Assume each chessboard is in a flat plane therfore z is set as 0. Then the object points are the same for each calibration image.  Thus, `objp` is just a replicated array of coordinates, and `objpoints` will be appended with a copy of it every time I successfully detect all chessboard corners in a test image.  `imgpoints` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection.  

I then used the output `objpoints` and `imgpoints` to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function.  I applied this distortion correction to the test image using the `cv2.undistort()` function and obtained this result: 

![Alt text](/output_images/camera_cal_output/image_dist_undist.png)

###Pipeline (single images)

####1. Provide an example of a distortion-corrected image.
After obtaining the camera matrix from previous step, I then applied undistortion function  cv2.undistort to undistore the raw image. Here is an example of before and after image undistortion operation: 
![alt text](/output_images/Undistored_Image/0.jpg)
For more samples, please find through ![Here] (/output_images/Undistored_Image/)

####2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.
I used a combination of color and gradient thresholds to generate a binary image (thresholding steps at lines # through # in `another_file.py`).  Here's an example of my output for this step.  (note: this is not actually from one of the test images)

![alt text][image3]

####3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

The code for my perspective transform includes a function called `warper()`, which appears in lines 1 through 8 in the file `example.py` (output_images/examples/example.py) (or, for example, in the 3rd code cell of the IPython notebook).  The `warper()` function takes as inputs an image (`img`), as well as source (`src`) and destination (`dst`) points.  I chose the hardcode the source and destination points in the following manner:

```
src = np.float32(
    [[(img_size[0] / 2) - 55, img_size[1] / 2 + 100],
    [((img_size[0] / 6) - 10), img_size[1]],
    [(img_size[0] * 5 / 6) + 60, img_size[1]],
    [(img_size[0] / 2 + 55), img_size[1] / 2 + 100]])
dst = np.float32(
    [[(img_size[0] / 4), 0],
    [(img_size[0] / 4), img_size[1]],
    [(img_size[0] * 3 / 4), img_size[1]],
    [(img_size[0] * 3 / 4), 0]])

```
This resulted in the following source and destination points:

| Source        | Destination   | 
|:-------------:|:-------------:| 
| 585, 460      | 320, 0        | 
| 203, 720      | 320, 720      |
| 1127, 720     | 960, 720      |
| 695, 460      | 960, 0        |

I verified that my perspective transform was working as expected by drawing the `src` and `dst` points onto a test image and its warped counterpart to verify that the lines appear parallel in the warped image.

![alt text][image4]
