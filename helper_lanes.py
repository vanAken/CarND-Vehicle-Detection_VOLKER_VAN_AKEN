import numpy as np
import cv2

def dir_s_threshold(image, sobel_kernel=15, thresh=(0.7, 1.2)):
    # s-CHANEL from i-net
    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    # Calculate the x and y gradients
    sobelx = cv2.Sobel(hls[:,:,2], cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(hls[:,:,2], cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    # Take the absolute value of the gradient direction, 
    absgraddir = np.arctan2(np.absolute(sobely), np.absolute(sobelx))
    # Return the binary image
    binary_output =  np.zeros_like(absgraddir)
    binary_output[(absgraddir >= thresh[0]) & (absgraddir <= thresh[1])] = 1
    return binary_output

def abs_sobel_thresh(img, orient="x", sobel_kernel=3, thresh=(0,255)):
    # Grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # process x or y
    if orient == "x":
        abs_sobel = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel))
    if orient == "y":
        abs_sobel = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel))
    
    scaled_sobel = np.uint8(255* abs_sobel/ np.max(abs_sobel))
    # Return the binary image
    binary_output =  np.zeros_like(scaled_sobel)
    binary_output[(scaled_sobel >= thresh[0]) & (scaled_sobel <= thresh[1])] = 1
    return binary_output

def mag_thresh(img, sobel_kernel=3, thresh=(0, 255)):
    # Grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # Calculate the x and y gradients
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    gradmag = np.sqrt(sobelx**2+sobely**2)
    
    scaled_gradmag = np.uint8(255* gradmag/np.max(gradmag))
    # Return the binary image
    binary_output =  np.zeros_like(scaled_gradmag)
    binary_output[(scaled_gradmag >= thresh[0]) & (scaled_gradmag <= thresh[1])] = 1
    return binary_output

def dir_thresh(image, sobel_kernel=15, thresh=(0,np.pi/2)):
    # GRAY
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2gray)
    # Calculate the x and y gradients
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    # Take the absolute value of the gradient direction, 
    absgraddir = np.arctan2(np.absolute(sobely), np.absolute(sobelx))
    # Return the binary image
    binary_output =  np.zeros_like(absgraddir)
    binary_output[(absgraddir >= thresh[0]) & (absgraddir <= thresh[1])] = 1
    return binary_output

def color_thresh(img, sthresh=(0, 255), vthresh=(0, 255)):
    # S-CHANNEL
    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    s_channel = hls[:,:,2]
    s_binary_output =  np.zeros_like(s_channel)
    s_binary_output[(s_channel >= sthresh[0]) & (s_channel <= sthresh[1])] = 1
    # V-CHANNEL
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    v_channel = hsv[:,:,2]
    v_binary_output =  np.zeros_like(v_channel)
    v_binary_output[(v_channel >= vthresh[0]) & (v_channel <= vthresh[1])] = 1
    # Combine the two binary thresholds 
    combined_binary = np.zeros_like(s_channel)
    combined_binary[(s_binary_output == 1) & (v_binary_output == 1)] = 1
    return combined_binary 

def binarize(img_undistorted):
    # binarize the frame s.t. lane lines are highlighted as much as possible  
    img_binary = np.zeros_like(img_undistorted[:,:,0])
    gradx = abs_sobel_thresh(img_undistorted, orient='x', thresh=(12,255)) #12  
    grady = abs_sobel_thresh(img_undistorted, orient='y', thresh=(25,255)) #25
    c_binary =  color_thresh(img_undistorted,sthresh=(100,255), vthresh=(50,255))
    img_binary [((gradx == 1) & (grady == 1) | (c_binary == 1))] = 255	 
    return img_binary

def warper(img):
    # warp the image into birdviw
    bot_width = .76  # percent of bottom trapizoid height
    mid_width = .08  # percent of middle trapizoid height
    hight_pct = .62  # percent for trapizoid height
    bott_trim = .935 # percent from top to bottom to avoid the car hood
    
    src = np.float32([[img.shape[1]*(.5-mid_width/2),img.shape[0]*hight_pct],[img.shape[1]*(.5+mid_width/2),img.shape[0]*hight_pct],
                      [img.shape[1]*(.5+bot_width/2),img.shape[0]*bott_trim],[img.shape[1]*(.5-bot_width/2),img.shape[0]*bott_trim] ])

    left_off_pct  = 1/8 # part of left cut
    right_off_pct = 1/4 # part of right cut

    dst = np.float32([[img.shape[1]*left_off_pct     ,0                     ],[img.shape[1]*(1-right_off_pct),0                     ],
                      [img.shape[1]*(1-right_off_pct),img.shape[0]          ],[img.shape[1]*left_off_pct     ,img.shape[0]          ] ])
    
    M     = cv2.getPerspectiveTransform(src,dst) 
    M_inv = cv2.getPerspectiveTransform(dst,src) 
    warped= cv2.warpPerspective(img, M, (img.shape[1],img.shape[0]),flags=cv2.INTER_LINEAR)

    return warped , M , M_inv


def nice_output(road_lanes, img_left, img_right, curve_rad, center_diff, index):
    """
    Prepare the final pretty pretty output blend, given all intermediate pipeline images
    :param blend_on_road: color image of lane blend onto the road
    :param img_left: print in the left corner
    :param img_right: print in the right corner
    :param center_diff: differenz between the middle of the image, the car position to the middle of both lanes
    :return: nice output with all images and stuff stitched
    """
    h, w = road_lanes.shape[:2]

    thumb_ratio = 0.2
    thumb_h, thumb_w = int(thumb_ratio * h), int(thumb_ratio * w)

    off_x, off_y = 20, 15

    # add a gray rectangle to highlight the upper area
    mask = road_lanes.copy()
    mask = cv2.rectangle(mask, pt1=(0, 0), pt2=(w, thumb_h+2*off_y), color=(0, 0, 0), thickness=cv2.FILLED)
    road_lanes = cv2.addWeighted(src1=mask, alpha=0.2, src2=road_lanes, beta=0.8, gamma=0)

    # add thumbnail of img_left
    thumb_img_left = cv2.resize(img_left, dsize=(thumb_w, thumb_h))
    #thumb_pre = np.dstack([thumb_pre, thumb_pre, thumb_pre]) 
    road_lanes[off_y:thumb_h+off_y, off_x:off_x+thumb_w, :] = thumb_img_left

    # add thumbnail of img_right
    thumb_img_right = cv2.resize(img_right, dsize=(thumb_w, thumb_h))
    road_lanes[off_y:thumb_h+off_y, road_lanes.shape[1]-off_x-thumb_w:road_lanes.shape[1]-off_x, :] = thumb_img_right

    # limit of 9999m curvature 
    if  curve_rad > 9999.:
        curve_rad = 9999.
    curve_str = "Radius of curvature = {0:5.0f} m.".format(curve_rad)
    
    # decide for left and right side of the lane therefor take away the signum from center_diff
    side_pos = ' left'
    if center_diff <=0:
        side_pos = 'right'   
    center_diff = abs(center_diff)
    center_str = "Vehicel is {0:4.3f} m {1:} of center.".format(center_diff, side_pos)
    
    # format the indexline for framecounting
    index_str = "Index:  %5d "%index

    # write the stings on the Image    
    cv2.putText(road_lanes, curve_str , (300, 45) ,cv2.FONT_HERSHEY_SIMPLEX,0.9,(255, 255, 255),2,cv2.LINE_AA)
    cv2.putText(road_lanes, center_str, (300, 95) ,cv2.FONT_HERSHEY_SIMPLEX,0.9,(255, 255, 255),2,cv2.LINE_AA)      
    cv2.putText(road_lanes, index_str ,(300, 145) ,cv2.FONT_HERSHEY_SIMPLEX,0.9,(255, 255, 255),2,cv2.LINE_AA)
    return road_lanes

