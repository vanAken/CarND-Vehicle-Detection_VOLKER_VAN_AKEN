import numpy as np
import cv2
class tracker():
    # when starting a new instance please be shure to spacify all unassigned variables
    def __init__(self, Mywindow_width, Mywindow_height, Mymargin, My_ym=1, My_xm=1, Mysmooth_factor=10):
        # list that stores all the past (left,right) canter set values used for smoothing the output
        self.recent_centers =[]
        # window pixel width of the center values, used to count pixels inside denter windows to dertimine curve values
        self.window_width = Mywindow_width
        # window pixel height of the center values, used to count pixels inside denter windows to dertimine curve values
        # breaks the window in vertical levels
        self.window_height = Mywindow_height 
        # pixel distance in both directions to slide (left_window + right_window)  template for searching
        self.margin = Mymargin
        self.ym_per_pix = My_ym # meters per pixel in vertical axis
        self.xm_per_pix = My_xm # meters per pixel in horizontal axis
        self.smooth_factor = Mysmooth_factor
        self.margin = Mymargin
    # main tracking function for finding and storing lane segment positions
    def find_window_centroids(self, warped):

        window_width  = self.window_width
        window_height = self.window_height
        margin        = self.margin

        window_centroids = [] # store the (left,right) window centroid position per level
        window = np.ones(window_width) # create our window template that will use for convolution
        # First find the two starting positions for the left and the right lane by using np.sum to get the vertical image slice and than
        # np.convolve the vertical image slice with the 'window' template

        # Sum quarter bottom of image to get slice, could use a differnt ratio 
        l_sum = np.sum( warped[int(3*warped.shape[0]/4):,:int(warped.shape[1]/2)], axis=0 )
        l_center = np.argmax(np.convolve(window,l_sum))-window_width/2
        r_sum = np.sum(warped[int(3*warped.shape[0]/4):,int(warped.shape[1]/2):], axis=0)
        r_center = np.argmax(np.convolve(window,r_sum))-window_width/2+int(warped.shape[1]/2)

        # add what we found for the first layer
        window_centroids.append((l_center, r_center))

        # Go throught each layer looking for max pixel locations
        for level in range(1,(int)(warped.shape[0]/window_height)):
            # convolute the window into the vertical slice of the image
            image_layer = np.sum(warped[int(warped.shape[0]-(level+1)*window_height):int(warped.shape[0]-level*window_height),:], axis=0)
            conv_signal = np.convolve(window,image_layer)
            # Find the best left centroid by using past left center as a reference 
            # Use window_width/2 as offset, because convolution signal reference is at right side of the window, not center of window
            offset = window_width/2
            l_min_index = int(max(l_center+offset-margin,0))
            l_max_index = int(min(l_center+offset+margin,warped.shape[1]))
            l_center    = np.argmax(conv_signal[l_min_index:l_max_index])+l_min_index-offset

            # Find the best left centroid by using past left center as a reference 
            r_min_index = int(max(r_center+offset-margin,0))
            r_max_index = int(min(r_center+offset+margin,warped.shape[1]))
            r_center    = np.argmax(conv_signal[r_min_index:r_max_index])+r_min_index-offset

            # add what we found for that layer
            window_centroids.append((l_center, r_center))


        # Sanity Check 
        # - similar curvature
        curve_faktor = 5
        # - right distance horizontally horizontal toleranz
        hori_tol = window_width/2
        # - roughly parallel  

        # calcualate the radius 
        yvals = range(0,warped.shape[0])
        res_yvals = np.arange(warped.shape[0]-(window_height/2),0,-window_height)
        res_marginyvals = np.arange(warped.shape[0]-(window_height/2),0,-window_height)
        curve_fit_cr_L = np.polyfit(np.array(res_yvals,np.float32)*self.ym_per_pix, np.array(window_centroids,np.float32)[:,0]*self.xm_per_pix ,2)
        curve_fit_cr_R = np.polyfit(np.array(res_yvals,np.float32)*self.ym_per_pix, np.array(window_centroids,np.float32)[:,1]*self.xm_per_pix ,2)
        curve_rad_L = ((1+(2*curve_fit_cr_L[0]*yvals[-1]*self.ym_per_pix + curve_fit_cr_L[1])**2)**1.5) / np.absolute(2*curve_fit_cr_L[0])
        curve_rad_R = ((1+(2*curve_fit_cr_R[0]*yvals[-1]*self.ym_per_pix + curve_fit_cr_R[1])**2)**1.5) / np.absolute(2*curve_fit_cr_R[0])

        # if both lanes has differnt curveradius => do something
        if  (curve_faktor <  curve_rad_R/curve_rad_L or curve_rad_R/curve_rad_L < 1/curve_faktor) and len(self.recent_centers)>0:    
            # average recent_centers to calulaten the middle of the road!  
            smooth_recent_centers = np.average(self.recent_centers[-self.smooth_factor:], axis =0)
            ave_mid_center =   np.array(smooth_recent_centers[:,0]+smooth_recent_centers[:,1])/2
            # calcualte the wide road from all recent data and all levels  
            half_road_wide = np.average(np.array(smooth_recent_centers[:,1]-np.array(smooth_recent_centers)[:,0])/2) 
            for level in range(0,len(ave_mid_center)):
                # if left center out of toleranz hori_tol compare to the old lane than correct it 
                road_width_L = ave_mid_center[level]-window_centroids[level][0]
                if  (road_width_L - hori_tol) > half_road_wide:
                    window_centroids[level] = (window_centroids[level][1]-2*half_road_wide+window_width ,window_centroids[level][1])
                road_width_R = window_centroids[level][0]-ave_mid_center[level]
                # if right center out of toleranz hori_tol compare to the old lane than correct it 
                if  (road_width_R - hori_tol) > half_road_wide :
                    window_centroids[level] = (window_centroids[level][0],window_centroids[level][0]+2*half_road_wide-window_width )
        self.recent_centers.append(window_centroids)
        # return averaged value of the line centers                       and  curve_rad =                  and index 
        return np.average(self.recent_centers[-self.smooth_factor:], axis =0), (curve_rad_L+curve_rad_R)/2 ,len(self.recent_centers)


    def get_tracker_results(self, warped,window_centroids):
        window_width  = self.window_width
        window_height = self.window_height 

        # Points used to draw all the left and right windows
        l_points = np.zeros_like(warped)
        r_points = np.zeros_like(warped)

        # go thought each level and draw the windows
        for level in range(0,len(window_centroids)):

            l_center = window_centroids[level][0]
            l_mask = np.zeros_like(warped)
            l_mask[int(warped.shape[0]-(level+1)*window_height):int(warped.shape[0]-level*window_height),max(0,int(l_center-window_width)):min(int(l_center+window_width),warped.shape[1])] = 1

            r_center = window_centroids[level][1]
            r_mask = np.zeros_like(warped)
            r_mask[int(warped.shape[0]-(level+1)*window_height):int(warped.shape[0]-level*window_height),max(0,int(r_center-window_width)):min(int(r_center+window_width),warped.shape[1])] = 1

            # Add graphic points from window mask here to total pixels found
            l_points[(l_points == 255) | ((l_mask == 1))] = 255
            r_points[(r_points == 255) | ((r_mask == 1))] = 255
   
        # Draw the results
        template = np.array(r_points+l_points,np.uint8)# add both left and right window pixels togehter
        zero_channel = np.zeros_like(template) # create a zero color channle
        template =  np.array(cv2.merge((zero_channel,template,template)),np.uint8) # make window pixel green
        debug = np.array(cv2.merge((warped,warped,warped)),np.uint8) # making the orginal road pixels 3 color channles
        debug = cv2.addWeighted(debug,1,template,.5,0.) # overlay the orinal road image with window results

        return debug


