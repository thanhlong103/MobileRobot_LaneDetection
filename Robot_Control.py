import numpy as np
import cv2
from moviepy.editor import *
import serial

## Open communication between Raspberry Pi and Arduino ###
# port = '/dev/ttyAMA0'
# port = '/dev/ttyUSB0'
# arduino = serial.Serial(port, 9600, timeout=1)

### Define ranges to control robot's moving direction ###
midpoint = 320
std = 40
range1 = [midpoint-std, midpoint+std]
range2 = [range1[0]-std, range1[1]+std]
range3 = [range2[0]-std, range2[1]+std]

def region_of_interest(img, vertices):
    """Select the region of interest (ROI) from a defined list of vertices."""
    # Defines a blank mask.
    mask = np.zeros_like(img)   
    
    # Defining a 3 channel or 1 channel color to fill the mask.
    if len(img.shape) > 2:
        channel_count = img.shape[2]  # 3 or 4 depending on your image.
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255
        
    # Filling pixels inside the polygon.
    cv2.fillPoly(mask, vertices, ignore_mask_color)
    # cv2.fillPoly(mask, roi_middle_vertices, 0) #make polygon inside lane turns black
    
    # Returning the image only where mask pixels are nonzero.
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image

def draw_lines(img, lines, color=[255, 0, 0], thickness = 2):
    """Utility for drawing lines."""
    if lines is not None:
        for line in lines:
            for x1,y1,x2,y2 in line:
                cv2.line(img, (int(x1), int(y1)), (int(x2), int(y2)), color, thickness)

def hough_lines(img, rho, theta, threshold, min_line_len, max_line_gap):
    """Utility for defining Line Segments."""
    lines = cv2.HoughLinesP(
        img, rho, theta, threshold, np.array([]),
        minLineLength = min_line_len, maxLineGap = max_line_gap)
    if lines is not None:
        line_img = np.zeros((img.shape[0], img.shape[1], 3), dtype = np.uint8)
        draw_lines(line_img, lines)
        return line_img, lines
    if lines is None:
        print('NO LINES')

def separate_left_right_lines(lines):
    """Separate left and right lines depending on the slope."""
    left_lines = []
    right_lines = []
    if lines is not None:
        for line in lines:
            for x1, y1, x2, y2 in line:
                if (y1-y2) / (x1-x2) < 0: # Negative slope = left lane.
                    left_lines.append([x1, y1, x2, y2])
                elif (y1-y2) / (x1-x2) > 0: # Positive slope = right lane.
                    right_lines.append([x1, y1, x2, y2])
    return left_lines, right_lines

def cal_avg(values):
    """Calculate average value."""
    if values is None:
        return 0
    elif len(values) > 0:
        n = len(values)
        return sum(values) / n

def extrapolate_lines(lines, upper_border, lower_border):
    """Extrapolate lines keeping in mind the lower and upper border intersections."""
    slopes = []
    consts = []
    a = 0
    if (lines is not None) and (len(lines) != 0):
        for x1, y1, x2, y2 in lines:
            if x1 == x2: 
                x2 = x2 + 1
            slope = (y1-y2) / (x1-x2)
            slopes.append(slope)
            c = y1 - slope * x1
            consts.append(c)
                
        avg_slope = cal_avg(slopes)
        avg_consts = cal_avg(consts)
        if avg_slope == 0: 
            avg_slope == avg_slope + 0.00001
        # Calculate average intersection at lower_border.
        x_lane_lower_point = float((lower_border - avg_consts) / avg_slope)

        # Calculate average intersection at upper_border.
        x_lane_upper_point = float((upper_border - avg_consts) / avg_slope)

        return [x_lane_lower_point, lower_border, x_lane_upper_point, upper_border]
        
def extrapolated_lane_image(img, lines, roi_upper_border, roi_lower_border):
    """Main function called to get the final lane lines."""
    lanes_img = np.zeros((img.shape[0], img.shape[1], 3), dtype = np.uint8)
    # Extract each lane.
    lines_left, lines_right = separate_left_right_lines(lines)
    lane_left = extrapolate_lines(lines_left, roi_upper_border, roi_lower_border)
    lane_right = extrapolate_lines(lines_right, roi_upper_border, roi_lower_border)
    
    # uppoint_left = (int(lane_left[0]), int(lane_left[1]))
    # uppoint_right = (int(lane_right[0]), int(lane_right[1]))
    # downpoint_left = (int(lane_left[2]), int(lane_left[3]))
    # downpoint_right = (int(lane_right[2]), int(lane_right[3]))
    
    # cv2.line(lanes_img, uppoint_left, downpoint_left, [255, 0, 0], 50)
    # cv2.line(lanes_img, uppoint_right, downpoint_right, [255, 0, 0], 50)
    
    if (lane_left is not None) and (lane_right is not None):
        draw_con(img, [[lane_left], [lane_right]])
        # Draw the midline.
        up_x = int((lane_left[2] + lane_right[2]) / 2)
        down_x = int((lane_left[0] + lane_right[0]) / 2)
        mid_up = (up_x, roi_upper_border)
        mid_down = (down_x,  roi_lower_border)
        cv2.line(img, mid_up, mid_down, [0, 0, 255], 3)
    if (lane_left is None) or (lane_right is None):
        mid_up = (640, 361)
        mid_down = (645, 360)
    return lanes_img, mid_up, mid_down

def draw_con(img, lines):
    """Fill in lane area."""
    points = []
    for x1,y1,x2,y2 in lines[0]:
        points.append([x1,y1])
        points.append([x2,y2])
    for x1,y1,x2,y2 in lines[1]:
        points.append([x2,y2])
        points.append([x1,y1])
     
    points = np.array([points], dtype = 'int32')  
        
    cv2.fillPoly(img, points, (0,255,0))

def distance_from_center(mid_up, mid_down):
    if (mid_up is None):
        mid_up = (640, 360)
    
    if (mid_down is None):
        mid_down = (640, 360)
        
    if (mid_up is not None) and (mid_down is not None):
        # Compute the slope
        param = (mid_up[0] - mid_down[0])
        if(param == 0):
            param = param + 1
        else:
            m = (mid_up[1] - mid_down[1]) / param
            
            # Compute the y-intercept
            b = mid_up[1] - m * mid_up[0]

            # Compute x 
            mid_x = (360 - b) / m

            # Calculate the distance between the two points in pixels
            dist_pixels = mid_x - 640

            # Calculate the distance between the two points in millimeters
            dist_mm = round(dist_pixels * 0.04285714 * 10, 3)
            
            if dist_mm is None:
                return None
            
            return dist_mm
        

def process_image(image):
    gray_select = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # h, s, v1 = cv2.split(hsv)
    
    # Remove noise using Gaussian blur.
    kernel_size = 13
    blur = cv2.GaussianBlur(gray_select, (kernel_size, kernel_size), 0)
    
    # Use global threshold based on grayscale intensity.
    threshold = cv2.inRange(blur, 175, 255)

    # Region masking: Select vertices according to the input image.
    image_w = threshold.shape[1]
    image_h = threshold.shape[0]
    roi_vertices = np.array([[[0, image_h],
                                [image_w, image_h],
                                [image_w, image_h/5],
                                [0, image_h/5]]], dtype = np.int32)
    
    # roi_middle_vertices = np.array([[200, 1200],
    #                         [400, 1200],
    #                         [400, 300],
    #                         [200, 300]])

    gray_select_roi = region_of_interest(threshold, roi_vertices)
    
    # Canny Edge Detection.
    low_threshold = 20
    high_threshold = 120
    img_canny = cv2.Canny(gray_select_roi, low_threshold, high_threshold)
    
    dilate = cv2.dilate(img_canny, (5,5) , iterations=5)
    
    # Hough transform parameters set according to the input image.
    rho = 1
    theta = np.pi/180
    threshold = 150
    min_line_len = 100
    max_line_gap = 200
    hough, lines = hough_lines(dilate, rho, theta, threshold, min_line_len, max_line_gap)
    
    # Extrapolate lanes.
    roi_upper_border = 0
    roi_lower_border = 720
    # roi_upper_border = int(image.shape[0] * 3 / 10)
    # roi_lower_border = image.shape[0]
    lane_img, mid_up, mid_down = extrapolated_lane_image(image, lines, roi_upper_border, roi_lower_border)
    
    dist_mid = distance_from_center(mid_up, mid_down)
    center_line = [[[640, 0, 640, 1280]]]
    draw_lines(lane_img, center_line)
    
    # Combined using weighted image.
    image_result = cv2.addWeighted(image, 1, lane_img, 0.4, 0.0)
    
    scale_percent = 30 # percent of original size
    width = int(image_result.shape[1] * scale_percent / 100)
    height = int(image_result.shape[0] * scale_percent / 100)
    dim = (width, height)
  
    # resize image
    resized_2 = cv2.resize(image_result, dim, interpolation = cv2.INTER_AREA)
    
    return resized_2, hough, dist_mid

def get_midpoints(image):
    gray_select = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # h, s, v1 = cv2.split(hsv)
    
    # Remove noise using Gaussian blur.
    kernel_size = 13
    blur = cv2.GaussianBlur(gray_select, (kernel_size, kernel_size), 0)
    
    # Use global threshold based on grayscale intensity.
    threshold = cv2.inRange(blur, 175, 255)

    # Region masking: Select vertices according to the input image.
    image_w = threshold.shape[1]
    image_h = threshold.shape[0]
    roi_vertices = np.array([[[0, image_h],
                                [image_w, image_h],
                                [image_w, image_h/5],
                                [0, image_h/5]]], dtype = np.int32)
    
    # roi_middle_vertices = np.array([[200, 1200],
    #                         [400, 1200],
    #                         [400, 300],
    #                         [200, 300]])

    gray_select_roi = region_of_interest(threshold, roi_vertices)
    
    # Canny Edge Detection.
    low_threshold = 20
    high_threshold = 120
    img_canny = cv2.Canny(gray_select_roi, low_threshold, high_threshold)
    
    dilate = cv2.dilate(img_canny, (5,5) , iterations=5)
    
    # Hough transform parameters set according to the input image.
    rho = 1
    theta = np.pi/180
    threshold = 150
    min_line_len = 100
    max_line_gap = 200
    hough, lines = hough_lines(dilate, rho, theta, threshold, min_line_len, max_line_gap)
    
    # Extrapolate lanes.
    roi_upper_border = 0
    roi_lower_border = 720
    # roi_upper_border = int(image.shape[0] * 3 / 10)
    # roi_lower_border = image.shape[0]
    lane_img, mid_up, mid_down = extrapolated_lane_image(image, lines, roi_upper_border, roi_lower_border)
    return mid_up, mid_down

########## MAIN LOOP ##########
if __name__ == '__main__':
    cap = cv2.VideoCapture("real_lane.mp4")
    # cap = cv.VideoCapture(0)
    
    while True:
        ret, frame = cap.read()
        if frame[0][0][0] == 0: # To avoid black frames of webcam at launch
            continue
        top_val, bottom_val = get_midpoints(frame)
        top = top_val[0]
        print("top", top)
        # Display the resulting frame
        result, hough, dist_mid = process_image(frame)
        if dist_mid < 0 : 
            cv2.putText(result, f"RIGHT = {-dist_mid} mm", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 2)
        if dist_mid > 0 : 
            cv2.putText(result, f"LEFT = {dist_mid} mm", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 2)
        if dist_mid == 0 : 
            cv2.putText(result, f"CENTER = {dist_mid} mm", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 2)  
        cv2.imshow('Frame', result)
        cv2.waitKey(1)

        # CASES for LANE DETECTION
        if range1[0] <= top < range1[1]:
            # arduino.write(b'a')
            print('a')
        elif range2[0] <= top < range1[0]:
            # arduino.write(b'b')
            print('b')
        elif range3[0] < top < range2[0]:
            # arduino.write(b'c')
            print('c')
        elif range1[1] <= top < range2[1]:
            # arduino.write(b'd')
            print('d')
        elif range2[1] <= top < range3[1]:
            # arduino.write(b'e')
            print('e')
        elif top < range3[0]:
            # arduino.write(b'f')
            print('f')
        elif top >= range3[1]:
            # arduino.write(b'g')
            print('g')