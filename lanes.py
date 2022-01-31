import cv2
import numpy as np
import matplotlib.pyplot as plt

def canny(image):
    # Change the color image (3 channel) to gray scale (1 channel)
    gray = cv2.cvtColor(image,cv2.COLOR_RGB2GRAY)
    # Smoothening grayscale image using Gaussian kernel to reduce noise
    blur = cv2.GaussianBlur(gray,(5,5),0)
    # To outine the strongest gradient in the image
    canny = cv2.Canny(blur,50,150)
    # return the canny image
    return canny

def region_of_interest(image):
    # Identify the height of the image
    height = image.shape[0]
    # Define a polygon with ROI vertices which you like to extract
    # Hint: Use matplotlib.pyplot to observe the canny image with X and Y axis. 
    # This can help to identify the ROI
    polygons=np.array([
        [(200,height),(1100,height),(550,250)]
        ])
    # Creating the black mask image with same size as the image
    mask = np.zeros_like(image)
    # Fill the polygonal contour with white
    cv2.fillPoly(mask,polygons,255)
    # Masking the canny image to shoe the ROI traced by polygonal contour of mask
    masked_image=cv2.bitwise_and(image,mask)
    return masked_image

def display_lines(image, lines):
    line_image=np.zeros_like(image)
    if lines is not None:
        for line in lines:
            x1,y1,x2,y2=line.reshape(4)
            cv2.line(line_image,(x1,y1),(x2,y2),(255,0,0),10)
    return line_image

# Read the test image
image = cv2.imread('test_image.jpg')
# Copy the test image
lane_image=np.copy(image)
# Call canny function for Grayscale conversion,Image smoothening & simple edge detection
canny_image = canny(lane_image)
# Call region_of_interest function to crop the ROI from canny image
cropped_image = region_of_interest(canny_image)
# Define the hough space in the cropped image
lines=cv2.HoughLinesP(cropped_image, 2, np.pi/180, 100, np.array([]), minLineLength=40, maxLineGap=5)
# line image
line_image=display_lines(lane_image,lines)
# Blend the original road image with detected line image
combo_image=cv2.addWeighted(lane_image, 0.8, line_image, 1, 1 )
# Display the ROI
cv2.imshow("result",combo_image)
cv2.waitKey(0)
