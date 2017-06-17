# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
from __future__ import division
import cv2
from matplotlib import pyplot as plt
import numpy as np
from math import cos, sin

green = (0,255,0)

def show(image):
    
    # Figure size in inches
    plt.figure(figsize=(10,10))
    
    # Show image, with nearest neighbour interpolation
    plt.imshow(image, interpolation='nearest')
    

def circle_contour(image, contour):
    # Bounding ellipse
    image_with_ellipse = image.copy()
    #easy function
    ellipse = cv2.fitEllipse(contour)
    #add it
    cv2.ellipse(image_with_ellipse, ellipse, green, 2)
    return image_with_ellipse

def overlay_mask(mask, image):
    
    rgb_mask=cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB)
    #calculates the weightes sum of two arrays. in our case image arrays
    #input, how much to weight each. 
    #optional depth value set to 0 no need
    
    img = cv2.addWeighted(rgb_mask, 0.5, image, 0.5, 0)
    
    return img


def find_biggest_contour(image):
    
    image=image.copy()
    #input, gives all the contours, contour approximation compresses horizontal, 
    #vertical, and diagonal segments and leaves only their end points. For example, 
    #an up-right rectangular contour is encoded with 4 points.
    #Optional output vector, containing information about the image topology. 
    #It has as many elements as the number of contours.
    #we dont need it
    image,contours, hierarchy = cv2.findContours(image, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    
    # Isolate largest contour
    contour_sizes = [(cv2.contourArea(contour), contour) for contour in contours]
    biggest_contour = max(contour_sizes, key=lambda x: x[0])[1]

    mask = np.zeros(image.shape, np.uint8)
    cv2.drawContours(mask, [biggest_contour], -1, 255, -1)
    return biggest_contour, mask



def find_strawberry(image):
    
    #step1 correct order
    image=cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    #step2 correct size
    max_dimension= max(image.shape)
    scale = 700/max_dimension
    image=cv2.resize(image,None,fx=scale,fy=scale)
    
    #step3 clean your image
    image_blur=cv2.GaussianBlur(image, (7,7),0)
    image_blur_hsv=cv2.cvtColor(image_blur,cv2.COLOR_RGB2HSV)
    
    #step4 define filters
    
    min_red=np.array([0,180,100])
    max_red=np.array([10,256,256])
    
    mask1 = cv2.inRange(image_blur_hsv,min_red,max_red)
       
    min_red2=np.array([170,100,80])
    max_red2=np.array([180,256,256])
    
    mask2=cv2.inRange(image_blur_hsv,min_red2,max_red2)
    
    #step5 combine mask
    
    mask=mask1+mask2
    
    #step6 segmentation
    
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15,15))
    mask_closed = cv2.morphologyEx(mask,cv2.MORPH_CLOSE,kernel)
    mask_clean = cv2.morphologyEx(mask_closed,cv2.MORPH_OPEN,kernel)
    
    #find_bid strawberry
    
    big_strawberry_contour, mask_strawberries = find_biggest_contour(mask_clean)
    
    # Overlay cleaned mask on image
    overlay = overlay_mask(mask_clean,image)
    
     # Circle biggest strawberry
    circled = circle_contour(overlay,big_strawberry_contour)
     
    show(circled)
     
     #we're done, convert back to original color scheme
    bgr =cv2.cvtColor(circled,cv2.COLOR_RGB2BGR)
     
    return bgr
 
     
 #read the image
image = cv2.imread('yo.jpg')
#detect it
result = find_strawberry(image)
#write the new image
cv2.imwrite('yo2.jpg', result)