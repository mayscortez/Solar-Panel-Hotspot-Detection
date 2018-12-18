#Script to detect the hotspots in solar panel image

#import appropriate libraries
import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt

#image path (should be greyscale image)
path = '/home/user/sandbox - Cortez/Pictures/frame52.jpg'

#read image from Pictures file as greyscale image
im = cv.imread(path,0)

#for testing
#plt.imshow(im, 'gray')

############################################################################
#                              CHECKPOINT 1
############################################################################

#For this picture, 80 doesn't work as threshold
#thresh = 80
thresh = 30
#runs thresholding algorithm on im
#THRESH_BINARY: IF im(x,y) > thresh --> dst(x,y) = maxVal
#               ELSE --> dst(x,y) = 0
ret,dst = cv.threshold(im,thresh,255,cv.THRESH_BINARY)

#looks for the contours in dst
#second argument is contour retrieval mode, third argument is contour approximation method
#im2 is image with found contours
#contours is list of locations of found contours
#hierarchy is ...
im2, contours, hierarchy = cv.findContours(dst,cv.RETR_TREE,cv.CHAIN_APPROX_SIMPLE)

#for testing
#plt.imshow(im2, 'gray')

############################################################################
#                              CHECKPOINT 2
############################################################################

#draws contours in gray on original image using contour locations from thresholded image
im3 = cv.drawContours(im, contours, -1, (128,255,0), 1)
#for testing
plt.imshow(im3,'gray')

############################################################################
#                              CHECKPOINT 3
############################################################################

#preMoment is a list that will carry all the information that moments returns for each object in contours
preMoment = []

#adding the different moment information into preMoment
for i in range(0,len(contours)):
   preMoment.append(cv.moments(contours[i],True))

#print('For testing, preMoment[] length is: ', len(preMoment))
#print('For testing, preMoment[3]: ', preMoment[3])
#print('For testing, preMoment[3]["m00"]: ', preMoment[3]["m00"])

#moment will hold the coordinates of the moments
moment = []

#calculate the moments of each contour
for i in range(0,len(preMoment)):
    #print('{0:d} \n'.format(i))
    #print('{0:f} \n'.format(preMoment[i]["m00"]))

    # if denominator is not zero, calculate moment and store as list of coordinates in moment
    if (preMoment[i]["m00"] != 0):
        xcoord = int(preMoment[i]["m10"] / preMoment[i]["m00"])
        ycoord = int(preMoment[i]["m01"] / preMoment[i]["m00"])
        moment.append([xcoord,ycoord])
    #otherwise too small for us to care about
    else:
        #print('false\n')
        moment.append([0,0])

#print('For testing, moment[] length is ', len(moment))
#print('\n', moment)

#area is a list containing the areas of each contour
area = []

#calculate the areas of each contour and append to area list
for i in range(0,len(contours)):
    #print('{0:d} \n'.format(i))
    area.append(cv.contourArea(contours[i]))

#print('\nFor testing, area[] length is ', len(area))
#print('\n', area)

#perimeter is a list containing the contour perimeters
perimeter = []

#calculate the perimeters of each contour and append to perimeter list
for i in range(0,len(contours)):
    #print('{0:d} \n'.format(i))
    perimeter.append(cv.arcLength(contours[i],True))

#print('\nFor testing, perimeter[] length is ', len(perimeter))
#print('\n', perimeter)

#cv.imwrite('/home/user/sandbox - Cortez/Pictures/contour_threshold.png', im3)

############################################################################
#                              CHECKPOINT 4
############################################################################

img = cv.imread(path,1)
#draws contours in gray on original image using contour locations from thresholded image
img2 = cv.drawContours(img, contours, -1, (128,255,0), 1)

# Create a mask of contours and create its inverse mask also
img2gray = cv.cvtColor(img2, cv.COLOR_BGR2GRAY)
ret, mask = cv.threshold(img2gray, 10, 255, cv.THRESH_BINARY)
mask_inv = cv.bitwise_not(mask)

# Now black-out the area of contours in original image
img1_bg = cv.bitwise_and(img, img, mask = mask_inv)

# Take only region of contours from contours image.
img2_fg = cv.bitwise_and(img2, img2, mask = mask)

# Put contours in image and modify the main image
dst = cv.add(img1_bg,img2_fg)
plt.imshow(dst)

############################################################################
#                              CHECKPOINT 5
############################################################################
rect = []
for i in range(0,len(contours)):
    #print('{0:d} \n'.format(i))
    rect.append(cv.minAreaRect(contours[i]))

#print('\nFor testing, rect[] length is ', len(rect))
#print('\n', rect)

#Printing bounding box example
box = cv.boxPoints(rect[100])
box = np.int0(box)
image = cv.drawContours(dst,[box],0,(255,0,0),3)
plt.imshow(image)

#cv.imwrite('/home/user/sandbox - Cortez/Pictures/contour_52_green01.png', image)