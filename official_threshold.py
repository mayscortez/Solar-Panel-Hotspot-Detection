# Mayleen Cortez, UNIV 498 Research with Dr. Miller
# Modified from https://docs.opencv.org/3.4/d7/d4d/tutorial_py_thresholding.html
#
# Algorithm for detecting if there are hotspots in a given image; does not yet account for finding the solar panel
# Last modified: 10/10/18

import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

#Boolean- tells us if there are hotspots, i.e. False means no hotspots and True means there is at least one hotspot
hotFlag = False

#set thresholding value
thresh = 127

#read the solar array image
img = cv.imread('bwgradient.jpg', 0)

# plt.imshow('Image',img)
# plt.waitKey(0)
# plt.destroyAllWindows


#checks each pixel of img, creates new image newImg where
#if the pixel in img is greater than threshold value, corresponding pixel in new image will be 0
#otherwise, corresponding pixel in new image will keep that pixel value
ret, newImg = cv.threshold(img, thresh, 255, cv.THRESH_TOZERO_INV)

# plt.imshow('New Image',newImg)
# plt.waitKey(0)
# plt.destroyAllWindows

#if all values in newImg are zero, then there are no hotspots so hotFlag remains false
#otherwise, there are (potential) hotspots so get locations and make hotFlag true
if np.count_nonzero(img) != 0:
    #returns locations as a tuple of arrays with indicies of non-zero elements
    locations = np.nonzero(newImg)
    hotFlag = True

print('Length of locations: ', len(locations))
print('\n')

#Should print out each saved coordinate in locations on new line
for i in range(0,len(locations)):
    print('At location {0:d} is: '.format(i), locations[i])
    print('\n')
