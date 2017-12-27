import cv2

from src import utils

img = cv2.imread('example.jpg')
img = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)

img = utils.augment(img)

img = cv2.cvtColor(img, cv2.COLOR_YUV2BGR)
cv2.imshow('IMAGE', img)
cv2.waitKey(0)

# TODO: Add a file to see examples with my grid!
# TODO: Finish this and plot examples! Get grid helper function from my other repos!