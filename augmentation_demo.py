import cv2

import utils

img = cv2.imread('example.jpg')
img = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)

img = utils.augment(img)

img = cv2.cvtColor(img, cv2.COLOR_YUV2BGR)
cv2.imshow('IMAGE', img)
cv2.waitKey(0)