import cv2

import utils

img = cv2.imread('../output/images/003 - Example Image Left.jpg')
img = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)

cv2.imwrite('../output/images/031 - Augmented Example All Contrast.jpg',
    cv2.cvtColor(utils.augment(img, 0, 2), cv2.COLOR_YUV2BGR))

cv2.imwrite('../output/images/032 - Augmented Example All Brightness.jpg',
    cv2.cvtColor(utils.augment(img, 1, 2), cv2.COLOR_YUV2BGR))

cv2.imwrite('../output/images/033 - Augmented Example Left Contrast.jpg',
    cv2.cvtColor(utils.augment(img, 2, 2), cv2.COLOR_YUV2BGR))

cv2.imwrite('../output/images/034 - Augmented Example Right Contrast.jpg',
    cv2.cvtColor(utils.augment(img, 4, 2), cv2.COLOR_YUV2BGR))

cv2.imwrite('../output/images/035 - Augmented Example Left Brightness.jpg',
    cv2.cvtColor(utils.augment(img, 3, 2), cv2.COLOR_YUV2BGR))

cv2.imwrite('../output/images/036 - Augmented Example Right Brightness.jpg',
    cv2.cvtColor(utils.augment(img, 5, 2), cv2.COLOR_YUV2BGR))

cv2.imwrite('../output/images/037 - Augmented Example Contrast & Brightness.jpg',
    cv2.cvtColor(utils.augment(img, 6, 2), cv2.COLOR_YUV2BGR))

cv2.imwrite('../output/images/038 - Augmented Example Brightness & Contrast.jpg',
    cv2.cvtColor(utils.augment(img, 7, 2), cv2.COLOR_YUV2BGR))

cv2.imwrite('../output/images/039 - Augmented Example Blur.jpg',
    cv2.cvtColor(utils.augment(img, 8, 1), cv2.COLOR_YUV2BGR))

cv2.imwrite('../output/images/040 - Augmented Example Sharp.jpg',
    cv2.cvtColor(utils.augment(img, 8, 0), cv2.COLOR_YUV2BGR))

cv2.imwrite('../output/images/041 - Augmented Example Contrast & Brightness & Blur.jpg',
    cv2.cvtColor(utils.augment(img, 6, 1), cv2.COLOR_YUV2BGR))

cv2.imwrite('../output/images/042 - Augmented Example Brightness & Contrast & Sharp.jpg',
    cv2.cvtColor(utils.augment(img, 7, 0), cv2.COLOR_YUV2BGR))