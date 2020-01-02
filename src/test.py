import os
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import cv2


img = os.path.abspath('../datasets/fingers/test/0a8c5a58-d75f-4e48-9f06-ac8f8f722ae6_2R.png')
np_img = np.array(Image.open(img), dtype='uint8')

# gray_roi = cv2.cvtColor(np_img, cv2.COLOR_BGR2GRAY)

blurred_roi = cv2.GaussianBlur(np_img, (5, 5), 0)
thresholded_roi = cv2.threshold(blurred_roi, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)[1]
result_roi = cv2.morphologyEx(thresholded_roi, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8), iterations=1)

cv2.imwrite("ROI.jpg", result_roi)