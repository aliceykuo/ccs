import numpy as np
import cv2
from matplotlib import pyplot as plt
from skimage import io
%matplotlib inline


# img = cv2.imread('/Users/kuoyen/Desktop/beachsamp1.jpg')
img = io.imread('/Users/kuoyen/Desktop/messi5.jpg')
img.shape

mask = np.zeros(img.shape[:2], np.uint8)

bgdModel = np.zeros((1,65),np.float64)
fgdModel = np.zeros((1,65),np.float64)

rect = (50, 50, 450, 290)
cv2.grabCut(img, mask, rect, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_RECT)

mask2 = np.where((mask==2) | (mask==0),0,1).astype('uint8')
new_img = img * mask2[:, :, np.newaxis]

new_img_1d = new_img.ravel()
img_1d = img.ravel()

blacking_out_ind = np.where(new_img_1d != 0)
img_1d[blacking_out_ind] = 0
background_extracted_img = img_1d.reshape((342, 548, 3))
plt.imshow(background_extracted_img)