import numpy as np
from scipy import ndimage, misc
from skimage.color import rgb2gray
from skimage.exposure import equalize_hist
from skimage.filters import roberts, sobel, canny
from scipy import ndimage
from skimage.segmentation import slic
from helper import pad_list
import cv2

'''
DOC:
This file has all the image filters and transformation
functions that were used during the exploratory phase.
'''


def filter_function(img, filter=''):
    if filter == 'hist':
        filtered_image = equalize_hist(img)
    elif filter == 'background':
        filtered_image = background(img)
    elif filter == 'segment':
        filtered_image = segments_slic = slic(img,
                        n_segments=3, compactness=10, sigma=1)
    return filtered_image


def sift(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    sift = cv2.SIFT()
    kp, des = sift.detectAndCompute(gray, None)
    img = cv2.drawKeypoints(gray, kp)
    if des is None:
        des = [0]
        sift_all.extend([0])
        print "des empty for", i
    else:
        for arr in des:
            sift_arr.extend(arr)
        sift_all.append(sift_arr)
    sift_padded = pad_list(sift_all)
    return sift_padded


def edge_detection(img, method=''):
    img = rgb2gray(img)
    if method == "roberts":
        edge_detector = roberts(img)
    elif method == 'sobel':
        edge_detector = sobel(img)
    elif method == 'canny':
        edge_detector = canny(img)
    return edge_detector


def background(img):
    mask = np.zeros(img.shape[:2], np.uint8)
    bgdModel = np.zeros((1, 65), np.float64)
    fgdModel = np.zeros((1, 65), np.float64)
    rect = (50, 50, 450, 290)
    cv2.grabCut(img, mask, rect, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_RECT)
    mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
    new_img = img * mask2[:, :, np.newaxis]
    new_img_1d = new_img.ravel()
    img_1d = img.ravel()
    blacking_out_ind = np.where(new_img_1d != 0)
    img_1d[blacking_out_ind] = 0
    background_extracted_img = img_1d.reshape((342, 548, 3))
    return background_extracted_img


def whiten(img):
    mu = np.mean(img)
    std = np.std(img)
    img = (img - mu) / std
    return img


def dropout(feature_matrix, p, output):
    vlength = len(post_feat_mat)
    d = np.random.binomial(1, n, vlength)
    dropout_arr = post_feat_mat * d
    return dropout_arr
