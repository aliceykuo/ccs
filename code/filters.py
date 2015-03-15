import sys
import os
import numpy as np
# import itertools
from sklearn.preprocessing import StandardScaler
from skimage import io
import scipy
from scipy import ndimage, misc
from skimage.color import rgb2gray
from skimage.exposure import equalize_hist
from skimage.segmentation import felzenszwalb, slic, quickshift
from PIL import Image
from scipy.cluster.vq import vq
from sklearn.cross_validation import train_test_split
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score
from sklearn.ensemble import RandomForestClassifier
from sklearn import preprocessing
import pickle as pkl
from dom_color import DominantColor
import cv2
import time
import datetime
import os, errno

from sklearn.feature_extraction import image

def run_extract_patch():


def extract_patch(img, patch_size = (10,10), max_patches = 100,dominant_color_vector=None):
    pass 
    if dominant_color_vector is None:
        raise 'No dominant colors found!'
    # won't need to read in image 
    # img = io.imread('/Users/kuoyen/Documents/myweddingstyle/images/uniform_100/ballroom/ballroom0.jpg')
    '''
    pre trans matrix has dominant colors of (SAMPLES X 15)
    have original img matrix that will be divided into "n patches" 
    blow up feature matrix space  ----> becomes (SAMPLES x n_patches ) (i.g. 1000 x 5 = 500 ---> 5000, 15 + patch_img_mat raveled)

    '''
    patches = image.extract_patches_2d(img, patch_size, max_patches)
    slices = np.shape(patches)[0]
    arr = []
    for i in xrange(slices):
        arr.append(np.concatenate(slices[i].ravel(),axis=1))
    arr = np.array(arr)
    print img.shape
    print "patches of 10", patches.shape
    print patches_nomax.shape
    return arr



def filter_function(img, filter = ''):
    if filter == 'hist':
        filtered_image = equalize_hist(img)
    elif filter == 'background':
        filtered_image = bkg()
    elif filter == 'segment':
        filtered_image = segments_slic = slic(img, n_segments=3, compactness=10, sigma=1)
    return filtered_image


def whiten(img):
    mu = np.mean(img)
    std = np.std(img)
    img = (img - mu) / std
    return img
