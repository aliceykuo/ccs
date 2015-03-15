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



def main():
    pass 
    ''' Standardize size of image ''' 

    ''' Run load_files '''

    ''' Extract features from raw image '''

    ''' Transform img - his, segment, sift, extract_patches  '''

    ''' extract information from "transform step"  '''

    ''' build feature matrix (pre matrix, post matrix "