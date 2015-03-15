from skimage import io
import pandas as pd
import numpy as np
from skimage.transform import resize, downscale_local_mean
from sklearn.preprocessing import StandardScaler
# from PIL import Image
import cv2
import os, sys
import pickle as pkl

class PreprocessImg(object):

    def __init__(self, path='/Users/kuoyen/Documents/myweddingstyle/images/uniform'):
        self.path = path
   
# setup a standard image size - downsize and get all images to the same dimensions 
# converts into a numpy array of RGB pixels
    def img_to_matrix(self, n=100, directory = ''):
        """doc string"""
        img_dir_path = os.path.join(self.path, directory)
        dirs = os.listdir(img_dir_path)
        label_img_arr = None
        for ith_image, file_img in enumerate(dirs):
            # if not raw_subdir.startswith('.'):
            #     raw_img_subdir.append(raw_subdir)
            if ith_image <= 10:
                if ith_image == 0 or ith_image % 100 == 0:
                    print 'Vectorized', ith_image
            if 'jpg' in file_img:
                img_path = os.path.join(img_dir_path, file_img)
                img = io.imread(img_path)
                img = img.ravel()
                label_img = np.append(directory, img)[np.newaxis, :]
                if label_img_arr is None:
                    label_img_arr = label_img
                else:
                    label_img_arr = np.r_[label_img_arr, label_img]
                    # print 'Something wrong with image', file_img
        return label_img_arr

    def run_img_to_matrix(self, dir_list=[], n=10):
        label_img_arr_lst = []
        for directory in dir_list:
            print directory
            label_img_arr_lst.append(self.img_to_matrix(directory, n))
        # return label_img_arr_list.shape
        return np.concatenate(label_img_arr_lst, axis=0)
            # '%s _arr' % (directory) = img_to_matrix(directory)
            # print '%s _arr' % (directory)

    #dropout: binomial sampling of a vector equal to the lengh of my feature matrix


    def dropout(flatten_img, p, output):
        vlength = len(img_mat)
        d = np.random.binomial(1, n, vlength)
        dropout_arr = flatten_img * d
        return dropout_arr

if __name__=='__main__':
    pp = PreprocessImg()
    mat = pp.img_to_matrix(n=5, directory='beach')
    print mat
    # pp.run_img_to_matrix(dir_list = ['beach', 'rustic', 'ballroom'])


    # ballroom_arr = img_to_matrix(directory='ballroom')
    # vintage_arr = img_to_matrix(directory='vintage')
    # beach_arr = img_to_matrix(directory='beach')
    # merged = np.concatenate((beach_arr, rustic_arr, ballroom_arr, vintage_arr), axis =0)
    # pkl.dump(rustic_arr, open('..//rustic_wedding.pkl', 'wb'))

    # merged_100 = np.concatenate((beach_arr, rustic_arr, ballroom_arr, vintage_arr), axis =0)
    # pkl.dump(merged_100, open('../image_dir/merged_100.pkl', 'wb'))