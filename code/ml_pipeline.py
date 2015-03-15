from skimage import io
import pandas as pd
import numpy as np
from skimage.transform import resize, downscale_local_mean
from sklearn.preprocessing import StandardScaler
import cv2
import os, sys
import pickle as pkl
from PIL import Image
from scipy.cluster.vq import vq
import struct
import Image
import scipy
import scipy.misc
import scipy.cluster

class PreprocessImg(object):

    def __init__(self, path='/Users/kuoyen/Documents/myweddingstyle/images/uniform', size=(28,28)):
        self.path = path
        self.size = size
        self.std_size = size[0] *size[1] *3
        # print self.std_size
        self.img_dir_path = None
        self.raw_mat = None

    def downsize_raw(self, ):
        # s_size = (100, 75)
        pass 

    def create_feat_mat(self):
        pass

    # def raw_img_mat(self, ):
    #   pass

    # def filtered_img():
 #      pass
   
# setup a standard image size - downsize and get all images to the same dimensions 
# converts into a numpy array of RGB pixels
    def img_to_matrix(self, n=5, directory = 'beach'):
        """reads in img from uniform path and converts to np array"""
        raw_img_mat = None
        img_dir_path = os.path.join(self.path, directory)
        img_subdir = self.check_filetype(img_dir_path)
        for ith_image, file_img in enumerate(img_subdir):
            if ith_image <= n:
                file_img = self.check_jpg(file_img)
                img_path = os.path.join(img_dir_path, file_img)
                img = io.imread(img_path)
                #check size here
                label_img = np.append(directory, img)[np.newaxis, :]
                # if len(label_img[0]) != self.std_size:
                #     print len(label_img[0])
                if raw_img_mat is None:
                    raw_img_mat = label_img
                else:
                    raw_img_mat = np.r_[raw_img_mat, label_img]
        return raw_img_mat

    def run_img_to_matrix(self, dir_list=[], n=2):
        raw_img_mat_lst = []
        for directory in dir_list:
            raw_img_mat = self.img_to_matrix(directory = directory, n=2)
            raw_img_mat_lst.append(raw_img_mat)
        feat_matrix = np.concatenate(raw_img_mat_lst, axis=0)
        return feat_matrix[:,1:]

#EXTRACT ADDITIONAL FEATURES 
    def select_filters(self):
        pass

    def dominant_color(self, directory='', n=3, k=10): 
        raw_img_path = ('/Users/kuoyen/Documents/myweddingstyle/images/raw')
        # TODO: remove pathing to get image 
        img_dir_path = os.path.join(raw_img_path, directory)
        print img_dir_path
        img_subdir = self.check_filetype(img_dir_path)
        for ith_image, file_img in enumerate(img_subdir):
            if ith_image < n:
                file_img = self.check_jpg(file_img)
                print file_img
                filename = os.path.join(img_dir_path, file_img)
                img = Image.open(filename).resize( (60,60) )
                # img = img.resize( (28,28), Image.ANTIALIAS)     # optional, to reduce time
                ar = scipy.misc.fromimage(img)
                shape = ar.shape
                ar = ar.reshape((scipy.product(shape[:2]), shape[2]))
                # print ar.shape
                # print 'finding clusters'
                codes, dist = scipy.cluster.vq.kmeans(ar, k)
                # print 'cluster centres:\n', codes
                vecs, dist = scipy.cluster.vq.vq(ar, codes)         # assign codes
                counts, bins = scipy.histogram(vecs, len(codes))    # count occurrences
                index_top = np.argsort(counts)#[0:-4:-1]                   # find most frequent
                # print len(index_top)
                for i in codes[index_top]:
                    colour = ''.join(chr(c) for c in i).encode('hex')
                    print 'top color is %s (#%s)' % (i, colour)         
                top_colors = [] 
                for i in range(len(index_top)):
                    top_colors.append(codes[i])
                print top_colors
                # dom_colors.append(top_colors)
                # return dom_colors
                            # dom_color = np.concatenate(top_colors, axis=0)
                            # print dom_color
                            # peak = codes[index_max]
                            # colour = ''.join(chr(c) for c in peak).encode('hex')
                            # print 'most frequent is %s (#%s)' % (peak, colour)

    def run_dominant_colors(self, dir_list=[], n=10):
        for directory in dir_list:
            dom_colors = self.dominant_color(directory=directory, n=5, k=3)
            return dom_colors

#FILTERS
    #dropout: binomial sampling of a vector equal to the lengh of my feature matrix
    def dropout(flatten_img, p, output):
        vlength = len(img_mat)
        d = np.random.binomial(1, n, vlength)
        dropout_arr = flatten_img * d
        return dropout_arr

    def flatten_mat(self, mat):
        for i in mat:
            img_vec = i.ravel()
            label_img = np.append(directory, img)[np.newaxis, :]
        return img

    def standard_scaler(self, X): 
        # X = StandardScaler
        pass

    def check_filetype(self, img_dir_path):
        img_subdir = [i for i in os.listdir(img_dir_path) if not i.startswith('.')]
        return img_subdir

    def check_jpg(self, file_img):
        if 'jpg' in file_img:
            return file_img
    # def main():
    #     ml = PreprocessingImg()
    #     self.raw_mat = ml.run_img_to_matrix(dir_list= ['beach', 'rustic', 'ballroom'])

if __name__=='__main__':
    pp = PreprocessImg()
    pp.run_img_to_matrix(dir_list= ['beach', 'rustic', 'ballroom'])
    pp.run_domminant_colors(dir_list= ['beach', 'rustic', 'ballroom'])
    # merged_100 = np.concatenate((beach_arr, rustic_arr, ballroom_arr, vintage_arr), axis =0)
    # pkl.dump(merged_100, open('../image_dir/merged_100.pkl', 'wb'))