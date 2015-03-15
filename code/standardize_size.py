from skimage import io
import pandas as pd
import numpy as np
import os
from skimage import io
from skimage.transform import resize, downscale_local_mean

class StandardizeSize(object):

    def __init__(self, path='/Users/kuoyen/Documents/myweddingstyle/images', raw_dir='raw', uniform_dir='uniform_100_v2'):
        self.raw_dir_path = os.path.join(path, raw_dir)
        self.uniform_dir_path = os.path.join(path, uniform_dir)
        self.raw_subdirs = os.listdir(self.raw_dir_path)
        if not os.path.isdir(self.uniform_dir_path):
            os.mkdir(self.uniform_dir_path) 

    def copy_directory(self):
        # '''Objective: Make a copy of the raw image directory. check to see if it already \
        #     exists. Same naming convention except with "uniform" appended to dir name. '''
        for raw_subdir in self.raw_subdirs:
            uniform_subdir = os.path.join(self.uniform_dir_path, raw_subdir)
            if not os.path.isdir(uniform_subdir):
                os.mkdir(uniform_subdir)

    def standardize(self, n=10, size=(100, 100)):
        print "standardizing to:", size 
        raw_img_subdir = []
        for raw_subdir in self.raw_subdirs: 
            if not raw_subdir.startswith('.'):
                raw_img_subdir.append(raw_subdir)
        for i in raw_img_subdir: 
            uniform_subdir = os.path.join(self.uniform_dir_path, i)
            self.raw_subdir_path = os.path.join(self.raw_dir_path, i)
            self.raw_sub = os.listdir(self.raw_subdir_path)
            self.raw_sub = self.raw_sub[:1050] ###################################hard coded 
            # print len(self.raw_sub) ####
            for ith_image, file_img in enumerate(self.raw_sub): 
                if 'jpg' in file_img:   
                    std_img = self.run_standardize_img(file_img, size)
                    if std_img is not None:
                        uniform_img_path = os.path.join(uniform_subdir, file_img)
                        io.imsave(uniform_img_path, std_img)

    def run_standardize_img(self, file_img, size):
        img = io.imread(self.raw_subdir_path+ '/' + file_img)
        if len(img.shape) == 3:
            std_img = resize(img, size)
            return std_img
        else:
            print 'Image is not colored: ', file_img
        
if __name__ == '__main__':
    ss = StandardizeSize()
    ss.copy_directory()
    print ss.standardize(n=1000)
    # print io.imread('/Users/kuoyen/Documents/myweddingstyle/images/uniform/beach/beach100.jpg')
