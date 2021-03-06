from skimage import io
import pandas as pd
import numpy as np
import os
from skimage.transform import resize, downscale_local_mean


class StandardizeSize(object):
    '''
            DOC: The StandardizeSize class does the following:
            1. Copies the directory structure of the "raw" image directory.
            2. Resizes each image and saves them in the corresponding
               "uniform" subdirectory.
    '''

    def __init__(self, path='/Users/kuoyen/Documents/myweddingstyle/images',
                        raw_dir='raw', uniform_dir='uniform_100_v2'):
        self.raw_dir_path = os.path.join(path, raw_dir)
        self.uniform_dir_path = os.path.join(path, uniform_dir)
        self.raw_subdirs = os.listdir(self.raw_dir_path)
        if not os.path.isdir(self.uniform_dir_path):
            os.mkdir(self.uniform_dir_path) 

    def copy_directory(self):
        '''
        Create a copy of the raw image directory. Check to see if it
        already exists. Preserves the same naming convention as the raw
        image directory except that the root directory is "uniform".
        '''
        for raw_subdir in self.raw_subdirs:
            uniform_subdir = os.path.join(self.uniform_dir_path, raw_subdir)
            if not os.path.isdir(uniform_subdir):
                os.mkdir(uniform_subdir)

    def standardize(self, n=1000, size=(100, 100)):
        print "standardizing to:", size 
        raw_img_subdir = []
        for raw_subdir in self.raw_subdirs: 
            if not raw_subdir.startswith('.'):
                raw_img_subdir.append(raw_subdir)
        for i in raw_img_subdir: 
            uniform_subdir = os.path.join(self.uniform_dir_path, i)
            self.raw_subdir_path = os.path.join(self.raw_dir_path, i)
            self.raw_sub = os.listdir(self.raw_subdir_path)
            self.raw_sub = self.raw_sub[:n]  
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
            print 'Image is black and white: ', file_img
        
if __name__ == '__main__':
    ss = StandardizeSize()
    ss.copy_directory()
    print ss.standardize(n=1000)