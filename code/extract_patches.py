import scipy
from PIL import Image
from image_loader import ImageLoader
import numpy as np
import pickle as pkl
from helper import star
from dom_colors import DominantColor
from sklearn.feature_extraction import image
from skimage import io
from sklearn.preprocessing import StandardScaler


class ExtractPatches(object):

    def __init__(self, img_num, root_dir='/Users/kuoyen/Documents/myweddingstyle/images/uniform_100'):
        dc = DominantColor(img_num, root_dir)
        self.all_colors_mat = dc.run_dom_colors(10, 5, False, 'raw_dom_color.pkl')
        self.all_files = dc.all_files
        self.all_img_mat = dc.all_img_mat[0]
        self.label_vector = dc.label_vector


    def extract_patch(self, img, patch_size = (20,20), max_patches = 25, dominant_color_vector = None):
        img = io.imread(img)
        if dominant_color_vector is None:
            raise 'No dominant colors found!'
        patches = image.extract_patches_2d(img, patch_size, max_patches)
        slices = max_patches
        arr = []
        for i in xrange(slices):
            # print  "patches[i].ravel() shape", patches[i]
            # print "completed print patches"
            patch = np.concatenate(patches[i].ravel()[np.newaxis, :], axis=1)
            dom_col_patch = np.concatenate((dominant_color_vector, patch), axis=1 )
            # print "printing patch type ", type(patch)
            # print "$$$$$$", type(dominant_color_vector)
            arr.append(dom_col_patch)
        arr = np.array(arr)
        print "@@@@@@@@@@@", arr
        print np.array(arr).shape
        #     arr.append(np.concatenate(arr[np.newaxis, :], axis=1))
        # # arr = np.array(arr)
        # # print arr.shape
        return arr

    def run_extract_patch(self):
        print "********* number of files processing", len(self.all_files)
        print "all colors mat shape", self.all_colors_mat.shape
        self.label_vector = np.array(self.label_vector)[:,np.newaxis]
        label_color_mat = np.concatenate((self.label_vector, self.all_colors_mat), axis = 1)
        # print "@@@", label_color_mat
        all_img_patches = []
        for img, dom_color_vec in zip(self.all_files, label_color_mat):
            print "________________________________________________"
            img_patches = self.extract_patch(img = img, dominant_color_vector = dom_color_vec)
            print "@@@@@", img_patches.shape
            all_img_patches.append(np.array(img_patches))

        # feat_matrix = np.append((np.array(all_img_patches)), axis =0)
        #     if feat_matrix is None:
        #         feat_matrix = img_patches
        #     else:
        #         feat_matrix = np.concatenate((feat_matrix, img_patches), axis =0)
        # print feat_matrix.shape
        self.feature_mat = np.concatenate(all_img_patches, axis=0)
        print "DONE with FEATURE MATRIX:", self.feature_mat.shape
        return self.feature_mat

    def standard_scaler(self):
        scaler = StandardScaler()
        self.feat_mat_scaled = self.scaler.fit_transform(self.feature_mat)
        print "scaled matrix"
        return self.feature_mat_scaled

    def save_pkl(self):
        pkl_path = '/Users/kuoyen/Desktop/wedding/extract_patches25_1000img_7dom_0145.pkl'
        pkl.dump(self.feature_mat_scaled, open(pkl_path, 'wb'))
        print "finished pickling", pkl_path

    # self.y = np.array(self.label_vector)[:,np.newaxis]

if __name__ == '__main__':
    ep = ExtractPatches(1000)
    ep.run_extract_patch()
    ep.standard_scaler()
    ep.save_pkl()


