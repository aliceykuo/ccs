import numpy as np
import pickle as pkl
from dom_colors import DominantColor
from sklearn.feature_extraction import image
from skimage import io
from sklearn.preprocessing import StandardScaler


class ExtractPatches(object):
    '''
    DOC:
    ExtractPatches converts an image into smaller patches determined by
    each patch's size and the number of patches desired.
    '''

    def __init__(self, img_num, root_dir='/Users/kuoyen/Documents/myweddingstyle/images/uniform_100'):
        dc = DominantColor(img_num, root_dir)
        self.all_colors_mat = dc.run_dom_colors(10, 5, False, 'raw_dom_color.pkl')
        self.all_files = dc.all_files
        self.all_img_mat = dc.all_img_mat[0]
        self.label_vector = dc.label_vector

    def extract_patch(self, img, patch_size=(20, 20), max_patches=25, dominant_color_vector=None):
        img = io.imread(img)
        if dominant_color_vector is None:
            raise 'No dominant colors found!'
        patches = image.extract_patches_2d(img, patch_size, max_patches)
        slices = max_patches
        arr = []
        for i in xrange(slices):
            patch = np.concatenate(patches[i].ravel()[np.newaxis, :], axis=1)
            dom_col_patch = np.concatenate((dominant_color_vector, patch), axis=1)
            arr.append(dom_col_patch)
        arr = np.array(arr)
        return arr

    def run_extract_patch(self):
        print "Number of files processing", len(self.all_files)
        self.label_vector = np.array(self.label_vector)[:, np.newaxis]
        label_color_mat = np.concatenate((self.label_vector, self.all_colors_mat), axis=1)
        all_img_patches = []
        for img, dom_color_vec in zip(self.all_files, label_color_mat):
            img_patches = self.extract_patch(img=img, dominant_color_vector=dom_color_vec)
            all_img_patches.append(np.array(img_patches))

        self.feature_mat = np.concatenate(all_img_patches, axis=0)
        print "COMPLETED FEATURE MATRIX:", self.feature_mat.shape
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

if __name__ == '__main__':
    ep = ExtractPatches(1000)
    ep.run_extract_patch()
    # ep.standard_scaler()
    ep.save_pkl()