import numpy as np
from dom_color import DominantColor
from sklearn.feature_extraction import image
from skimage import io


class ExtractPatches(object):
    '''
    DOC:
    ExtractPatches converts an image into smaller patches determined by
    each patch's size and the number of patches desired.
    '''

    def __init__(self, fnames):
        self.all_files = fnames
        dc = DominantColor(self.all_files)
        self.all_colors_mat = dc.run_dom_colors(10, 5)

    def extract_patch(self, img, patch_size=(20, 20), max_patches=25,
                      dominant_color_vector=None):
        img = io.imread(img)
        if dominant_color_vector is None:
            raise Exception('No dominant colors found.')
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
        print "all colors mat shape", self.all_colors_mat.shape

        all_img_patches = []
        for img, dom_color_vec in zip(self.all_files, self.all_colors_mat):
            img_patches = self.extract_patch(img=img, dominant_color_vector=dom_color_vec)
            all_img_patches.append(np.array(img_patches))

        feature_mat = np.concatenate(all_img_patches, axis=0)
        print "FEATURE MATRIX COMPLETED:", feature_mat.shape
        return feature_mat

if __name__ == '__main__':
    ep = ExtractPatches(['/Users/kuoyen/Documents/myweddingstyle/images/uniform_100/'])
    ep.run_extract_patch()
