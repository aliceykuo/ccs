import scipy
from PIL import Image
from scipy.cluster.vq import vq
from image_loader import ImageLoader
import numpy as np
from scipy.misc import fromimage
import pickle as pkl
from helper import star


class DominantColor(object):

    def __init__(self, fnames):
        self.all_files = fnames


    @staticmethod
    def _dom_color(fname, k, ncolor):
        dom_colors = []
        img = Image.open(fname)
        ar = fromimage(img)
        shp = ar.shape
        ar = ar.reshape((scipy.product(shp[:2]), shp[2]))
        codes, dist = scipy.cluster.vq.kmeans(ar, k)
        vecs, dist = scipy.cluster.vq.vq(ar, codes)         # assign codes
        counts, bins = scipy.histogram(vecs, len(codes))    # count occurrences
        index_top = scipy.argsort(counts)[:-(ncolor + 1):-1]           # find most frequent in desc order
        img_dom_colors = []
        for i in codes[index_top]:
            img_dom_colors.append(i)
            # colour = ''.join(chr(c) for c in i).encode('hex')
            # print 'most frequent is %s (#%s)' % (i, colour)
        dom_colors.append(img_dom_colors)

        return dom_colors


    def run_dom_colors(self, k, ncolor):
        """
        :param k:
        :param ncolor:
        :param segment: BOOLEAN if you want to run segments of an image or not
        :return:
        """
        all_color_lst = []

        for ith_file, fname in enumerate(self.all_files):
            if ith_file == 0 or ith_file % 1000 == 0:
                print 'Completed', ith_file
            img_dom_colors = self._dom_color(fname, k, ncolor)
            all_color_lst.append(np.array(img_dom_colors).ravel()[np.newaxis, :])

        # print all_color_lst
        all_color_mat = np.concatenate(all_color_lst, axis=0)
        star(30)
        print 'DONE!!!!', all_color_mat.shape
        star(30)
        return all_color_mat

if __name__ == '__main__':
    dc = DominantColor(['/Users/kuoyen/Documents/myweddingstyle/images/uniform_100/'])
    dc.run_dom_colors(10, 5)



