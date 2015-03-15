__author__ = 'kuoyen'

import scipy
from PIL import Image
from scipy.cluster.vq import vq
from image_loader import ImageLoader
import numpy as np
from scipy.misc import fromimage
import pickle as pkl
from helper import star


class DominantColor(object):

    def __init__(self, img_num, root_dir='/Users/kuoyen/Documents/myweddingstyle/images/uniform_100'):
        il = ImageLoader(root_dir, img_num=img_num, segment=False)
        il.load_images()
        self.all_files = il.all_files
        self.all_seg_files = il.all_seg_files
        self.all_img_mat = il.img_label_pair[0]
        self.label_vector = il.label_vector

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
        index_top = scipy.argsort(counts)[:-8:-1]           # find most frequent in desc order
        # print "length of ncolor", len(index_top)
        img_dom_colors = []
        for i in codes[index_top]:
            img_dom_colors.append(i)
            # colour = ''.join(chr(c) for c in i).encode('hex')
            # print 'most frequent is %s (#%s)' % (i, colour)
        dom_colors.append(img_dom_colors)

        return dom_colors

    def _seg_dom_color(self, fname_tup, k, ncolor):
        """Return all the dom colors in all the fnames as a 0 dimensional array"""
        all_seg_dom_colors = []
        for fname in fname_tup:
            # Get dom color
            seg_dom_colors = self._dom_color(fname, k=k, ncolor=ncolor)
            # Make sure dom color is of a certain shape
            seg_dom_colors = self._reshape_segment(seg_dom_colors, ncolor)
            all_seg_dom_colors.append(seg_dom_colors)
        return all_seg_dom_colors

    @staticmethod
    def _reshape_segment(seg_dom_colors, ncolor):
        """Make sure the dom color is of certain shape"""
        seg_dom_colors = seg_dom_colors[0]
        num_seg_colors = len(seg_dom_colors)
        if num_seg_colors != ncolor and num_seg_colors:
            diff = ncolor - num_seg_colors
            print "IMG has less than 3 dom colors by:", diff
            first_color_lst = [seg_dom_colors[0]]
            pad_colors = first_color_lst * diff
            first_color_lst.extend(pad_colors)
            return first_color_lst
        elif not num_seg_colors:
            # if len(seg_dom_colors) == 0
            print 'No dom color found. Returning a list of whites'
            return [np.array([255, 255, 255])] * ncolor
        return seg_dom_colors

    def run_dom_colors(self, k, ncolor, segment, pkl_fname):
        """
        :param k:
        :param ncolor:
        :param segment: BOOLEAN if you wanna run dom color on segments of an image or not
        :return:
        """
        all_color_lst = []
        if not segment:
            for ith_file, fname in enumerate(self.all_files):
                if ith_file == 0 or ith_file % 1000 == 0:
                    print 'Completed', ith_file
                img_dom_colors = self._dom_color(fname, k, ncolor)
                all_color_lst.append(np.array(img_dom_colors).ravel()[np.newaxis, :])

        if segment:
            for ith_file, fname_tup in enumerate(self.all_seg_files):
                if ith_file == 0 or ith_file % 1000 == 0:
                    print 'Completed', ith_file
                img_dom_colors = self._seg_dom_color(fname_tup, k, ncolor)
                # print img_dom_colors
                append = np.array(img_dom_colors).ravel()[np.newaxis, :]
                all_color_lst.append(append)
                #break point condition: np.shape(append)[1] != 27

        # print all_color_lst
        all_color_mat = np.concatenate(all_color_lst, axis=0)
        star(30)
        print 'DONE!!!!', all_color_mat.shape
        star(30)
        pkl.dump(all_color_mat, open(pkl_fname, 'wb'))
        return all_color_mat

if __name__ == '__main__':
    dc = DominantColor(1000)
    dc.run_dom_colors(10, 7, False, 'raw_dom_color.pkl')
    # dc.run_dom_colors(5, 3, True, 'seg_dom_color.pkl')



