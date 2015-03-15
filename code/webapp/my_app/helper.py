__author__ = 'kuoyen'

import os
import errno
from skimage import io


def write_image(img_matrix, filename):
    io.imsave(filename, img_matrix)


def mkdir_p(path):
    try:
        os.mkdir(path)
    except OSError as exc:
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass

def star(n):
    print ''.join(['-'] * n)
# def filter_function(img, filter='hist'):
#     if filter == 'hist':
#         filtered_image = equalize_hist(img)
#     elif filter == 'background':
#         filtered_image = bkg()
#     elif filter == 'segment':
#         filtered_image = slic(img, n_segments=3, compactness=10, sigma=1)
#     return filtered_image
#
# def whiten(img):
#     mu = np.mean(img)
#     std = np.std(img)
#     img = (img - mu) / std
#     return img
#
# #
# def pad_list(lst):
#     inner_max_len = max(map(len, lst))
#     map(lambda x: x.extend([0]*(inner_max_len-len(x))), lst)
#     return np.array(lst)