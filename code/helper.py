import os
import errno
from skimage import io

'''
DOC:
This file contains general functions.
'''


def write_image(img_matrix, filename):
    io.imsave(filename, img_matrix)


def mkdir_p(path):
    try:
        os.mkdir(path)
    except OSError as exc:
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass


def pad_list(lst):
    inner_max_len = max(map(len, lst))
    map(lambda x: x.extend([0]*(inner_max_len-len(x))), lst)
    return np.array(lst)


def star(n):
    print ''.join(['-'] * n)


def delete_files():
    folder = '/Users/kuoyen/Documents/myweddingstyle/images/transformed_1'
    for the_file in os.listdir(folder):
        file_path = os.path.join(folder, the_file)
        try:
            if os.path.isfile(file_path):
                os.unlink(file_path)
        except Exception, e:
            print e
