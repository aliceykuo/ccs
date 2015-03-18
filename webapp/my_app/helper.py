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


def pad_list(lst):
    inner_max_len = max(map(len, lst))
    map(lambda x: x.extend([0]*(inner_max_len-len(x))), lst)
    return np.array(lst)