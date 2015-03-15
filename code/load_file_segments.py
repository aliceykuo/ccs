import sys
import os
import numpy as np
# import itertools
from sklearn.preprocessing import StandardScaler
from skimage import io
import scipy
from scipy import ndimage, misc
from skimage.color import rgb2gray
from skimage.exposure import equalize_hist
from skimage.segmentation import felzenszwalb, slic, quickshift
from PIL import Image
from scipy.cluster.vq import vq
from sklearn.cross_validation import train_test_split
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score
from sklearn.ensemble import RandomForestClassifier
from sklearn import preprocessing
import pickle as pkl
from dom_color import DominantColor
import cv2
import time
import datetime
import os, errno

def mkdir_p(path):
    try:
        os.makedirs(path)
    except OSError as exc: # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else: raise

def write_image(img_matrix, filename, label_int):
    io.imsave(filename, img_matrix)
    # print "wrote file", filename

    # scipy.misc.imsave(filename, img_matrix)
    # return img_matrix, filename, label_int
    # pass

def filter_function(img, filter = ''):
    if filter == 'hist':
        filtered_image = equalize_hist(img)
    elif filter == 'background':
        filtered_image = bkg()
    elif filter == 'segment':
        filtered_image = segments_slic = slic(img, n_segments=3, compactness=10, sigma=1)
    return filtered_image

# def filter_function(img, filter = ''):
#     w_img = whiten(img)
#     if filter == 'hist':
#         filtered_image = equalize_hist(w_img)
#     elif filter == 'segment':
#         filtered_image = segments_slic = slic(w_img, n_segments=3, compactness=10, sigma=1)
#     elif filter == 'sift':
#         filtered_image = sift(w_img)
#     elif filter == 'segment':
#         filtered_image = segment(w_img)
#     return filtered_image


def whiten(img):
    mu = np.mean(img)
    std = np.std(img)
    img = (img - mu) / std
    return img

# def sift(img):
#     gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#     sift = cv2.SIFT()
#     kp, des = sift.detectAndCompute(gray,None)
#     img = cv2.drawKeypoints(gray, kp)
#     return img, des
    #des returns matrix of a different shape than original 

def pad_list(lst):
    inner_max_len = max(map(len, lst))
    map(lambda x: x.extend([0]*(inner_max_len-len(x))), lst)
    return np.array(lst)

class ImageLoader(object):
    def __init__(self, root_dir='.', n=5, img_callback = write_image, filter_func = filter_function):
        self.root_dir = root_dir
        self.n = n
        self.img_callback = img_callback
        self.filter_func = filter_function
        self.all_files = None
        self.all_files_transformed = None
        self.label_vector = None

    def all_image_paths(self):
        ''' Extract all subfolders with images and return all image files and their labels. '''
        self.all_files = []
        labels = [i for i in (self.get_immediate_subdirectories(self.root_dir)) 
                    if not i.startswith('.')]
        for root, subFolders, files in os.walk(self.root_dir):
            files = [i for i in files if not i.startswith('.')]
            files = files[:2] #hard coded - will not read in
            for i in files:
                self.all_files.append(os.path.abspath(root) + '/'.join(subFolders) + '/' + i)
        return self.all_files,labels

    def get_immediate_subdirectories(self, a_dir):
        return [name for name in os.listdir(a_dir)
            if os.path.isdir(os.path.join(a_dir, name))]

    def get_label(self, path):
        split = path.split('/')
        return split[-2]

    def load_image(self, file_img, size = (28,28)):
        # img = scipy.ndimage.imread(file_img)
        img = io.imread(file_img)
        # img = img.resize(size)
        return img
        # img_read = io.imread('/Users/kuoyen/Documents/capstone/images/uniform/ballroom/ballroom0.jpg')
        
    def load_images(self):
        '''
        Return a tuple containing list of pairs of: numpy array and integer label 
        and dictionary mapping integer label to label name
        '''
        self.all_files, labels = self.all_image_paths()
        print "number of files processing", len(self.all_files)
        self.label_ints = {}
        # change label list to label_ints dictionary
        for i in xrange(len(labels)):
            # ie. { 'beach': 0, 'rustic': 1 }
            self.label_ints[labels[i]] = i
        self.img_label_pair = []
        self.label_vector = []
        for file_img in self.all_files:
            #/Users/kuoyen/Documents/capstone/images/uniform/ballroom/ballroom0.jpg
            img = self.load_image(file_img)
            label = self.get_label(file_img)
            label_int = self.label_ints[label]
            self.label_vector.append(label_int)
            self.img_label_pair.append((img,label_int))
            self.img_callback(img, file_img, label_int)
        # self.label_vector = np.array(self.label_vector)[:,np.newaxis]
        print "completed load_images"
        return self.img_label_pair,self.label_ints

    def dom_colors(self, k=10, files='raw_files'):
        dom_colors =[]
        if files == 'raw_files':
            file_set = self.all_files
        elif files == 'transformed':
            file_set = self.all_files_transformed
        for i in file_set:
            # print "processing dom_colors", i
            img = Image.open(i)
            # segments = segment_image(i)
            ar = scipy.misc.fromimage(img)
            shape = ar.shape
            ar = ar.reshape((scipy.product(shape[:2]), shape[2]))
            codes, dist = scipy.cluster.vq.kmeans(ar, k)
            vecs, dist = scipy.cluster.vq.vq(ar, codes)         
            counts, bins = scipy.histogram(vecs, len(codes))   
            index_top = scipy.argsort(counts) [:-6:-1]         
            img_dom_colors = []
            for i in codes[index_top]:
                img_dom_colors.append(i)
                colour = ''.join(chr(c) for c in i).encode('hex')
                # print 'most frequent is %s (#%s)' % (i, colour)
            # print "#######img_dom_colors", img_dom_colors
            dom_colors.append(img_dom_colors)
        return dom_colors


    def dom_colors_by_segment(self, k=3):
        self.seg_dom_colors_mat = None
        # (3, 45) 
        for segment_set in self.all_segmented_filenames:
            # print segment_set
            dom_colors =[] 
            #dom_colors is a nested list of lists of length (the number of samples)
            # each is an array of 3 sets
            for segment in segment_set:
                # print "processing segment dom colors:", segment
                #segment is the image segment that is being passed through
                img = Image.open(segment)
                ar = scipy.misc.fromimage(img)
                shape = ar.shape
                ar = ar.reshape((scipy.product(shape[:2]), shape[2]))
                codes, dist = scipy.cluster.vq.kmeans(ar, k)
                vecs, dist = scipy.cluster.vq.vq(ar, codes)         
                counts, bins = scipy.histogram(vecs, len(codes))   
                index_top = scipy.argsort(counts) [:-4:-1]         
                img_dom_colors = []
                #img_dom_colors is the k (5) arrays of dominant colors detected in each image (segment)
                for i in codes[index_top]:
                    img_dom_colors.append(i)
                    colour = ''.join(chr(c) for c in i).encode('hex')
                    # print 'most frequent is %s (#%s)' % (i, colour)
                # print "THE LENGTH OF img_dom_colors", img_dom_colors

                if len(img_dom_colors) != k:
                    diff = k - len(img_dom_colors)
                    print "IMG has less than 3 dom colors", diff, segment
                    pad_lst = ([(sum(img_dom_colors[0])/3)]*3) * diff
                    img_dom_colors.extend(pad_lst) 
                dom_colors.append(img_dom_colors)
            dom_colors = pad_list(dom_colors)
            print "dom_color len", len(dom_colors), segment_set
            dom_colors = np.array(dom_colors).ravel()[np.newaxis, :]
            # print "dom_color dom_colors.shape
            if self.seg_dom_colors_mat is None:
                self.seg_dom_colors_mat = dom_colors
            else: 
                try:
                    self.seg_dom_colors_mat = np.concatenate((self.seg_dom_colors_mat, dom_colors),  axis=0)
                except Exception, e:
                    print e
                    continue
        return self.seg_dom_colors_mat

    def pre_trans(self):
        dom_colors = self.dom_colors()
        self.pre_feat_mat = None 
        for dom_col in dom_colors:
            dom_col = np.array(dom_col).ravel()[np.newaxis, :]
            if self.pre_feat_mat is None:
                self.pre_feat_mat = dom_col
            else: 
                self.pre_feat_mat = np.concatenate((self.pre_feat_mat,dom_col),  axis=0)
        print "pre_trans_dimensions", self.pre_feat_mat.shape
        return self.pre_feat_mat 

    def sift(self):
        print "starting to run sift"
        sift_all = []
        for i, pair in enumerate(self.img_label_pair):
            file_img = pair[0]
            sift_arr = []
            gray = cv2.cvtColor(file_img, cv2.COLOR_BGR2GRAY)
            sift = cv2.SIFT()
            kp, des = sift.detectAndCompute(gray,None)
            img = cv2.drawKeypoints(gray, kp)
            if des is None:
                des = [0]
                sift_all.extend([0])
                print "des empty for", i
            else:
                for arr in des:
                    sift_arr.extend(arr)
                sift_all.append(sift_arr)
        sift_padded = pad_list(sift_all)
        # print "___________________________", sift_padded.shape 
        print "completed sift"
        return sift_padded

    def segment_image(self):
        self.all_segmented_filenames = []
        for i, pair in enumerate(self.img_label_pair):
            file_img = pair[0]
            label_int = pair[1]
            segments = []
            seg_1 = file_img[:33,:]
            seg_2 = file_img[33:67,:]
            # print "***** segment 2", len(file_img[10:20,:])
            seg_3 = file_img[67:,:]
            # print "***** segment 3", len(file_img[20:,:])
            segments.append(seg_1)
            segments.append(seg_2)
            segments.append(seg_3)
            segmented_filenames = []
            for ind, seg in enumerate(segments):
                filename = '/Users/kuoyen/Documents/capstone/images/segmented_1/'+ str(i) + 'seg' + str(ind) + '.jpg'
                # print "segmented filename", filename
                self.img_callback(seg, filename, label_int)
                segmented_filenames.append(filename)
            self.all_segmented_filenames.append(segmented_filenames)
        segment_colors = self.dom_colors_by_segment()
        #returned after dom_colors_by_segment runs
        print "+++++++++++++ seg_dom_colors_mat.shape", self.seg_dom_colors_mat.shape 
        #+++++++++++++ seg_dom_colors_mat.shape (3996, 27)
        return self.seg_dom_colors_mat
        # return self.seg_dom_colors
        # return self.all_segmented_filenames

    def transform_filter(self, img_filter = 'hist'):
        self.all_files_transformed = []
        self.transformed_img_mat = None
        sift_all = []
        for i, pair in enumerate(self.img_label_pair):
            file_img = pair[0]
            label_int = pair[1]
            filtered_img = self.filter_func(file_img, img_filter)
            #also return img mat after transformation 
            filtered_img_mat = np.array(filtered_img).ravel()[np.newaxis, :]  
            # sift_arr = self.sift(file_img) 
            # sift_all.append(sift_arr)
            if self.transformed_img_mat is None:
                self.transformed_img_mat = filtered_img_mat
            else:
                self.transformed_img_mat = np.concatenate((self.transformed_img_mat, filtered_img_mat), axis=0)
            filename = '/Users/kuoyen/Documents/capstone/images/transformed_1/filtered_' + str(label_int) + '_'+str(i)+'.jpg'
            self.all_files_transformed.append(filename)
            self.img_callback(file_img, filename, label_int)
        print "post tranformed image matrix shape", self.transformed_img_mat.shape
        # return self.all_files_transformed, self.transformed_img_mat
        return self.transformed_img_mat


    # def transform_filter(self, img_filter = 'hist'):
    #     self.all_files_transformed = []
    #     self.transformed_img_mat = None
    #     for i, pair in enumerate(self.img_label_pair):
    #         file_img = pair[0]
    #         filtered_img = self.filter_func(file_img, img_filter)
    #         #also return img mat after transformation 
    #         filtered_img_mat = np.array(filtered_img).ravel()[np.newaxis, :]        
    #         if self.transformed_img_mat is None:
    #             self.transformed_img_mat = filtered_img_mat
    #         else:
    #             self.transformed_img_mat = np.concatenate((self.transformed_img_mat, filtered_img_mat), axis=0)
    #         label_int = pair[1]
    #         filename = '/Users/kuoyen/Documents/capstone/images/transformed/filtered_' + str(label_int) + '_'+str(i)+'.jpg'
    #         self.all_files_transformed.append(filename)
    #         self.img_callback(file_img, filename, label_int)
    #     print "***********"
    #     print "post tranformed image matrix shape", self.transformed_img_mat.shape
    #     return self.all_files_transformed, self.transformed_img_mat


    def post_trans(self):
        # self.post_feat_mat = np.zeros(shape=(6,27))
        segment_colors_mat = self.segment_image()
        print "segment_colors_mat", segment_colors_mat.shape
        sift_mat = self.sift()
        print "sift shape", sift_mat.shape
        # dom_colors = self.dom_colors(files='transformed')
        self.transformed_img_mat = self.transform_filter()

        self.post_feat_mat = None
        dom_colors = self.dom_colors(files='transformed')
        for dom_col in dom_colors:
            dom_col = np.array(dom_col).ravel()[np.newaxis, :]
            print "dom_col shape", dom_col.shape
            if self.post_feat_mat is None:
                self.post_feat_mat = dom_col
            else: 
                self.post_feat_mat = np.concatenate((self.post_feat_mat, dom_col),  axis=0)
        print "self.post_feat_mat shape", self.post_feat_mat.shape
        self.post_feat_mat = np.concatenate((self.post_feat_mat, segment_colors_mat, sift_mat), axis=1)
        print "~~~~~~~~~~~~~~~~~~~~~"
        # print "self.transformed_img_mat shape:", self.transformed_img_mat.shape
        # print "sift_mat shape:", sift_mat.shape
        # self.post_feat_mat = np.concatenate((self.post_feat_mat, self.transformed_img_mat, sift_mat), axis=1)
        # self.post_feat_mat = np.concatenate((self.post_feat_mat, sift_mat), axis=1)
        print "complete post_trans"
        print "self.post_mat shape:", self.post_feat_mat
        return self.post_feat_mat 


    def create_feat_matrix(self):
        self.full_feat_mat = np.concatenate((self.pre_feat_mat, self.post_feat_mat), axis=1)
        print "creating matrix"
        print "full_mat_shape", self.full_feat_mat.shape
        return self.full_feat_mat

    def standard_scaler(self):
        self.scaler = StandardScaler()
        self.X = self.scaler.fit_transform(self.full_feat_mat)
        # self.X = preprocessing.scale(self.full_feat_mat)
        # self.X = self.full_feat_mat 
        # y = self.label_vector
        self.y = np.array(self.label_vector)[:,np.newaxis]
        # y = np.array(self.label_vector)[np.newaxis,:]
        print "********"
        print "X shape", self.X.shape
        print "y shape", self.y.shape
        return self.X, self.y
    
    def save_pkl(self, size=28, samples='1' ,scaled = 'scaled', colors='k5'):
        data = np.concatenate((self.y, self.X), axis =1)
        pkl_path = '../images/pkl/mat' + '_' + str(size) + '_'+ samples + '_'+ scaled  + '_' + colors + 'seg_sift_0311_0148.pkl'
        pkl.dump(data, open(pkl_path, 'wb'))
        print "finished pickling", pkl_path

    def dropout(self, post_feat_mat, p, output):
        vlength = len(post_feat_mat)
        d = np.random.binomial(1, n, vlength)
        dropout_arr = post_feat_mat * d
        return dropout_arr

    def delete_files(self):
        folder = '/Users/kuoyen/Documents/capstone/images/transformed_1'
        for the_file in os.listdir(folder):
            file_path = os.path.join(folder, the_file)
            try:
                if os.path.isfile(file_path):
                    os.unlink(file_path)
            except Exception, e:
                print e

if __name__ == '__main__':
   loader = ImageLoader(sys.argv[1], sys.argv[2])
   loader.delete_files()
   # ts = time.time()
   # st = datetime.datetime.fromtimestamp(ts).strftime('%m%d_%H%M')
   # self.path = '/Users/kuoyen/Documents/capstone/images/transformed' + st
   # trans_dir = mkdir_p(path)
   # print "print path", path
    #ipython load_file.py /Users/kuoyen/Documents/capstone/images/uniform/ 1
    #ipython load_file.py /Users/kuoyen/Documents/capstone/images/uniform_60 1
   loader.all_image_paths()
   loader.load_images()
   loader.pre_trans()
   loader.post_trans()
   loader.create_feat_matrix()
   # X, y = loader.standard_scaler()
   loader.standard_scaler()
   loader.save_pkl(size=100, samples='test', scaled='stdscaled', colors='k5')
  #  # loader.run_model(X, y)
  #  mat_60_50 = np.concatenate((y, X), axis =1)
  #  pkl.dump(mat_60_50, open('../images/pkl/mat_60_50_nonscaled.pkl', 'wb'))
  #  print "finished pickling ../images/pkl/mat_60_50_noscaled.pkl"
  # 