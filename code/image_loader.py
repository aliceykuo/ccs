
from skimage import io
import os
from helper import mkdir_p, write_image


class ImageLoader(object):
    def __init__(self, root_dir='.', img_num=5, segment=True, img_callback=write_image):
        """
        :param root_dir: Parent dir of raw image files
        :param img_num: Number of images run per category (beach / rustic)
        """
        self.root_dir = root_dir
        self.segment_dir = '%s_segment' % root_dir

        self.img_num = img_num
        self.all_files = []
        self.all_seg_files = []
        self.labels = []
        self.all_files_transformed = None
        self.label_vector = None
        self.segment = segment
        self.img_label_pair = []
        self.img_callback = img_callback

    def get_immediate_subdirectories(self, a_dir):
        """Returning file names within dir"""
        return [name for name in os.listdir(a_dir)
                if os.path.isdir(os.path.join(a_dir, name))]

    def all_image_paths(self):
        """
        Go into subdirectory of the category (beach /rustic) and return all the filenames
        and category for each subdir
        """
        self.labels = [i for i in (self.get_immediate_subdirectories(self.root_dir))
                       if not i.startswith('.')]

        for root, subFolders, files in os.walk(self.root_dir):
            files = [i for i in files if not i.startswith('.')]
            files = files[:self.img_num]  # hard coded - will not read in
            for i in files:
                self.all_files.append(os.path.abspath(root) + '/'.join(subFolders) + '/' + i)

    def get_label(self, path):
        split = path.split('/')
        return split[-2]

    def load_image(self, file_img):
        img = io.imread(file_img)
        return img

    def load_images(self, img_dim=(100, 100), n_segments=3):
        """
        Return a tuple containing list of pairs of: numpy array and integer label
        """
        self.all_image_paths()
        print "number of files processing", len(self.all_files)
        label_ints = {}
        # change label list to label_ints dictionary
        for i in xrange(len(self.labels)):
            # ie. { 'beach': 0, 'rustic': 1 }
            label_ints[self.labels[i]] = i
        self.label_vector = []
        # Processing raw file (not segmented)
        for file_img in self.all_files:
            # path = /Users/kuoyen/Documents/capstone/images/uniform/ballroom/ballroom0.jpg
            #/Users/kuoyen/Documents/capstone/images/uniform/ballroom/ballroom0.jpg
            img = self.load_image(file_img)
            label = self.get_label(file_img)
            label_int = label_ints[label]
            self.label_vector.append(label_int)
            self.img_label_pair.append((img, label_int))  # list of tuples of (np.array, int_label)
        print "completed load_images (without segmenting)"

        # Segment images and store in new directory if necessary
        if self.segment:
            self.segment_image(img_dim=img_dim, n_segments=n_segments)
        print "completed load_images (with segmenting)"


    def segment_image(self, img_dim=(100, 100), n_segments=3):
        # create segmented image parent folder if it does not already exist
        mkdir_p(self.segment_dir)

        for ith_img, tup in enumerate(self.img_label_pair):
            file_img, label_int = tup
            img_height, img_width = img_dim
            seg_h = img_height / n_segments

            segments = []
            seg_1 = file_img[:seg_h, :]
            seg_2 = file_img[seg_h:(seg_h * 2), :]
            seg_3 = file_img[(seg_h * 2):, :]

            segments.append(seg_1)
            segments.append(seg_2)
            segments.append(seg_3)

            segmented_filenames = []
            for ith_seg, seg in enumerate(segments):
                seg_fname = os.path.join(self.segment_dir, '%d_%d.jpg' % (ith_img, ith_seg))
                segmented_filenames.append(seg_fname)
                self.img_callback(seg, seg_fname)
            self.all_seg_files.append(segmented_filenames)


if __name__ == '__main__':
    il = ImageLoader(root_dir='/Users/kuoyen/Documents/myweddingstyle/images/uniform_100',
                     img_num=1000, segment=False)
    il.load_images()
